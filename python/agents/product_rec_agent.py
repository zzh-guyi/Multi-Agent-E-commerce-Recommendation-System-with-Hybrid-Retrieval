"""
Product Recommendation Agent

职责：

1. Recall 模式

   UserProfile
       ↓
   MySQL
       ↓
   硬过滤：库存 + 价格
       ↓
   Query Rewrite
       ↓
   Embedding
       ↓
   ┌────────────────────────────┐
   │       Hybrid Search        │
   │                            │
   │ Vector Search              │
   │      +                     │
   │ Keyword Search             │
   └────────────┬───────────────┘
                ↓
           RRF Fusion
                ↓
           candidates


Rerank 模式

candidates
       |
       |
       +----------------+
       |                |
 enable_llm_rerank    disabled
       |                |
       ↓                ↓
 LLM Ranking       Recall Order
       |
       ↓
      TopN


重要：

ProductRecAgent 严格区分 Recall 和 Rerank。

candidates is None
    → Recall ONLY

candidates is not None
    → Rerank ONLY


Supervisor：

UserProfile
    ↓
ProductRecall
    ↓
┌──────────────┬──────────────┐
↓              ↓
Rerank       Inventory
└──────────────┴──────────────┘
              ↓
            Filter
              ↓
        MarketingCopy
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from config import get_settings

from models.schemas import (
    Product,
    ProductRecResult,
    UserProfile,
)

from .base_agent import BaseAgent

from services.product_store import ProductStore
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.metrics import metrics_collector


# ============================================================
# Query Rewrite Prompt
# ============================================================

QUERY_REWRITE_PROMPT = """
你是一个电商搜索 Query Rewrite 专家。

你的任务是根据用户画像，将用户的长期偏好、
最近行为和实时兴趣，转换成一个适合商品语义检索的
高质量搜索 Query。

用户画像：

{user_profile}

要求：

1. 保留用户明确的商品需求。
2. 优先考虑用户最近浏览、最近购买和实时兴趣。
3. 保留用户偏好的商品类别。
4. 不要把价格范围写入语义 Query。
5. 不要输出解释。
6. 不要输出 JSON。
7. 只输出一条适合 Embedding 的自然语言搜索 Query。
8. Query 应该描述用户真正想购买/寻找的商品。
9. 不要出现“用户偏好类别”“最近浏览”等字段名称。
10. 不要编造用户没有表现出的需求。

例如：

用户偏好：
运动、跑步、户外
最近浏览：
跑鞋、运动袜
实时兴趣：
马拉松

应该生成类似：

适合跑步和马拉松训练的专业运动鞋及相关跑步装备

只返回 Query 本身。
"""


# ============================================================
# LLM Rerank Prompt
# ============================================================

RERANK_PROMPT = """
根据用户偏好，从候选商品中选出最匹配的
{num_items}个商品并排序。

用户偏好：
{user_profile}

候选商品：
{candidates}

排序规则：
1. 用户偏好类目匹配程度
2. 商品名称与需求的相关程度
3. 用户最近浏览和购买行为
4. 商品之间保持一定类目多样性
5. 不要推荐明显重复的商品
6. 只能从候选商品中选择
7. 不要生成不存在的商品ID

价格和库存已由程序处理，无需判断。

只输出商品ID的JSON数组，不要输出其他文字。

例如：
["P001","P003","P007","P004","P006"]
"""

RERANK_TIMEOUT = 20.0


# ============================================================
# Tool Calling ReAct Prompt
# ============================================================

TOOL_REACT_SYSTEM_PROMPT = """
你是一个电商商品推荐专家。

你的任务是根据用户偏好，
使用工具搜索并筛选出最适合的商品。

可用工具：
- search_products(query, min_price, max_price, limit): 搜索商品，返回商品列表
  返回字段：product_id, name, category, price, brand, tags, stock, image_url
- get_product_detail(product_id): 查询单个商品的详细信息
  返回字段：product_id, name, category, price, description, brand, seller_id, stock, tags, image_url

执行步骤：
1. 先用 search_products 搜索与用户兴趣相关的商品（limit 建议 15-20）
2. search_products 已返回商品名称、类目、价格、品牌、标签、库存等完整推荐所需字段，
   通常不需要再调用 get_product_detail。
3. 仅在以下情况才调用 get_product_detail：
   - search_products 返回的商品数量太少（少于 5 个），需要补充更多候选
   - 用户明确要求了解某个商品的详细描述
   - 候选商品信息不足，必须查看 description 才能做出判断
4. get_product_detail 最多允许调用 3 个商品，不要对全部候选商品逐个查询。
5. 综合评估后，从候选商品中选择最匹配的 {num_items} 个商品
6. 最后只输出 JSON 数组格式的商品ID列表，例如：["P001","P002"]

注意：
- 必须通过工具获取真实数据，不能凭空编造商品ID
- 最终答案必须是 JSON 数组格式
- 不要因为"想查看更多信息"而调用 get_product_detail
- search_products 的返回结果已足够完成推荐任务
"""

MAX_TOOL_ITERATIONS = 3


class ProductRecAgent(BaseAgent):
    """
    商品推荐 Agent。

    两种执行模式：

    --------------------------------------------------------
    Recall 模式
    --------------------------------------------------------

    user_profile
         ↓
    MySQL
         ↓
    价格 + 库存硬过滤
         ↓
    Query Rewrite
         ↓
    Embedding
         ↓
    Vector Search
         +
    Keyword Search
         ↓
    RRF Fusion
         ↓
    candidates

    --------------------------------------------------------
    Rerank 模式
    --------------------------------------------------------

    candidates
         ↓
    LLM Rerank
         ↓
    TopN
    """

    def __init__(self, enable_tool_calling: bool = False):
        settings = get_settings()
        self.enable_llm_rerank = settings.enable_llm_rerank
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )

        # ====================================================
        # Tool Calling
        # ====================================================

        self.enable_tool_calling = enable_tool_calling

        if enable_tool_calling:
            from tools.product_tools import (
                search_products,
                get_product_detail,
            )
            from tools.inventory_tools import check_stock

            self.tools = [
                search_products,
                get_product_detail,
                check_stock,
            ]

            self._search_products_tool = search_products
            self._get_product_detail_tool = get_product_detail
            self._check_stock_tool = check_stock

        else:
            self.tools = []

        # ====================================================
        # LLM
        # ====================================================

        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0,
            max_tokens=300,
            max_retries=0,
        )

        # ====================================================
        # MySQL
        # ====================================================

        self.product_store = ProductStore()

        # ====================================================
        # Embedding
        # ====================================================

        self.embedding_service = EmbeddingService()

        # ====================================================
        # Milvus
        # ====================================================

        self.vector_store = VectorStore()

        # ====================================================
        # LLM with Tools
        # ====================================================

        if self.enable_tool_calling:
            self.llm_with_tools = ChatOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                temperature=0,
                max_tokens=300,
                max_retries=0,
            ).bind_tools(self.tools)

        else:
            self.llm_with_tools = None

        self._recall_metrics: dict[str, int] = {}

    # ========================================================
    # Metrics
    # ========================================================

    def _record_tool_metric(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        skipped: bool = False,
    ):
        try:
            metrics_collector.record_tool_call(
                tool_name,
                success,
                latency_ms,
                skipped,
            )
        except Exception as exc:
            print(
                f"[ProductRecAgent] "
                f"Warning: record_tool_metric failed: {exc}"
            )

    def _record_recall_metric(
        self,
        latency_ms: float,
        success: bool,
        iterations: int,
        tool_calls: int,
        sp_calls: int,
        dp_calls: int,
    ):
        try:
            metrics_collector.record_recall(
                latency_ms,
                success,
                iterations,
                tool_calls,
                sp_calls,
                dp_calls,
            )
        except Exception as exc:
            print(
                f"[ProductRecAgent] "
                f"Warning: record_recall failed: {exc}"
            )

    def _record_rerank_metric(
        self,
        latency_ms: float,
        success: bool,
        timeout: bool,
        fallback: bool,
        candidate_count: int,
        output_count: int,
    ):
        try:
            metrics_collector.record_rerank(
                latency_ms,
                success,
                timeout,
                fallback,
                candidate_count,
                output_count,
            )
        except Exception as exc:
            print(
                f"[ProductRecAgent] "
                f"Warning: record_rerank failed: {exc}"
            )

    # ========================================================
    # Main Execute
    # ========================================================

    async def _execute(
        self,
        **kwargs: Any,
    ) -> ProductRecResult:

        user_profile: UserProfile | None = kwargs.get(
            "user_profile"
        )

        num_items: int = kwargs.get(
            "num_items",
            10,
        )

        provided_candidates = kwargs.get(
            "candidates",
            None,
        )

        num_items = max(
            1,
            int(num_items),
        )

        print(
            "[ProductRecAgent] =============================="
        )
        print(
            "[ProductRecAgent] 开始执行"
        )
        print(
            "[ProductRecAgent] num_items:",
            num_items,
        )

        # ====================================================
        # Rerank 模式
        # ====================================================

        if provided_candidates is not None:

            candidates = list(
                provided_candidates
            )

            print(
                "[ProductRecAgent] 模式: RERANK"
            )

            print(
                "[ProductRecAgent] 使用外部候选商品:",
                len(candidates),
            )

            if not candidates:

                print(
                    "[ProductRecAgent] "
                    "没有候选商品，跳过 LLM Rerank"
                )

                print(
                    "[ProductRecAgent] =============================="
                )

                return ProductRecResult(
                    success=True,
                    products=[],
                    recall_strategy="external_candidates",
                    data={
                        "mode": "external_candidates",
                        "candidate_count": 0,
                        "reranked": 0,
                        "rerank_latency_ms": 0.0,
                        "output_count": 0,
                    },
                    confidence=0.5,
                )

            # ------------------------------------------------
            # LLM Rerank
            # ------------------------------------------------

            rerank_start = time.perf_counter()

            if self.enable_llm_rerank:

                ranked_ids = await self._rerank(
                    profile=user_profile,
                    candidates=candidates,
                    num_items=num_items,
                )

            else:

                ranked_ids = [
                    p.product_id
                    for p in candidates[:num_items]
                ]

            rerank_latency = (
                time.perf_counter()
                - rerank_start
            ) * 1000

            print(
                "[ProductRecAgent] Rerank耗时:",
                f"{rerank_latency:.1f}ms",
            )

            # ------------------------------------------------
            # ID → Product
            # ------------------------------------------------

            id_to_product = {
                product.product_id: product
                for product in candidates
            }

            final_products: list[Product] = []

            for product_id in ranked_ids:

                product = id_to_product.get(
                    product_id
                )

                if product is not None:
                    final_products.append(
                        product
                    )

            # ------------------------------------------------
            # Rerank 不足时补齐
            # ------------------------------------------------

            if len(final_products) < num_items:

                ranked_id_set = set(
                    ranked_ids
                )

                for product in candidates:

                    if product.product_id in ranked_id_set:
                        continue

                    final_products.append(
                        product
                    )

                    if len(final_products) >= num_items:
                        break

            final_products = final_products[
                :num_items
            ]

            print(
                "[ProductRecAgent] "
                "Rerank最终推荐数量:",
                len(final_products),
            )

            print(
                "[ProductRecAgent] "
                "Rerank最终推荐:",
                [
                    (
                        product.product_id,
                        product.name,
                    )
                    for product in final_products
                ],
            )

            print(
                "[ProductRecAgent] =============================="
            )

            return ProductRecResult(
                success=True,
                products=final_products,
                recall_strategy="external_candidates",
                data={
                    "mode": "external_candidates",
                    "candidate_count": len(candidates),
                    "reranked": len(ranked_ids),
                    "rerank_latency_ms": round(
                        rerank_latency,
                        1,
                    ),
                    "output_count": len(final_products),
                },
                confidence=0.85,
            )

        # ====================================================
        # Recall 模式
        # ====================================================

        if (
            self.enable_tool_calling
            and self.llm_with_tools
        ):

            print(
                "[ProductRecAgent] "
                "模式: RECALL (Tool Calling/ReAct)"
            )

            tool_start = time.perf_counter()

            final_product_ids = await self._tool_rec_call(
                profile=user_profile,
                num_items=num_items,
            )

            tool_latency = (
                time.perf_counter()
                - tool_start
            ) * 1000

            if self._recall_metrics:
                self._record_recall_metric(
                    latency_ms=tool_latency,
                    success=True,
                    **self._recall_metrics,
                )

            all_products = (
                self.product_store.get_all_products()
            )

            id_to_product: dict[str, Product] = {}

            for row in all_products:

                tags = row.get(
                    "tags",
                    "",
                )

                if isinstance(tags, str):

                    tags = [
                        t.strip()
                        for t in tags.split(",")
                        if t.strip()
                    ]

                elif not isinstance(tags, list):

                    tags = []

                id_to_product[
                    row["product_id"]
                ] = Product(
                    product_id=row["product_id"],
                    name=row["name"],
                    category=row["category"],
                    price=float(row["price"]),
                    description=row.get(
                        "description",
                        "",
                    ),
                    brand=row.get(
                        "brand",
                        "",
                    ),
                    seller_id=row.get(
                        "seller_id",
                        "",
                    ),
                    stock=int(
                        row.get("stock") or 0
                    ),
                    tags=tags,
                    image_url=row.get(
                        "image_url",
                        "",
                    ),
                )

            safe_ids = [
                pid
                for pid in final_product_ids
                if isinstance(pid, str)
            ]

            final_products = [
                id_to_product[pid]
                for pid in safe_ids
                if pid in id_to_product
            ][:num_items]

            print(
                "[ProductRecAgent] "
                "Tool Calling Recall耗时:",
                f"{tool_latency:.1f}ms",
            )

            print(
                "[ProductRecAgent] "
                "Tool Calling 最终推荐:",
                [
                    (
                        p.product_id,
                        p.name,
                    )
                    for p in final_products
                ],
            )

            print(
                "[ProductRecAgent] =============================="
            )

            return ProductRecResult(
                success=True,
                products=final_products,
                recall_strategy="tool_calling_react",
                data={
                    "mode": "tool_calling_react",
                    "num_items": num_items,
                    "tool_latency_ms": round(
                        tool_latency,
                        1,
                    ),
                    "tool_calls": final_product_ids,
                },
                confidence=0.9,
            )

        # ====================================================
        # 非 Tool Calling：Hybrid Recall
        # ====================================================

        print(
            "[ProductRecAgent] 模式: RECALL"
        )

        recall_limit = min(
            max(
                num_items * 3,
                15,
            ),
            30,
        )

        print(
            "[ProductRecAgent] recall_limit:",
            recall_limit,
        )

        recall_start = time.perf_counter()

        recall_data = await self._recall(
            user_profile=user_profile,
            limit=recall_limit,
        )

        recall_latency = (
            time.perf_counter()
            - recall_start
        ) * 1000

        candidates = recall_data["products"]

        print(
            "[ProductRecAgent] Recall耗时:",
            f"{recall_latency:.1f}ms",
        )

        print(
            "[ProductRecAgent] Recall候选数量:",
            len(candidates),
        )

        if not candidates:

            print(
                "[ProductRecAgent] "
                "Recall没有候选商品"
            )

            print(
                "[ProductRecAgent] =============================="
            )

            return ProductRecResult(
                success=True,
                products=[],
                recall_strategy="hybrid",
                data={
                    "mode": "recall_only",
                    "candidate_count": 0,
                    "reranked": 0,
                    "recall_latency_ms": round(
                        recall_latency,
                        1,
                    ),
                    "rerank_latency_ms": 0.0,
                    "query": recall_data.get(
                        "query",
                        "",
                    ),
                    "query_vector_dim": recall_data.get(
                        "query_vector_dim",
                        0,
                    ),
                    "vector_results": recall_data.get(
                        "vector_results",
                        [],
                    ),
                    "keyword_results": recall_data.get(
                        "keyword_results",
                        [],
                    ),
                    "rrf_results": recall_data.get(
                        "rrf_results",
                        [],
                    ),
                },
                confidence=0.5,
            )

        print(
            "[ProductRecAgent] "
            "Recall完成，返回候选商品"
        )

        print(
            "[ProductRecAgent] Recall候选:",
            [
                (
                    product.product_id,
                    product.name,
                )
                for product in candidates
            ],
        )

        print(
            "[ProductRecAgent] =============================="
        )

        return ProductRecResult(
            success=True,
            products=candidates,
            recall_strategy="hybrid",
            data={
                "mode": "recall_only",
                "candidate_count": len(candidates),
                "reranked": 0,
                "recall_latency_ms": round(
                    recall_latency,
                    1,
                ),
                "rerank_latency_ms": 0.0,
                "query": recall_data.get(
                    "query",
                    "",
                ),
                "query_vector_dim": recall_data.get(
                    "query_vector_dim",
                    0,
                ),
                "vector_results": recall_data.get(
                    "vector_results",
                    [],
                ),
                "keyword_results": recall_data.get(
                    "keyword_results",
                    [],
                ),
                "rrf_results": recall_data.get(
                    "rrf_results",
                    [],
                ),
            },
            confidence=0.80,
        )

    # ========================================================
    # Tool Calling: ReAct Loop
    # ========================================================

    async def _tool_rec_call(
        self,
        profile,
        num_items: int,
    ) -> list[str]:

        messages = [
            SystemMessage(
                content=TOOL_REACT_SYSTEM_PROMPT.format(
                    num_items=num_items
                )
            ),
            HumanMessage(
                content=self._build_tool_profile_summary(
                    profile
                )
            ),
        ]

        print(
            "[ProductRecAgent] Tool Calling ReAct 开始"
        )

        print(
            "[ProductRecAgent] 用户画像:",
            self._build_tool_profile_summary(profile),
        )

        tool_calls_log = []

        for iteration in range(
            MAX_TOOL_ITERATIONS
        ):

            try:

                response = await (
                    self.llm_with_tools.ainvoke(
                        messages
                    )
                )

            except Exception as exc:

                print(
                    "[ProductRecAgent] "
                    "LLM 调用失败:",
                    str(exc),
                )

                break

            if response.tool_calls:

                print(
                    "[ProductRecAgent] Iteration",
                    iteration,
                    "- Tool calls:",
                    [
                        tc.get("name")
                        for tc in response.tool_calls
                    ],
                )

                messages.append(response)

                dp_done_count = sum(
                    1
                    for log in tool_calls_log
                    if log["tool"]
                    == "get_product_detail"
                )

                dp_remaining = max(
                    0,
                    3 - dp_done_count,
                )

                pending_tcs = []
                skipped_tcs = []

                for tc in response.tool_calls:

                    tool_name = tc.get(
                        "name",
                        "",
                    )

                    tool_args = tc.get(
                        "args",
                        {},
                    )

                    tool_call_id = tc.get(
                        "id",
                        "",
                    )

                    if (
                        tool_name
                        == "get_product_detail"
                        and dp_remaining <= 0
                    ):

                        skipped_tcs.append(
                            {
                                "tc": tc,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "tool_call_id": tool_call_id,
                            }
                        )

                    else:

                        pending_tcs.append(
                            {
                                "tc": tc,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "tool_call_id": tool_call_id,
                            }
                        )

                    if (
                        tool_name
                        == "get_product_detail"
                    ):
                        dp_remaining -= 1

                iteration_start = time.perf_counter()

                async def _run_tool(tc_info):

                    tool_name = tc_info[
                        "tool_name"
                    ]

                    tool_args = tc_info[
                        "tool_args"
                    ]

                    if (
                        tool_name
                        == "search_products"
                    ):

                        return await (
                            self._search_products_tool.ainvoke(
                                tool_args
                            )
                        )

                    if (
                        tool_name
                        == "get_product_detail"
                    ):

                        return await (
                            self._get_product_detail_tool.ainvoke(
                                tool_args
                            )
                        )

                    if (
                        tool_name
                        == "check_stock"
                    ):

                        return await (
                            self._check_stock_tool.ainvoke(
                                tool_args
                            )
                        )

                    return (
                        f"Unknown tool: {tool_name}"
                    )

                async def _run_skipped(tc_info):

                    return (
                        "[ProductRecAgent] "
                        "get_product_detail call "
                        "limit (3) reached."
                    )

                results = await asyncio.gather(
                    *[
                        _run_tool(ti)
                        for ti in pending_tcs
                    ],
                    return_exceptions=True,
                )

                skipped_results = await asyncio.gather(
                    *[
                        _run_skipped(ti)
                        for ti in skipped_tcs
                    ],
                    return_exceptions=True,
                )

                all_tcs = (
                    pending_tcs
                    + skipped_tcs
                )

                all_results = (
                    list(results)
                    + list(skipped_results)
                )

                iteration_tool_count = 0
                iteration_search_count = 0
                iteration_detail_count = 0

                for tc_info, result in zip(
                    all_tcs,
                    all_results,
                ):

                    tool_name = tc_info[
                        "tool_name"
                    ]

                    tool_args = tc_info[
                        "tool_args"
                    ]

                    tool_call_id = tc_info[
                        "tool_call_id"
                    ]

                    iteration_tool_count += 1

                    if (
                        tool_name
                        == "search_products"
                    ):
                        iteration_search_count += 1

                    elif (
                        tool_name
                        == "get_product_detail"
                    ):
                        iteration_detail_count += 1

                    latency_ms = (
                        time.perf_counter()
                        - iteration_start
                    ) * 1000

                    if isinstance(
                        result,
                        Exception,
                    ):

                        self._record_tool_metric(
                            tool_name,
                            False,
                            latency_ms,
                        )

                        tool_calls_log.append(
                            {
                                "tool": tool_name,
                                "args": tool_args,
                                "error": str(result),
                            }
                        )

                        messages.append(
                            ToolMessage(
                                content=(
                                    f"Error: {str(result)}"
                                ),
                                tool_call_id=tool_call_id,
                            )
                        )

                        continue

                    self._record_tool_metric(
                        tool_name,
                        True,
                        latency_ms,
                    )

                    tool_calls_log.append(
                        {
                            "tool": tool_name,
                            "args": tool_args,
                            "result_length": len(
                                str(result)
                            ),
                        }
                    )

                    messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call_id,
                        )
                    )

                print(
                    "[ProductRecAgent] "
                    "Iteration",
                    iteration,
                    "summary:",
                    "tool_count=",
                    iteration_tool_count,
                    "search_products_count=",
                    iteration_search_count,
                    "get_product_detail_count=",
                    iteration_detail_count,
                )

            else:

                final_content = (
                    str(
                        response.content
                    ).strip()
                )

                print(
                    "[ProductRecAgent] "
                    "LLM 最终回答:",
                    final_content[:200],
                )

                messages.append(response)

                break

        final_ids = await self._parse_tool_products(
            messages,
            num_items,
        )

        total_sp = sum(
            1
            for log in tool_calls_log
            if log["tool"]
            == "search_products"
        )

        total_dp = sum(
            1
            for log in tool_calls_log
            if log["tool"]
            == "get_product_detail"
        )

        self._recall_metrics = {
            "iterations": len(tool_calls_log),
            "tool_calls": len(tool_calls_log),
            "sp_calls": total_sp,
            "dp_calls": total_dp,
        }

        validated_ids = [
            pid
            for pid in final_ids
            if isinstance(pid, str)
            and pid
        ]

        return validated_ids

    # ========================================================
    # Parse Tool Products
    # ========================================================

    async def _parse_tool_products(
        self,
        messages: list,
        num_items: int,
    ) -> list[str]:

        def _strip_markdown_fence(
            text: str,
        ) -> str:

            text = text.strip()

            if text.startswith("```"):

                text = text[3:].lstrip()

                if text.startswith("json"):
                    text = text[4:].lstrip()

                if text.endswith("```"):
                    text = text[:-3].rstrip()

            return text

        def _extract_ids(raw_result):

            if (
                not isinstance(
                    raw_result,
                    list,
                )
                or not raw_result
            ):
                return []

            ids = []

            for item in raw_result:

                if isinstance(item, str):

                    if (
                        item
                        and item not in ids
                    ):
                        ids.append(item)

                elif isinstance(
                    item,
                    dict,
                ):

                    pid = item.get(
                        "product_id"
                    )

                    if (
                        isinstance(
                            pid,
                            str,
                        )
                        and pid
                        and pid not in ids
                    ):
                        ids.append(pid)

                if len(ids) >= num_items:
                    break

            return ids

        final_ai_message = next(
            (
                msg
                for msg in reversed(messages)
                if isinstance(
                    msg,
                    AIMessage,
                )
            ),
            None,
        )

        if final_ai_message is None:

            print(
                "[ProductRecAgent] "
                "ERROR: no final AIMessage found"
            )

            return []

        final_content = str(
            final_ai_message.content
        ).strip()

        cleaned = _strip_markdown_fence(
            final_content
        )

        if cleaned.startswith("["):

            try:

                result = json.loads(
                    cleaned
                )

                ids = _extract_ids(
                    result
                )

                if ids:
                    return ids

            except json.JSONDecodeError:
                pass

        start = cleaned.find("[")
        end = cleaned.rfind("]")

        if (
            start != -1
            and end != -1
            and end > start
        ):

            try:

                result = json.loads(
                    cleaned[start:end + 1]
                )

                ids = _extract_ids(
                    result
                )

                if ids:
                    return ids

            except json.JSONDecodeError:
                pass

        print(
            "[ProductRecAgent] "
            "ERROR: failed to parse final LLM product IDs"
        )

        return []

    # ========================================================
    # Tool Profile Summary
    # ========================================================

    @staticmethod
    def _build_tool_profile_summary(
        profile,
    ) -> str:

        if not profile:
            return "用户为新用户，无历史行为数据。"

        parts = []

        if profile.preferred_categories:

            parts.append(
                "偏好类别: "
                + ", ".join(
                    profile.preferred_categories
                )
            )

        if profile.recent_views:

            parts.append(
                "最近浏览: "
                + ", ".join(
                    profile.recent_views[:5]
                )
            )

        if profile.recent_purchases:

            parts.append(
                "最近购买: "
                + ", ".join(
                    profile.recent_purchases[:5]
                )
            )

        if profile.real_time_tags:

            if isinstance(
                profile.real_time_tags,
                dict,
            ):

                tags_text = ", ".join(
                    str(k)
                    + ":"
                    + str(v)
                    for k, v
                    in list(
                        profile.real_time_tags.items()
                    )[:5]
                )

            else:

                tags_text = ", ".join(
                    str(x)
                    for x in profile.real_time_tags
                )

            parts.append(
                "实时标签: "
                + tags_text
            )

        if profile.segments:

            seg_names = [
                s.value
                for s in profile.segments
            ]

            parts.append(
                "用户分群: "
                + ", ".join(seg_names)
            )

        if parts:

            return (
                "用户画像: "
                + "; ".join(parts)
            )

        return "用户为新用户，无历史行为数据。"

    # ========================================================
    # Hybrid Recall
    # ========================================================

    async def _recall(
        self,
        user_profile: UserProfile | None,
        limit: int,
        query_text: str | None = None,
        excluded_product_ids: set[str] | None = None,
    ) -> dict[str, Any]:

        print(
            "[ProductRecAgent] "
            "开始 Hybrid Recall"
        )

        # ====================================================
        # 1. MySQL
        # ====================================================

        mysql_start = time.perf_counter()

        rows = self.product_store.get_all_products()

        mysql_latency = (
            time.perf_counter()
            - mysql_start
        ) * 1000

        print(
            "[ProductRecAgent] MySQL耗时:",
            f"{mysql_latency:.1f}ms",
        )

        if not rows:

            return {
                "products": [],
                "query": "",
                "query_vector_dim": 0,
                "vector_results": [],
                "keyword_results": [],
                "rrf_results": [],
            }

        # ====================================================
        # 2. MySQL → Product
        # ====================================================

        products: list[Product] = []

        for row in rows:

            tags = row.get(
                "tags",
                "",
            )

            if isinstance(tags, str):

                tags = [
                    tag.strip()
                    for tag in tags.split(",")
                    if tag.strip()
                ]

            elif not isinstance(tags, list):

                tags = []

            products.append(
                Product(
                    product_id=row["product_id"],
                    name=row["name"],
                    category=row["category"],
                    price=float(row["price"]),
                    description=row.get(
                        "description"
                    ) or "",
                    brand=row.get(
                        "brand"
                    ) or "",
                    seller_id=row.get(
                        "seller_id"
                    ) or "",
                    stock=int(
                        row.get("stock") or 0
                    ),
                    tags=tags,
                    image_url=row.get(
                        "image_url"
                    ) or "",
                )
            )

        print(
            "[ProductRecAgent] MySQL商品总数:",
            len(products),
        )

        # ====================================================
        # Evaluation / 测试商品排除
        # ====================================================

        if excluded_product_ids:

            before_exclude = len(
                products
            )

            products = [
                product
                for product in products
                if product.product_id
                not in excluded_product_ids
            ]

            print(
                "[ProductRecAgent] 排除指定商品:",
                f"{before_exclude} -> {len(products)}",
                "excluded=",
                sorted(
                    excluded_product_ids
                ),
            )

        # ====================================================
        # 3. 没有 UserProfile
        # ====================================================

        if not user_profile:

            available_products = [
                product
                for product in products
                if product.stock > 0
            ]

            available_products = (
                available_products[:limit]
            )

            return {
                "products": available_products,
                "query": "",
                "query_vector_dim": 0,
                "vector_results": [],
                "keyword_results": [],
                "rrf_results": [],
            }

        # ====================================================
        # 4. MySQL 硬过滤
        # ====================================================

        min_price, max_price = (
            user_profile.price_range
        )

        price_matched = [
            product
            for product in products
            if (
                product.stock > 0
                and min_price
                <= product.price
                <= max_price
            )
        ]

        print(
            "[ProductRecAgent] "
            "MySQL价格+库存过滤:",
            f"{len(products)} -> "
            f"{len(price_matched)}",
        )

        if not price_matched:

            return {
                "products": [],
                "query": "",
                "query_vector_dim": 0,
                "vector_results": [],
                "keyword_results": [],
                "rrf_results": [],
            }

        # ====================================================
        # 4.5 Ensure Milvus
        # ====================================================

        ensure_indexing = (
            self.vector_store.ensure_products_indexed(
                price_matched,
                self.embedding_service,
            )
        )

        print(
            "[ProductRecAgent] "
            "Milvus 索引: "
            "total=",
            ensure_indexing["total"],
            "existing=",
            ensure_indexing["existing"],
            "created=",
            ensure_indexing["created"],
        )

        # ====================================================
        # 5. Query Rewrite
        # ====================================================

        query_start = time.perf_counter()

        if (
            query_text is not None
            and query_text.strip()
        ):

            query_text = re.sub(
                r"\s+",
                " ",
                query_text,
            ).strip()[:500]

            print(
                "[ProductRecAgent] 使用外部 Query:",
                query_text,
            )

        else:

            if (
                self.enable_tool_calling
                and self.llm_with_tools
            ):

                query_text = (
                    await self._tool_rewrite_query(
                        user_profile
                    )
                )

            else:

                query_text = (
                    await self._rewrite_query(
                        user_profile
                    )
                )

        query_latency = (
            time.perf_counter()
            - query_start
        ) * 1000

        print(
            "[ProductRecAgent] "
            "Query Rewrite耗时:",
            f"{query_latency:.1f}ms",
        )

        print(
            "[ProductRecAgent] Rewrite Query:"
        )

        print(query_text)

        # ====================================================
        # 6. Embedding
        # ====================================================

        embedding_start = time.perf_counter()

        query_vector = (
            self.embedding_service.embed(
                query_text
            )
        )

        embedding_latency = (
            time.perf_counter()
            - embedding_start
        ) * 1000

        print(
            "[ProductRecAgent] Embedding耗时:",
            f"{embedding_latency:.1f}ms",
        )

        print(
            "[ProductRecAgent] Query Vector Dim:",
            len(query_vector),
        )

        # ====================================================
        # 7. Vector Search
        # ====================================================

        vector_search_limit = max(
            limit * 5,
            50,
        )

        vector_start = time.perf_counter()

        vector_results = (
            self.vector_store.search(
                query_embedding=query_vector,
                limit=vector_search_limit,
            )
        )

        vector_latency = (
            time.perf_counter()
            - vector_start
        ) * 1000

        print(
            "[ProductRecAgent] Vector Search耗时:",
            f"{vector_latency:.1f}ms",
        )

        print(
            "[ProductRecAgent] Vector Raw Results:",
            len(vector_results),
        )

        # ====================================================
        # 8. Vector IDs
        # ====================================================

        vector_ranked_ids: list[str] = []

        vector_debug_results: list[
            dict[str, Any]
        ] = []

        allowed_product_ids = {
            product.product_id
            for product in price_matched
        }

        for result in vector_results:

            entity = result.get(
                "entity",
                {},
            )

            product_id = entity.get(
                "product_id"
            )

            if not product_id:
                continue

            if (
                product_id
                not in allowed_product_ids
            ):
                continue

            if (
                product_id
                in vector_ranked_ids
            ):
                continue

            vector_ranked_ids.append(
                product_id
            )

            vector_debug_results.append(
                {
                    "product_id": product_id,
                    "distance": result.get(
                        "distance"
                    ),
                }
            )

            # 注意：
            # 当前先保持原来的行为，
            # 只取 limit 个 Vector 候选。
            if len(vector_ranked_ids) >= limit:
                break

        print(
            "[ProductRecAgent] Vector结果:",
            vector_debug_results,
        )

        # ====================================================
        # 9. Keyword Search
        # ====================================================

        keyword_start = time.perf_counter()

        keyword_ranked_ids = (
            self._keyword_search(
                query_text=query_text,
                products=price_matched,
                limit=limit,
            )
        )

        keyword_latency = (
            time.perf_counter()
            - keyword_start
        ) * 1000

        print(
            "[ProductRecAgent] Keyword Search耗时:",
            f"{keyword_latency:.1f}ms",
        )

        print(
            "[ProductRecAgent] Keyword结果:",
            keyword_ranked_ids,
        )

        # ====================================================
        # 10. RRF Fusion
        # ====================================================

        fusion_start = time.perf_counter()

        fused_ids = self._fuse_results(
            vector_ranked_ids=vector_ranked_ids,
            keyword_ranked_ids=keyword_ranked_ids,
            limit=limit,
        )

        fusion_latency = (
            time.perf_counter()
            - fusion_start
        ) * 1000

        print(
            "[ProductRecAgent] Hybrid Fusion耗时:",
            f"{fusion_latency:.1f}ms",
        )

        print(
            "[ProductRecAgent] Hybrid Fusion结果:",
            fused_ids,
        )

        # ====================================================
        # 11. ID → Product
        # ====================================================

        id_to_product = {
            product.product_id: product
            for product in price_matched
        }

        candidates: list[Product] = []

        for product_id in fused_ids:

            product = id_to_product.get(
                product_id
            )

            if product is not None:
                candidates.append(
                    product
                )

        # ====================================================
        # 12. Hybrid 结果不足时补齐
        # ====================================================

        if len(candidates) < limit:

            candidate_ids = {
                product.product_id
                for product in candidates
            }

            remaining = [
                product
                for product in price_matched
                if product.product_id
                not in candidate_ids
            ]

            preferred_categories = set(
                user_profile.preferred_categories
                or []
            )

            remaining.sort(
                key=lambda product: (
                    product.category
                    in preferred_categories,
                    product.stock > 0,
                ),
                reverse=True,
            )

            candidates.extend(
                remaining
            )

        candidates = candidates[
            :limit
        ]

        print(
            "[ProductRecAgent] "
            "最终Hybrid候选:",
            len(candidates),
        )

        # ====================================================
        # 13. 返回 Recall 调试数据
        # ====================================================

        return {
            "products": candidates,
            "query": query_text,
            "query_vector_dim": len(
                query_vector
            ),
            "vector_results": (
                vector_debug_results
            ),
            "keyword_results": (
                keyword_ranked_ids
            ),
            "rrf_results": fused_ids,
        }

    # ========================================================
    # Query Rewrite
    # ========================================================

    async def _rewrite_query(
        self,
        profile: UserProfile,
    ) -> str:

        rewrite_start = time.perf_counter()

        preferred_categories = [
            str(item).strip()
            for item in (
                profile.preferred_categories
                or []
            )
            if str(item).strip()
        ]

        recent_views = [
            str(item).strip()
            for item in (
                profile.recent_views
                or []
            )
            if str(item).strip()
        ]

        recent_purchases = [
            str(item).strip()
            for item in (
                profile.recent_purchases
                or []
            )
            if str(item).strip()
        ]

        if isinstance(
            profile.real_time_tags,
            dict,
        ):

            real_time_tags = [
                str(item).strip()
                for item in (
                    profile.real_time_tags.keys()
                    if profile.real_time_tags
                    else []
                )
                if str(item).strip()
            ]

        else:

            real_time_tags = [
                str(item).strip()
                for item in (
                    profile.real_time_tags
                    or []
                )
                if str(item).strip()
            ]

        def unique_items(
            items: list[str],
            max_items: int,
        ) -> list[str]:

            result: list[str] = []

            for item in items:

                if not item:
                    continue

                if item in result:
                    continue

                result.append(item)

                if len(result) >= max_items:
                    break

            return result

        preferred_categories = unique_items(
            preferred_categories,
            5,
        )

        recent_views = unique_items(
            recent_views,
            5,
        )

        recent_purchases = unique_items(
            recent_purchases,
            5,
        )

        real_time_tags = unique_items(
            real_time_tags,
            5,
        )

        query_parts: list[str] = []

        if real_time_tags:

            query_parts.append(
                "当前关注"
                + "、".join(
                    real_time_tags
                )
                + "相关商品"
            )

        if preferred_categories:

            query_parts.append(
                "偏好"
                + "、".join(
                    preferred_categories
                )
                + "类商品"
            )

        if recent_views:

            query_parts.append(
                "近期浏览过"
                + "、".join(
                    recent_views
                )
                + "相关商品"
            )

        if recent_purchases:

            query_parts.append(
                "近期购买过"
                + "、".join(
                    recent_purchases
                )
                + "相关商品"
            )

        if query_parts:

            query = (
                "推荐"
                + "，".join(query_parts)
                + "，优先选择与用户兴趣和历史行为相关的商品"
            )

        else:

            query = "综合类热门商品"

        query = re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

        query = query[:500]

        rewrite_latency = (
            time.perf_counter()
            - rewrite_start
        ) * 1000

        print(
            "[ProductRecAgent] "
            "Rule-based Query Rewrite耗时:",
            f"{rewrite_latency:.1f}ms",
        )

        print(
            "[ProductRecAgent] "
            "Rule-based Rewrite Query:"
        )

        print(query)

        return query

    # ========================================================
    # Fallback Query Builder
    # ========================================================

    @staticmethod
    def _build_query(
        profile: UserProfile,
    ) -> str:

        categories = ", ".join(
            profile.preferred_categories
            or []
        )

        recent_views = ", ".join(
            (
                profile.recent_views
                or []
            )[:10]
        )

        recent_purchases = ", ".join(
            (
                profile.recent_purchases
                or []
            )[:10]
        )

        if isinstance(
            profile.real_time_tags,
            dict,
        ):

            real_time_tags = ", ".join(
                str(tag)
                for tag in profile.real_time_tags.keys()
            )

        else:

            real_time_tags = ", ".join(
                str(tag)
                for tag in (
                    profile.real_time_tags
                    or []
                )
            )

        return (
            f"用户感兴趣的商品类别："
            f"{categories}。"
            f"最近浏览："
            f"{recent_views}。"
            f"最近购买："
            f"{recent_purchases}。"
            f"当前兴趣："
            f"{real_time_tags}。"
        )

    # ========================================================
    # Tool Calling Query Rewrite
    # ========================================================

    async def _tool_rewrite_query(
        self,
        profile,
    ):

        rewrite_start = time.perf_counter()

        if not self.llm_with_tools:
            return await self._rewrite_query(
                profile
            )

        try:

            messages = [
                SystemMessage(
                    content=(
                        "你是电商搜索 Query Rewrite 专家。"
                        "你可以使用 search_products Tool 来了解当前商品库的内容。"
                        "请先调用 search_products 搜索与用户兴趣相关的商品，"
                        "然后根据搜索结果和用户画像，构造一个高质量的搜索 Query。"
                        "只输出最终的搜索 Query，不要输出其他内容。"
                    )
                ),
                HumanMessage(
                    content=(
                        "用户偏好类别："
                        + str(
                            profile.preferred_categories
                            or []
                        )
                        + "\n"
                        "最近浏览："
                        + str(
                            profile.recent_views
                            or []
                        )
                        + "\n"
                        "最近购买："
                        + str(
                            profile.recent_purchases
                            or []
                        )
                        + "\n"
                        "实时标签："
                        + str(
                            profile.real_time_tags
                            or {}
                        )
                    )
                ),
            ]

            response = await (
                self.llm_with_tools.ainvoke(
                    messages
                )
            )

            if response.tool_calls:

                tool_results = []

                for tc in response.tool_calls:

                    if (
                        tc.get("name")
                        == "search_products"
                    ):

                        args = tc.get(
                            "args",
                            {},
                        )

                        query_arg = args.get(
                            "query",
                            "",
                        )

                        if query_arg:

                            tool_results.append(
                                "搜索关键词: "
                                + query_arg
                            )

                if tool_results:

                    query_text = "、".join(
                        tool_results
                    )

                else:

                    query_text = str(
                        response.content
                    ).strip()

            else:

                query_text = str(
                    response.content
                ).strip()

            query_text = re.sub(
                r"\s+",
                " ",
                query_text,
            ).strip()

            query_text = query_text[:500]

        except Exception as exc:

            print(
                "[ProductRecAgent] "
                "Tool Calling Query Rewrite 失败:",
                str(exc),
            )

            query_text = await (
                self._rewrite_query(
                    profile
                )
            )

        rewrite_latency = (
            time.perf_counter()
            - rewrite_start
        ) * 1000

        print(
            "[ProductRecAgent] "
            "Tool Calling Query Rewrite耗时:",
            f"{rewrite_latency:.1f}ms",
        )

        print(
            "[ProductRecAgent] "
            "Tool Calling Rewrite Query:",
            query_text,
        )

        return query_text

    # ========================================================
    # Keyword Search
    # ========================================================

    @staticmethod
    def _keyword_search(
        query_text: str,
        products: list[Product],
        limit: int,
    ) -> list[str]:

        if not products or not query_text:
            return []

        # ====================================================
        # 1. Query Tokenization
        # ====================================================

        query_tokens = (
            ProductRecAgent._tokenize_query(
                query_text
            )
        )

        if not query_tokens:

            return [
                product.product_id
                for product in products[:limit]
            ]

        # ====================================================
        # 2. 预先建立同义词扩展
        # ====================================================

        expanded_tokens: list[str] = []

        for token in query_tokens:

            if token not in expanded_tokens:
                expanded_tokens.append(token)

            for synonym in (
                ProductRecAgent._KEYWORD_SYNONYMS.get(
                    token,
                    [],
                )
            ):

                if synonym not in expanded_tokens:
                    expanded_tokens.append(
                        synonym
                    )

        # ====================================================
        # 3. Scoring
        # ====================================================

        scored_products: list[
            tuple[Product, float]
        ] = []

        for product in products:

            name = (
                product.name or ""
            ).lower()

            category = (
                product.category or ""
            ).lower()

            brand = (
                product.brand or ""
            ).lower()

            tags = [
                str(tag).lower()
                for tag in (
                    product.tags or []
                )
            ]

            description = (
                product.description or ""
            ).lower()

            all_text = " ".join(
                [
                    name,
                    category,
                    brand,
                    " ".join(tags),
                    description,
                ]
            )

            score = 0.0

            matched_original_tokens = 0

            matched_tokens: set[str] = set()

            # ------------------------------------------------
            # Token 匹配
            # ------------------------------------------------

            for token in expanded_tokens:

                token_lower = (
                    token.lower()
                )

                matched = False

                # 商品名称：最高权重
                if token_lower in name:

                    score += 8.0
                    matched = True

                    # 精确名称短语额外奖励
                    if name == token_lower:
                        score += 5.0

                # 商品类别
                if token_lower in category:

                    score += 6.0
                    matched = True

                # 商品标签
                tag_match = any(
                    token_lower in tag
                    for tag in tags
                )

                if tag_match:

                    score += 5.0
                    matched = True

                # 品牌
                if token_lower in brand:

                    score += 4.0
                    matched = True

                # 商品描述
                if token_lower in description:

                    score += 2.0
                    matched = True

                if matched:

                    matched_tokens.add(
                        token_lower
                    )

            # ------------------------------------------------
            # 原始 Query Token 覆盖率
            # ------------------------------------------------

            original_token_set = {
                token.lower()
                for token in query_tokens
            }

            for token in original_token_set:

                token_lower = token.lower()

                if (
                    token_lower in name
                    or token_lower in category
                    or any(
                        token_lower in tag
                        for tag in tags
                    )
                    or token_lower in brand
                    or token_lower in description
                ):

                    matched_original_tokens += 1

            if original_token_set:

                coverage = (
                    matched_original_tokens
                    / len(original_token_set)
                )

                score += coverage * 10.0

            # ------------------------------------------------
            # Query 完整短语匹配
            # ------------------------------------------------

            normalized_query = (
                re.sub(
                    r"\s+",
                    "",
                    query_text.lower(),
                )
            )

            if (
                normalized_query
                and normalized_query in all_text
            ):

                score += 12.0

            # ------------------------------------------------
            # 多个 Token 同时命中额外奖励
            # ------------------------------------------------

            if len(matched_tokens) >= 2:
                score += 2.0

            if len(matched_tokens) >= 3:
                score += 3.0

            if len(matched_tokens) >= 4:
                score += 4.0

            if score > 0:

                scored_products.append(
                    (
                        product,
                        score,
                    )
                )

        # ====================================================
        # 4. 排序
        # ====================================================

        scored_products.sort(
            key=lambda item: (
                item[1],
                -float(item[0].price),
            ),
            reverse=True,
        )

        return [
            product.product_id
            for product, _score
            in scored_products[:limit]
        ]

    # ========================================================
    # Keyword Synonyms
    # ========================================================

    _KEYWORD_SYNONYMS: dict[str, list[str]] = {

        # ----------------------------------------------------
        # 价格 / 性价比
        # ----------------------------------------------------

        "便宜": [
            "性价比",
            "高性价比",
            "实惠",
            "低价",
        ],

        "实惠": [
            "便宜",
            "性价比",
            "高性价比",
        ],

        "性价比": [
            "便宜",
            "高性价比",
            "实惠",
        ],

        "高性价比": [
            "性价比",
            "便宜",
            "实惠",
        ],

        # ----------------------------------------------------
        # 便携 / 轻薄
        # ----------------------------------------------------

        "轻便": [
            "便携",
            "轻薄",
        ],

        "便携": [
            "轻便",
            "轻薄",
        ],

        "轻薄": [
            "轻便",
            "便携",
        ],

        # ----------------------------------------------------
        # 性能
        # ----------------------------------------------------

        "高性能": [
            "性能",
            "旗舰",
        ],

        "性能": [
            "高性能",
        ],

        # ----------------------------------------------------
        # 快充
        # ----------------------------------------------------

        "快充": [
            "充电器",
            "氮化镓",
        ],

        "充电器": [
            "快充",
            "氮化镓",
        ],

        "氮化镓": [
            "快充",
            "充电器",
        ],

        # ----------------------------------------------------
        # 无线 / 蓝牙
        # ----------------------------------------------------

        "无线": [
            "蓝牙",
        ],

        "蓝牙": [
            "无线",
        ],

        # ----------------------------------------------------
        # 降噪
        # ----------------------------------------------------

        "降噪": [
            "耳机",
        ],

        "降噪耳机": [
            "降噪",
            "耳机",
            "无线",
        ],

        # ----------------------------------------------------
        # 办公
        # ----------------------------------------------------

        "办公": [
            "学习",
            "电脑",
            "鼠标",
            "显示器",
        ],

        "工作": [
            "办公",
        ],

        # ----------------------------------------------------
        # 学习
        # ----------------------------------------------------

        "学习": [
            "办公",
        ],

        # ----------------------------------------------------
        # 拍照 / 摄影
        # ----------------------------------------------------

        "拍照": [
            "摄影",
            "影像",
        ],

        "摄影": [
            "拍照",
            "影像",
        ],

        # ----------------------------------------------------
        # 游戏 / 娱乐
        # ----------------------------------------------------

        "游戏": [
            "娱乐",
        ],

        "娱乐": [
            "游戏",
        ],

        # ----------------------------------------------------
        # 户外 / 运动
        # ----------------------------------------------------

        "运动": [
            "户外",
        ],

        "户外": [
            "运动",
        ],

        # ----------------------------------------------------
        # 数码产品
        # ----------------------------------------------------

        "数码产品": [
            "手机",
            "平板",
            "耳机",
            "配件",
        ],

        "数码": [
            "手机",
            "平板",
            "耳机",
            "配件",
        ],

        # ----------------------------------------------------
        # 手机
        # ----------------------------------------------------

        "手机": [
            "智能手机",
        ],

        # ----------------------------------------------------
        # 平板
        # ----------------------------------------------------

        "平板": [
            "学习",
            "办公",
            "娱乐",
        ],

        # ----------------------------------------------------
        # 耳机
        # ----------------------------------------------------

        "耳机": [
            "无线",
            "蓝牙",
            "降噪",
        ],

        # ----------------------------------------------------
        # 配件
        # ----------------------------------------------------

        "配件": [
            "充电器",
            "快充",
        ],
    }

    # ========================================================
    # Tokenizer
    # ========================================================

    @staticmethod
    def _tokenize_query(
        query_text: str,
    ) -> list[str]:

        if not query_text:
            return []

        text = (
            str(query_text)
            .lower()
            .replace("\n", " ")
        )

        # ====================================================
        # 1. 删除无检索价值的字段名称
        # ====================================================

        stop_phrases = [
            "用户偏好类别",
            "价格范围",
            "用户分群",
            "最近浏览",
            "最近购买",
            "实时兴趣标签",
            "最近浏览商品",
            "最近购买商品",
            "用户感兴趣的商品类别",
            "当前兴趣",
            "偏好类别",
            "偏好",
            "推荐",
            "相关商品",
            "商品",
            "当前关注",
            "近期浏览过",
            "近期购买过",
            "优先选择与用户兴趣和历史行为相关的",
        ]

        for phrase in stop_phrases:

            text = text.replace(
                phrase,
                " ",
            )

        # ====================================================
        # 2. 英文 / 数字
        # ====================================================

        english_tokens = re.findall(
            r"[a-zA-Z0-9]+",
            text,
        )

        # ====================================================
        # 3. 中文连续文本
        # ====================================================

        chinese_chunks = re.findall(
            r"[\u4e00-\u9fff]+",
            text,
        )

        tokens: list[str] = []

        def add_token(
            token: str,
        ):

            token = token.strip()

            if not token:
                return

            if len(token) < 2:
                return

            if token not in tokens:
                tokens.append(token)

        # ====================================================
        # 4. 领域词优先识别
        #
        # 防止：
        # "高性价比" 被拆成大量无意义窗口
        # ====================================================

        domain_terms = sorted(
            {
                term
                for term in ProductRecAgent._KEYWORD_SYNONYMS
                if len(term) >= 2
            },
            key=len,
            reverse=True,
        )

        for chunk in chinese_chunks:

            remaining = chunk

            found_domain_term = False

            for term in domain_terms:

                if term in remaining:

                    add_token(term)

                    remaining = remaining.replace(
                        term,
                        " ",
                    )

                    found_domain_term = True

            # ------------------------------------------------
            # 原始中文短语
            #
            # 只有长度较短时才直接加入，
            # 避免整句成为一个 Token。
            # ------------------------------------------------

            compact = re.sub(
                r"\s+",
                "",
                chunk,
            )

            if (
                2
                <= len(compact)
                <= 4
            ):

                add_token(compact)

            # ------------------------------------------------
            # 中文 2/3 字 N-gram
            # ------------------------------------------------

            for size in (2, 3):

                if len(compact) < size:
                    continue

                for i in range(
                    len(compact) - size + 1
                ):

                    piece = compact[
                        i:i + size
                    ]

                    add_token(piece)

        # ====================================================
        # 5. 英文 / 数字
        # ====================================================

        for token in english_tokens:
            add_token(token)

        # ====================================================
        # 6. 清理明显的通用虚词
        # ====================================================

        generic_tokens = {
            "适合",
            "相关",
            "推荐",
            "优先",
            "用户",
            "当前",
            "近期",
            "使用",
            "以及",
            "商品",
            "产品",
            "类别",
            "历史",
            "行为",
            "兴趣",
        }

        tokens = [
            token
            for token in tokens
            if token not in generic_tokens
        ]

        # ====================================================
        # 7. 去重并限制 Token 数量
        # ====================================================

        result: list[str] = []

        for token in tokens:

            if token not in result:

                result.append(
                    token
                )

        return result[:80]

    # ========================================================
    # Hybrid Fusion
    # ========================================================

    @staticmethod
    def _fuse_results(
        vector_ranked_ids: list[str],
        keyword_ranked_ids: list[str],
        limit: int,
    ) -> list[str]:

        # ----------------------------------------------------
        # Reciprocal Rank Fusion
        # ----------------------------------------------------

        k = 60

        scores: dict[str, float] = {}

        # ====================================================
        # Vector Search
        # ====================================================

        for rank, product_id in enumerate(
            vector_ranked_ids,
            start=1,
        ):

            scores[product_id] = (
                scores.get(
                    product_id,
                    0.0,
                )
                + 1.0
                / (
                    k + rank
                )
            )

        # ====================================================
        # Keyword Search
        # ====================================================

        for rank, product_id in enumerate(
            keyword_ranked_ids,
            start=1,
        ):

            scores[product_id] = (
                scores.get(
                    product_id,
                    0.0,
                )
                + 1.0
                / (
                    k + rank
                )
            )

        # ====================================================
        # 排序
        # ====================================================

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            product_id
            for product_id, _score
            in ranked[:limit]
        ]

    # ========================================================
    # LLM Rerank
    # ========================================================

    async def _rerank(
        self,
        profile: UserProfile | None,
        candidates: list[Product],
        num_items: int,
    ) -> list[str]:

        if not profile:

            return [
                product.product_id
                for product in candidates[
                    :num_items
                ]
            ]

        if not candidates:
            return []

        profile_summary = {
            "preferred_categories": (
                profile.preferred_categories
                or []
            ),
            "segments": [
                segment.value
                for segment in (
                    profile.segments
                    or []
                )
            ],
            "price_range": (
                profile.price_range
            ),
            "recent_views": (
                profile.recent_views
                or []
            )[:5],
            "recent_purchases": (
                profile.recent_purchases
                or []
            )[:5],
            "real_time_tags": (
                profile.real_time_tags
                or []
            ),
        }

        candidate_summary: list[
            dict[str, Any]
        ] = []

        for product in candidates:

            candidate_summary.append(
                {
                    "id": product.product_id,
                    "name": product.name,
                    "category": product.category,
                    "price": product.price,
                }
            )

        prompt = RERANK_PROMPT.format(
            num_items=num_items,
            user_profile=json.dumps(
                profile_summary,
                ensure_ascii=False,
            ),
            candidates=json.dumps(
                candidate_summary,
                ensure_ascii=False,
            ),
        )

        messages = [
            SystemMessage(
                content=(
                    "你是电商推荐排序专家。"
                    "你的任务是对候选商品进行相关性排序。"
                    "只能从候选商品中选择。"
                    "不能创建新的商品ID。"
                    "必须严格返回JSON数组。"
                )
            ),
            HumanMessage(
                content=prompt
            ),
        ]

        print(
            "[Rerank] candidates=",
            len(candidates),
        )

        llm_start = time.perf_counter()

        try:

            response = await asyncio.wait_for(
                self.llm.ainvoke(messages),
                timeout=RERANK_TIMEOUT,
            )

            llm_latency = (
                time.perf_counter()
                - llm_start
            ) * 1000

            print(
                "[Rerank] llm_success",
                "latency_ms=",
                f"{llm_latency:.1f}",
            )

        except asyncio.TimeoutError:

            llm_latency = (
                time.perf_counter()
                - llm_start
            ) * 1000

            print(
                "[Rerank] llm_timeout",
                "timeout=",
                f"{RERANK_TIMEOUT}s",
                "latency_ms=",
                f"{llm_latency:.1f}",
                "fallback=true",
            )

            try:

                print(
                    "[Rerank] "
                    "Retrying LLM rerank once..."
                )

                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=RERANK_TIMEOUT,
                )

                llm_latency = (
                    time.perf_counter()
                    - llm_start
                ) * 1000

                print(
                    "[Rerank] "
                    "llm_success (retry)",
                    f"latency_ms={llm_latency:.1f}",
                )

            except asyncio.TimeoutError:

                output_count = len(
                    candidates[:num_items]
                )

                self._record_rerank_metric(
                    llm_latency,
                    False,
                    True,
                    True,
                    len(candidates),
                    output_count,
                )

                return [
                    product.product_id
                    for product in candidates[
                        :num_items
                    ]
                ]

            except Exception as exc:

                llm_latency = (
                    time.perf_counter()
                    - llm_start
                ) * 1000

                print(
                    "[Rerank] "
                    "llm_error (retry):",
                    str(exc),
                )

                output_count = len(
                    candidates[:num_items]
                )

                self._record_rerank_metric(
                    llm_latency,
                    False,
                    False,
                    True,
                    len(candidates),
                    output_count,
                )

                return [
                    product.product_id
                    for product in candidates[
                        :num_items
                    ]
                ]

        except Exception as exc:

            llm_latency = (
                time.perf_counter()
                - llm_start
            ) * 1000

            print(
                "[Rerank] llm_error",
                str(exc),
                "fallback=true",
            )

            output_count = len(
                candidates[:num_items]
            )

            self._record_rerank_metric(
                llm_latency,
                False,
                False,
                True,
                len(candidates),
                output_count,
            )

            return [
                product.product_id
                for product in candidates[
                    :num_items
                ]
            ]

        # ====================================================
        # 获取 LLM 返回内容
        # ====================================================

        raw_content = response.content

        if isinstance(
            raw_content,
            list,
        ):

            raw = "".join(
                str(item)
                for item in raw_content
            ).strip()

        else:

            raw = str(
                raw_content
            ).strip()

        print(
            "[ProductRecAgent] LLM返回:",
            raw,
        )

        # ====================================================
        # JSON解析
        # ====================================================

        try:

            if raw.startswith("```"):

                raw = re.sub(
                    r"^```(?:json)?\s*",
                    "",
                    raw,
                    flags=re.IGNORECASE,
                )

                raw = re.sub(
                    r"\s*```$",
                    "",
                    raw,
                ).strip()

            if not raw.startswith("["):

                start_index = raw.find("[")

                end_index = raw.rfind("]")

                if (
                    start_index != -1
                    and end_index != -1
                    and end_index > start_index
                ):

                    raw = raw[
                        start_index:
                        end_index + 1
                    ]

            result = json.loads(
                raw
            )

            if not isinstance(
                result,
                list,
            ):

                raise TypeError(
                    "LLM返回结果不是JSON数组"
                )

            candidate_ids = {
                product.product_id
                for product in candidates
            }

            valid_ids: list[str] = []

            for product_id in result:

                if (
                    isinstance(
                        product_id,
                        str,
                    )
                    and product_id
                    in candidate_ids
                    and product_id
                    not in valid_ids
                ):

                    valid_ids.append(
                        product_id
                    )

                if len(valid_ids) >= num_items:
                    break

            if not valid_ids:

                print(
                    "[ProductRecAgent] "
                    "LLM没有返回有效商品ID"
                )

                output_count = len(
                    candidates[:num_items]
                )

                self._record_rerank_metric(
                    llm_latency,
                    False,
                    False,
                    True,
                    len(candidates),
                    output_count,
                )

                return [
                    product.product_id
                    for product in candidates[
                        :num_items
                    ]
                ]

            valid_ids = valid_ids[
                :num_items
            ]

            self._record_rerank_metric(
                llm_latency,
                True,
                False,
                False,
                len(candidates),
                len(valid_ids),
            )

            return valid_ids

        except (
            json.JSONDecodeError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:

            print(
                "[ProductRecAgent] "
                "LLM Rerank解析失败:",
                str(exc),
            )

            output_count = len(
                candidates[:num_items]
            )

            self._record_rerank_metric(
                llm_latency,
                False,
                False,
                True,
                len(candidates),
                output_count,
            )

            return [
                product.product_id
                for product in candidates[
                    :num_items
                ]
            ]

