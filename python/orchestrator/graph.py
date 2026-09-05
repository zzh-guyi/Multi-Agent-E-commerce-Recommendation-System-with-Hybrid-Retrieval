from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agents import (
    InventoryAgent,
    MarketingCopyAgent,
    ProductRecAgent,
    UserProfileAgent,
)
from config import get_settings
from models.schemas import Product, UserProfile
from services.ab_test import ABTestEngine
from services.metrics import metrics_collector


def _merge_dicts(base: dict, update: dict) -> dict:
    return {**base, **update}


class PipelineState(TypedDict, total=False):
    request_id: str
    user_id: str
    scene: str
    num_items: int
    context: dict[str, Any]
    experiment_group: str
    user_profile: UserProfile | None
    raw_products: list[Product]
    ranked_products: list[Product]
    available_ids: set[str]
    final_products: list[Product]
    marketing_copies: list[dict[str, str]]
    agent_results: Annotated[dict[str, Any], _merge_dicts]
    current_node: str
    next_action: str
    iteration_count: int
    agent_visit_counts: dict[str, int]
    total_latency_ms: float
    _start_time: float
    _stage_latencies: dict[str, float]
    _routing_decisions: list[str]
    _supervisor_total_latency_ms: float


MAX_ITERATIONS = 20
MAX_AGENT_VISITS = 3
RERANK_CANDIDATE_LIMIT = 15
VALID_AGENT_NODES = {"user_profile", "product_recall", "rerank", "inventory", "marketing_copy"}

user_profile_agent = UserProfileAgent()
product_rec_agent_recall = ProductRecAgent(enable_tool_calling=True)
product_rec_agent_rerank = ProductRecAgent(enable_tool_calling=False)
marketing_copy_agent = MarketingCopyAgent()
inventory_agent = InventoryAgent()
ab_engine = ABTestEngine()

_settings = get_settings()
_supervisor_llm = ChatOpenAI(
    api_key=_settings.llm_api_key,
    base_url=_settings.llm_base_url,
    model=_settings.llm_model,
    temperature=0,
    max_tokens=100,
)

ROUTING_PROMPT = """你是电商推荐系统的路由专家。

当前系统状态：
- 用户画像：{has_profile}
- 召回商品数量：{recall_count}
- 排序后商品数量：{ranked_count}
- 可用库存商品：{available_count}
- 最终商品数量：{final_count}
- 营销文案：{has_copies}
- 已完成节点：{visited}
- 当前轮次：{iteration}

请根据当前状态，决定下一步应该执行哪个 Agent：
- user_profile：生成用户画像
- product_recall：召回商品（Tool Calling + Hybrid Retrieval）
- rerank：LLM 重排序候选商品
- inventory：检查商品库存
- marketing_copy：生成营销文案
- finish：流程完成，进入聚合阶段

输出要求：
只输出 JSON，格式为：
{{"next": "agent_name"}}

注意：
1. 如果流程已经完整（有 final_products 且有 marketing_copies），必须返回 finish
2. 如果没有 user_profile，应该先执行 user_profile
3. 如果没有召回商品，应该执行 product_recall
4. 如果有召回商品但没有排序，应该执行 rerank
5. 如果已排序但没有库存检查，应该执行 inventory
6. 如果库存检查完成但没有营销文案，应该执行 marketing_copy
7. 不能超过最大迭代次数
"""


def _get_visited(state: PipelineState) -> str:
    counts = state.get("agent_visit_counts", {})
    return ", ".join(f"{k}({v})" for k, v in counts.items()) if counts else "无"


def _build_routing_context(state: PipelineState) -> str:
    profile = state.get("user_profile")
    raw = state.get("raw_products", [])
    ranked = state.get("ranked_products", [])
    available = state.get("available_ids", set())
    final = state.get("final_products", [])
    copies = state.get("marketing_copies", [])
    iteration = state.get("iteration_count", 0)

    return ROUTING_PROMPT.format(
        has_profile="有" if profile else "无",
        recall_count=len(raw),
        ranked_count=len(ranked),
        available_count=len(available),
        final_count=len(final),
        has_copies="有" if copies else "无",
        visited=_get_visited(state),
        iteration=iteration,
    )


async def _llm_route(state: PipelineState) -> str:
    try:
        context = _build_routing_context(state)
        messages = [
            SystemMessage(content="你是一个推荐系统路由专家。只输出 JSON。"),
            HumanMessage(content=context),
        ]
        response = await _supervisor_llm.ainvoke(messages)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.strip("`").lstrip("json").strip()

        result = json.loads(content)
        next_action = result.get("next", "finish")

        if next_action not in VALID_AGENT_NODES | {"finish"}:
            next_action = "finish"

        return next_action
    except Exception as e:
        print(f"[Supervisor] LLM 路由失败: {e}，使用确定性路由")
        return _deterministic_route(state)


def _is_flow_complete(state: PipelineState) -> bool:
    final = state.get("final_products", [])
    copies = state.get("marketing_copies", [])
    return len(final) > 0 and len(copies) > 0


def _deterministic_route(state: PipelineState) -> str:
    profile = state.get("user_profile")
    raw = state.get("raw_products", [])
    ranked = state.get("ranked_products", [])
    available = state.get("available_ids", set())
    final = state.get("final_products", [])
    copies = state.get("marketing_copies", [])
    iteration = state.get("iteration_count", 0)

    if iteration > MAX_ITERATIONS:
        return "finish"
    if _is_flow_complete(state):
        return "finish"

    if not profile:
        return "user_profile"
    if not raw:
        return "product_recall"
    if not ranked:
        return "rerank"
    if not available:
        return "inventory"
    if not copies:
        return "marketing_copy"

    return "finish"


def _validate_route(state: PipelineState, next_action: str) -> str:
    profile = state.get("user_profile")
    raw = state.get("raw_products", [])
    ranked = state.get("ranked_products", [])
    available = state.get("available_ids", set())
    final = state.get("final_products", [])

    if not profile and next_action in {"inventory", "marketing_copy"}:
        return "user_profile"
    if not raw and next_action in {"rerank", "inventory", "marketing_copy"}:
        return "product_recall"
    if not ranked and next_action in {"inventory", "marketing_copy"}:
        return "rerank"
    if not available and next_action == "marketing_copy":
        return "inventory"

    return next_action


async def init_node(state: PipelineState) -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "_start_time": time.perf_counter(),
        "agent_results": {},
        "experiment_group": ab_engine.assign(state["user_id"]).get("group", "control"),
        "current_node": "init",
        "next_action": "supervisor",
        "iteration_count": 0,
        "agent_visit_counts": {},
        "_routing_decisions": [],
        "_supervisor_total_latency_ms": 0.0,
    }


async def supervisor_node(state: PipelineState) -> dict:
    _supervisor_start = time.perf_counter()
    iteration = state.get("iteration_count", 0) + 1

    if iteration > MAX_ITERATIONS:
        print(f"[Supervisor] 达到最大迭代次数 {MAX_ITERATIONS}，强制结束")
        return {
            "next_action": "finish",
            "iteration_count": iteration,
            "current_node": "supervisor",
        }

    # 确定性路由优先，避免每次路由都调用 LLM
    next_action = _deterministic_route(state)
    # 仅在确定性路由无法决策时才回退到 LLM（边界情况兜底）
    if next_action not in VALID_AGENT_NODES | {"finish"}:
        next_action = await _llm_route(state)
    next_action = _validate_route(state, next_action)

    if _is_flow_complete(state):
        next_action = "finish"

    print(f"[Supervisor] 第 {iteration} 轮路由 -> {next_action}")
    # Update routing decisions for state accumulation
    routing_decisions = list(state.get("_routing_decisions") or [])
    routing_decisions.append(next_action)


    _supervisor_latency_ms = (time.perf_counter() - _supervisor_start) * 1000
    return {
        "next_action": next_action,
        "iteration_count": iteration,
        "current_node": "supervisor",
        "supervisor_latency_ms": round(_supervisor_latency_ms, 1),
        "_routing_decisions": routing_decisions,
        "_supervisor_total_latency_ms": round(state.get("_supervisor_total_latency_ms", 0.0) + _supervisor_latency_ms, 1),
    }


async def user_profile_node(state: PipelineState) -> dict:
    result = await user_profile_agent.run(
        user_id=state["user_id"],
        context=state.get("context", {}),
    )
    counts = state.get("agent_visit_counts", {}).copy()
    counts["user_profile"] = counts.get("user_profile", 0) + 1

    return {
        "user_profile": getattr(result, "profile", None),
        "agent_results": {"user_profile": result},
        "current_node": "user_profile",
        "agent_visit_counts": counts,
    }


async def product_recall_node(state: PipelineState) -> dict:
    result = await product_rec_agent_recall.run(
        user_profile=state.get("user_profile"),
        num_items=state.get("num_items", 10) * 2,
    )
    counts = state.get("agent_visit_counts", {}).copy()
    counts["product_recall"] = counts.get("product_recall", 0) + 1

    return {
        "raw_products": getattr(result, "products", []),
        "agent_results": {"product_recall": result},
        "current_node": "product_recall",
        "agent_visit_counts": counts,
    }


async def rerank_node(state: PipelineState) -> dict:
    raw_products = state.get("raw_products", [])
    user_profile = state.get("user_profile")
    num_items = state.get("num_items", 10)

    if not raw_products:
        counts = state.get("agent_visit_counts", {}).copy()
        counts["rerank"] = counts.get("rerank", 0) + 1
        return {
            "ranked_products": [],
            "agent_results": {"rerank": None},
            "current_node": "rerank",
            "agent_visit_counts": counts,
        }

    result = await product_rec_agent_rerank.run(
        user_profile=user_profile,
        candidates=raw_products[:RERANK_CANDIDATE_LIMIT],
        num_items=num_items,
    )
    ranked = getattr(result, "products", raw_products[:num_items])
    counts = state.get("agent_visit_counts", {}).copy()
    counts["rerank"] = counts.get("rerank", 0) + 1

    _rerank_latency = getattr(getattr(result, "data", {}), "get", lambda k, d=0.0: d)("rerank_latency_ms", 0.0)
    return {
        "ranked_products": ranked,
        "agent_results": {"rerank": result},
        "current_node": "rerank",
        "agent_visit_counts": counts,
        "rerank_latency_ms": _rerank_latency if isinstance(_rerank_latency, (int, float)) else 0.0,
    }


async def inventory_node(state: PipelineState) -> dict:
    products = state.get("ranked_products", [])
    if not products:
        products = state.get("raw_products", [])

    print(f"[Inventory] 开始检查库存，候选商品数={len(products)}")
    result = await inventory_agent.run(products=products)
    counts = state.get("agent_visit_counts", {}).copy()
    counts["inventory"] = counts.get("inventory", 0) + 1

    available = set(getattr(result, "available_products", []))
    print(f"[Inventory] 库存检查完成，可用商品数={len(available)}")

    return {
        "available_ids": available,
        "agent_results": {"inventory": result},
        "current_node": "inventory",
        "agent_visit_counts": counts,
        "next_action": "filter",
    }


async def filter_node(state: PipelineState) -> dict:
    _filter_start = time.perf_counter()
    ranked_products = state.get("ranked_products", [])
    available_ids = state.get("available_ids", set())
    num_items = state.get("num_items", 10)

    print(f"[Filter] 开始过滤，排序后商品数={len(ranked_products)}，可用库存={len(available_ids)}")
    final_products = [
        product
        for product in ranked_products
        if product.product_id in available_ids
    ]
    if not final_products:
        final_products = list(ranked_products)
        print(f"[Filter] 库存过滤无结果，使用全部排序商品")

    print(f"[Filter] 过滤完成，最终商品数={len(final_products)}")

    return {
        "final_products": final_products[:num_items],
        "current_node": "filter",
        "next_action": "supervisor",
        "filter_latency_ms": round((time.perf_counter() - _filter_start) * 1000, 1),
    }


async def marketing_copy_node(state: PipelineState) -> dict:
    final_products = state.get("final_products", [])
    if not final_products:
        counts = state.get("agent_visit_counts", {}).copy()
        counts["marketing_copy"] = counts.get("marketing_copy", 0) + 1
        return {
            "marketing_copies": [],
            "agent_results": {"marketing_copy": None},
            "current_node": "marketing_copy",
            "agent_visit_counts": counts,
        }

    result = await marketing_copy_agent.run(
        user_profile=state.get("user_profile"),
        products=final_products,
    )
    counts = state.get("agent_visit_counts", {}).copy()
    counts["marketing_copy"] = counts.get("marketing_copy", 0) + 1

    return {
        "marketing_copies": getattr(result, "copies", []),
        "agent_results": {"marketing_copy": result},
        "current_node": "marketing_copy",
        "agent_visit_counts": counts,
    }


async def aggregate_node(state: PipelineState) -> dict:
    start_time = state.get("_start_time")
    total_latency = (time.perf_counter() - start_time) * 1000 if start_time else 0.0
    try:
        routing_decisions = list(state.get("_routing_decisions", []))
        supervisor_latency = state.get("_supervisor_total_latency_ms", 0.0)
        metrics_collector.record_supervisor(supervisor_latency, list(routing_decisions))
    except Exception as exc:
        print(f"[Supervisor] Warning: record_supervisor failed: {exc}")
    return {"total_latency_ms": round(total_latency, 1)}

def route_supervisor(state: PipelineState) -> str:
    next_action = state.get("next_action", "finish")

    # 安全检查：filter 不是 LLM 路由目标，强制重定向到 finish
    if next_action == "filter":
        print("[Supervisor] 非法路由目标 filter，强制结束")
        return "finish"

    if next_action in VALID_AGENT_NODES:
        counts = state.get("agent_visit_counts", {})
        if counts.get(next_action, 0) >= MAX_AGENT_VISITS:
            print(f"[Supervisor] {next_action} 已达到最大访问次数 {MAX_AGENT_VISITS}，强制结束")
            return "finish"

    return next_action


def build_recommendation_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("init", init_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("user_profile", user_profile_node)
    graph.add_node("product_recall", product_recall_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("inventory", inventory_node)
    graph.add_node("filter", filter_node)
    graph.add_node("marketing_copy", marketing_copy_node)
    graph.add_node("aggregate", aggregate_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "user_profile": "user_profile",
            "product_recall": "product_recall",
            "rerank": "rerank",
            "inventory": "inventory",
            "marketing_copy": "marketing_copy",
            "finish": "aggregate",
        },
    )

    graph.add_edge("user_profile", "supervisor")
    graph.add_edge("product_recall", "supervisor")
    graph.add_edge("rerank", "supervisor")
    graph.add_edge("inventory", "filter")
    graph.add_edge("filter", "supervisor")
    graph.add_edge("marketing_copy", "supervisor")
    graph.add_edge("aggregate", END)

    return graph.compile()
