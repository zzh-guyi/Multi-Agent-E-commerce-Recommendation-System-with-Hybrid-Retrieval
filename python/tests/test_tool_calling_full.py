"""
Tool Calling 完整集成测试

测试内容：
1. LangChain / LangChain Core / LangChain OpenAI 版本
2. Tool 是否能够正常导入
3. bind_tools() 是否正常工作
4. ProductRecAgent(enable_tool_calling=True) 是否正常初始化
5. 独立验证：
   LLM -> tool_calls -> Tool -> ToolMessage -> LLM final
6. ProductRecAgent._tool_rewrite_query() 实际调用
7. 默认 ProductRecAgent() 是否保持兼容
8. 不因为 ChatOpenAI 内部属性差异导致测试脚本本身崩溃

运行方式：

    docker compose exec api python test_tool_calling_full.py

注意：
- 本测试不会修改业务数据
- 本测试不会修改数据库
- 本测试不会修改 Milvus 数据
- 本测试不会修改 Supervisor / Graph
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import warnings


# ============================================================
# 基础环境
# ============================================================

os.chdir("/app")

if "/app" not in sys.path:
    sys.path.insert(0, "/app")


# ============================================================
# 辅助函数
# ============================================================


def safe_attr(obj, *names, default=None):
    """
    安全读取对象属性。

    不同版本的 ChatOpenAI 内部属性名称可能不同，
    因此测试脚本不能直接访问 a.llm.base_url。
    """
    for name in names:
        try:
            value = getattr(obj, name)
            if value is not None:
                return value
        except Exception:
            pass

    return default


def print_exception(prefix: str, exc: BaseException):
    """
    打印完整异常诊断信息。
    """
    print()
    print(prefix)
    print("  exception type : " + type(exc).__name__)
    print("  exception repr : " + repr(exc))
    print("  exception str  : " + str(exc))
    print()
    print("  traceback:")
    traceback.print_exc()
    print()


def truncate(value, max_length: int = 1000) -> str:
    """
    防止测试输出过长。
    """
    text = str(value)

    if len(text) <= max_length:
        return text

    return text[:max_length] + "...(truncated)"


# ============================================================
# 环境信息
# ============================================================


def print_environment():
    print("=" * 65)
    print("  Tool Calling 完整集成测试")
    print("=" * 65)
    print()

    print("[环境]")

    print("  WORKDIR : " + os.getcwd())

    model = os.getenv("ECOM_LLM_MODEL", "unknown")
    base_url = os.getenv("ECOM_LLM_BASE_URL", "unknown")
    api_key = os.getenv("ECOM_LLM_API_KEY")

    print("  LLM     : " + model + " @ " + base_url)

    if api_key:
        print("  API_KEY : set")
    else:
        print("  API_KEY : NOT SET")

    print()


# ============================================================
# [1] 依赖检查
# ============================================================


def dependency_check():
    print("[1] 依赖检查")

    try:
        import langchain

        print("  OK  langchain " + langchain.__version__)
    except Exception as exc:
        print("  FAIL langchain")
        print_exception("  dependency error:", exc)

    try:
        import langchain_core

        print("  OK  langchain_core " + langchain_core.__version__)
    except Exception as exc:
        print("  FAIL langchain_core")
        print_exception("  dependency error:", exc)

    try:
        import langchain_openai

        print("  OK  langchain_openai " + langchain_openai.__version__)
    except Exception as exc:
        print("  FAIL langchain_openai")
        print_exception("  dependency error:", exc)

    imports = [
        ("tools.product_tools", "tools.product_tools"),
        ("tools.inventory_tools", "tools.inventory_tools"),
        ("agents.product_rec_agent", "agents.product_rec_agent"),
        ("agents.inventory_agent", "agents.inventory_agent"),
    ]

    for module_name, display_name in imports:
        try:
            __import__(module_name)
            print("  OK  import " + display_name)
        except Exception as exc:
            print("  FAIL import " + display_name)
            print_exception("  import error:", exc)

    print()


# ============================================================
# [2] Tool 导入
# ============================================================


def tool_import_check():
    print("[2] Tool 导入")

    try:
        from tools.product_tools import (
            search_products,
            get_product_detail,
        )

        from tools.inventory_tools import check_stock

        print(
            "  search_products   -> "
            + str(getattr(search_products, "name", "UNKNOWN"))
        )

        print(
            "  get_product_detail-> "
            + str(getattr(get_product_detail, "name", "UNKNOWN"))
        )

        print(
            "  check_stock       -> "
            + str(getattr(check_stock, "name", "UNKNOWN"))
        )

        print()

        return (
            search_products,
            get_product_detail,
            check_stock,
        )

    except Exception as exc:
        print_exception("  Tool 导入失败:", exc)
        print()

        return None, None, None


# ============================================================
# [3] bind_tools 警告检查
# ============================================================


def bind_tools_warning_check(search_products):
    print("[3] bind_tools 警告检查")

    if search_products is None:
        print("  SKIP: search_products unavailable")
        print()
        return None

    try:
        from langchain_openai import ChatOpenAI

        model = os.getenv(
            "ECOM_LLM_MODEL",
            "deepseek-v4-flash",
        )

        base_url = os.getenv(
            "ECOM_LLM_BASE_URL",
            "https://api.deepseek.com",
        )

        api_key = os.getenv("ECOM_LLM_API_KEY")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            llm = ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0,
                max_tokens=300,
            )

            llm_with_tools = llm.bind_tools(
                [search_products]
            )

        tool_warnings = []

        for warning in caught:
            message = str(warning.message)

            if "tools is not default parameter" in message:
                tool_warnings.append(message)

        if tool_warnings:
            print("  FAIL: detected old tools= constructor warning")

            for message in tool_warnings:
                print("    " + message)

        else:
            print("  OK: no warning")

        print(
            "  llm type: "
            + type(llm_with_tools).__name__
        )

        print()

        return llm_with_tools

    except Exception as exc:
        print_exception("  bind_tools 测试失败:", exc)
        print()

        return None


# ============================================================
# [4] ProductRecAgent 初始化
# ============================================================


def product_agent_init_check():
    print("[4] ProductRecAgent 初始化")

    try:
        from agents.product_rec_agent import ProductRecAgent

        agent = ProductRecAgent(
            enable_tool_calling=True
        )

        print(
            "  enable_tool_calling : "
            + str(agent.enable_tool_calling)
        )

        print(
            "  llm_with_tools type : "
            + (
                type(agent.llm_with_tools).__name__
                if agent.llm_with_tools is not None
                else "None"
            )
        )

        print(
            "  tools               : "
            + str(
                [
                    getattr(tool, "name", "?")
                    for tool in agent.tools
                ]
            )
        )

        print()

        return agent

    except Exception as exc:
        print_exception(
            "  ProductRecAgent 初始化失败:",
            exc,
        )

        return None


# ============================================================
# [5] 独立端到端 Tool Calling
# ============================================================


async def standalone_tool_calling_test(
    search_products,
):
    print("[5] 端到端 Tool Calling 测试")
    print(
        "    LLM -> tool_calls -> "
        "search_products -> Tool result -> LLM final"
    )
    print()

    if search_products is None:
        print("  SKIP: search_products unavailable")
        print()
        return

    try:
        from langchain_openai import ChatOpenAI

        from langchain_core.messages import (
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        model = os.getenv(
            "ECOM_LLM_MODEL",
            "deepseek-v4-flash",
        )

        base_url = os.getenv(
            "ECOM_LLM_BASE_URL",
            "https://api.deepseek.com",
        )

        api_key = os.getenv("ECOM_LLM_API_KEY")

        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0,
            max_tokens=500,
        ).bind_tools(
            [search_products]
        )

        messages = [
            SystemMessage(
                content=(
                    "你是一个商品搜索助手。"
                    "当用户要求搜索商品时，"
                    "必须调用 search_products 工具。"
                )
            ),
            HumanMessage(
                content="帮我搜索手机"
            ),
        ]

        print("  [LLM] 发送消息...")

        start = time.perf_counter()

        response = await llm.ainvoke(messages)

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        print(
            "  [LLM] 响应类型 : "
            + type(response).__name__
        )

        print(
            "  [LLM] 耗时     : "
            + f"{elapsed:.1f}ms"
        )

        tool_calls = getattr(
            response,
            "tool_calls",
            None,
        )

        print(
            "  [LLM] tool_calls: "
            + (
                str(tool_calls)
                if tool_calls
                else "(无)"
            )
        )

        if not tool_calls:
            print()
            print(
                "  WARNING: LLM 没有产生 tool_calls"
            )

            print(
                "  response content:"
            )
            print(
                "  "
                + truncate(
                    getattr(
                        response,
                        "content",
                        "",
                    ),
                    1000,
                )
            )

            print()

            return

        messages.append(response)

        for index, tool_call in enumerate(
            tool_calls
        ):
            tool_name = tool_call.get(
                "name"
            )

            tool_args = tool_call.get(
                "args",
                {},
            )

            tool_id = tool_call.get(
                "id"
            )

            print(
                f"  [Tool] [{index}] "
                f"name={tool_name} "
                f"args={tool_args}"
            )

            if tool_name != search_products.name:
                print(
                    "  WARNING: LLM 调用了未知 Tool: "
                    + str(tool_name)
                )

                continue

            try:
                tool_start = time.perf_counter()

                result = await search_products.ainvoke(
                    tool_args
                )

                tool_elapsed = (
                    time.perf_counter()
                    - tool_start
                ) * 1000

                print(
                    f"  [Tool] [{index}] "
                    f"耗时={tool_elapsed:.1f}ms"
                )

                print(
                    f"  [Tool] [{index}] result: "
                    + truncate(
                        result,
                        1200,
                    )
                )

                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                    )
                )

            except Exception as exc:
                print_exception(
                    f"  [Tool] [{index}] 执行失败:",
                    exc,
                )

                return

        print()
        print(
            "  [LLM] 发送 Tool 结果，"
            "等待最终响应..."
        )

        try:
            final_start = time.perf_counter()

            final_response = await llm.ainvoke(
                messages
            )

            final_elapsed = (
                time.perf_counter()
                - final_start
            ) * 1000

            print(
                "  [LLM] 最终响应类型 : "
                + type(final_response).__name__
            )

            print(
                "  [LLM] 最终响应耗时 : "
                + f"{final_elapsed:.1f}ms"
            )

            final_content = getattr(
                final_response,
                "content",
                "",
            )

            print(
                "  [LLM] 最终文本:"
            )

            print(
                truncate(
                    final_content,
                    3000,
                )
            )

        except Exception as exc:
            print_exception(
                "  [LLM] 最终响应失败:",
                exc,
            )

        print()

    except Exception as exc:
        print_exception(
            "  端到端 Tool Calling 测试失败:",
            exc,
        )

        print()


# ============================================================
# [6] ProductRecAgent._tool_rewrite_query
# ============================================================


async def run_rewrite_test(agent):
    print(
        "[6] ProductRecAgent._tool_rewrite_query 测试"
    )

    print(
        "    (使用 agent.llm_with_tools，"
        "同 [5] 相同的模型)"
    )

    if agent is None:
        print(
            "  SKIP: ProductRecAgent unavailable"
        )
        print()
        return

    try:
        print(
            "  agent type: "
            + type(agent).__name__
        )

        model_name = safe_attr(
            agent.llm,
            "model_name",
            "model",
            default="unknown",
        )

        print(
            "  llm.model_name: "
            + str(model_name)
        )

        # ChatOpenAI 版本差异：
        # 有些版本没有 base_url 属性，
        # 可能使用 openai_api_base。
        base_url = safe_attr(
            agent.llm,
            "openai_api_base",
            "base_url",
            default=None,
        )

        if base_url is None:
            base_url = os.getenv(
                "ECOM_LLM_BASE_URL",
                "unknown",
            )

        print(
            "  llm.base_url: "
            + str(base_url)
        )

        tools = getattr(
            agent,
            "tools",
            [],
        )

        print(
            "  tools: "
            + str(
                [
                    getattr(tool, "name", "?")
                    for tool in tools
                ]
            )
        )

        print(
            "  llm_with_tools is llm: "
            + str(
                agent.llm_with_tools
                is agent.llm
            )
        )

        print()

        # ----------------------------------------------------
        # 构造一个最小 Profile
        #
        # 不依赖数据库、不依赖 API。
        # 目的只是测试 _tool_rewrite_query 本身。
        # ----------------------------------------------------

        # Use actual UserProfile from project schemas
        from models.schemas import UserProfile

        profile = UserProfile(
            user_id="TEST_TOOL_CALLING",
            preferred_categories=["手机"],
            recent_views=["手机"],
            recent_purchases=[],
            real_time_tags={},
        )

        print(
            "  profile:"
        )

        print(
            "  "
            + json.dumps(
                profile.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )

        print()
        print(
            "  [ProductRecAgent] "
            "_tool_rewrite_query() 开始..."
        )

        start = time.perf_counter()

        try:
            query = await agent._tool_rewrite_query(
                profile
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            print()
            print(
                "  [ProductRecAgent] "
                "_tool_rewrite_query() 完成"
            )

            print(
                "  elapsed: "
                + f"{elapsed:.1f}ms"
            )

            print(
                "  Query:"
            )

            print(
                "  "
                + str(query)
            )

            print()

            if isinstance(query, str):
                print(
                    "  OK: Query 是字符串"
                )
            else:
                print(
                    "  WARNING: Query 类型不是字符串: "
                    + type(query).__name__
                )

        except Exception as exc:
            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            print()
            print(
                "  FAIL: "
                "_tool_rewrite_query() 异常"
            )

            print(
                "  elapsed: "
                + f"{elapsed:.1f}ms"
            )

            print_exception(
                "  [ProductRecAgent] 详细异常:",
                exc,
            )

    except Exception as exc:
        print_exception(
            "  [6] 测试脚本自身异常:",
            exc,
        )

    print()


# ============================================================
# [7] 默认模式兼容性
# ============================================================


def default_mode_check():
    print("[7] 默认模式兼容性")

    try:
        from agents.product_rec_agent import (
            ProductRecAgent,
        )

        agent = ProductRecAgent()

        print(
            "  enable_tool_calling : "
            + str(
                agent.enable_tool_calling
            )
        )

        print(
            "  llm_with_tools      : "
            + str(
                agent.llm_with_tools
            )
        )

        print(
            "  tools               : "
            + str(agent.tools)
        )

        print(
            "  llm (原有)          : "
            + type(agent.llm).__name__
        )

        # 明确验证默认行为
        if agent.enable_tool_calling is False:
            print(
                "  OK: 默认 enable_tool_calling=False"
            )
        else:
            print(
                "  WARNING: 默认 "
                "enable_tool_calling 不是 False"
            )

        if agent.llm_with_tools is None:
            print(
                "  OK: 默认 llm_with_tools=None"
            )
        else:
            print(
                "  WARNING: 默认模式仍创建了 "
                "llm_with_tools"
            )

        print()

    except Exception as exc:
        print_exception(
            "  默认模式测试失败:",
            exc,
        )

        print()


# ============================================================
# 主程序
# ============================================================


async def main():
    print_environment()

    dependency_check()

    (
        search_products,
        get_product_detail,
        check_stock,
    ) = tool_import_check()

    bind_tools_warning_check(
        search_products
    )

    agent = product_agent_init_check()

    await standalone_tool_calling_test(
        search_products
    )

    await run_rewrite_test(
        agent
    )

    default_mode_check()

    print("=" * 65)
    print("  测试完成")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
