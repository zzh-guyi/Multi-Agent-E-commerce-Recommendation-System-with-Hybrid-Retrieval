
import sys, os, asyncio
os.chdir("/app")
sys.path.insert(0, "/app")

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

print("=== Tool Calling 验证测试 ===")
print("LLM:", os.environ.get("ECOM_LLM_MODEL", "not set"), "@", os.environ.get("ECOM_LLM_BASE_URL", "not set"))

# Test 1: bind_tools 正确用法
@tool
def test_tool(x: str) -> str:
    """测试工具"""
    return "result: " + x

try:
    llm = ChatOpenAI(
        api_key=os.environ.get("ECOM_LLM_API_KEY", ""),
        base_url=os.environ.get("ECOM_LLM_BASE_URL", ""),
        model=os.environ.get("ECOM_LLM_MODEL", ""),
        temperature=0,
        max_tokens=300,
    ).bind_tools([test_tool])
    print("OK: bind_tools 无警告")
except Exception as e:
    print("FAIL bind_tools:", e)

# Test 2: ProductRecAgent
try:
    from agents.product_rec_agent import ProductRecAgent
    agent = ProductRecAgent(enable_tool_calling=True)
    print("OK: ProductRecAgent 初始化成功")
    print("  llm_with_tools 类型:", type(agent.llm_with_tools).__name__)
    print("  tools:", [t.name for t in agent.tools])
except Exception as e:
    print("FAIL ProductRecAgent:", e)

# Test 3: 实际 Tool Calling
async def test_tool_call():
    @tool
    def search_products(query: str, limit: int = 5) -> str:
        """搜索商品"""
        return "[{\"product_id\": \"TEST001\", \"name\": \"手机\", \"price\": 1999.0}]"
    
    llm = ChatOpenAI(
        api_key=os.environ.get("ECOM_LLM_API_KEY", ""),
        base_url=os.environ.get("ECOM_LLM_BASE_URL", ""),
        model=os.environ.get("ECOM_LLM_MODEL", ""),
        temperature=0,
        max_tokens=300,
    ).bind_tools([search_products])
    
    messages = [
        SystemMessage(content="你是搜索助手，请用 search_products 工具搜索商品。"),
        HumanMessage(content="帮我搜索手机"),
    ]
    try:
        response = await llm.ainvoke(messages)
        print("OK: LLM 响应获取成功")
        print("  响应类型:", type(response).__name__)
        if hasattr(response, "tool_calls") and response.tool_calls:
            print("  tool_calls 数量:", len(response.tool_calls))
            for tc in response.tool_calls:
                print("    Tool:", tc.get("name"), "Args:", tc.get("args"))
        else:
            print("  无 tool_calls，直接文本响应:", str(response.content)[:200])
    except Exception as e:
        print("FAIL LLM call:", type(e).__name__, str(e)[:300])

asyncio.run(test_tool_call())
print("=== 测试完成 ===")
