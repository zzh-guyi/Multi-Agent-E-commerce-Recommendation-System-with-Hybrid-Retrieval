import time
import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


llm = ChatOpenAI(
    api_key="sk-556681f34ba14b65828770ddd606e6bc",
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
    temperature=0,
    max_tokens=100,
)


# 使用 ProductRecAgent 日志里打印出来的真实 Prompt
prompt = """你是商品推荐排序器。

用户偏好：
手机、耳机、平板、充电器

价格范围：
100-500元

候选商品：
[{"id":"P001","name":"iPhone 16 Pro","category":"手机","price":7999,"tags":["旗舰","新品"]},
{"id":"P002","name":"华为 Mate 70","category":"手机","price":5999,"tags":["旗舰","国产"]},
{"id":"P003","name":"AirPods Pro 3","category":"耳机","price":1899,"tags":["降噪","无线"]},
{"id":"P004","name":"Sony WH-1000XM6","category":"耳机","price":2499,"tags":["头戴","降噪"]},
{"id":"P005","name":"iPad Air M3","category":"平板","price":4799,"tags":["学习","办公"]},
{"id":"P006","name":"小米平板7 Pro","category":"平板","price":2499,"tags":["性价比","娱乐"]},
{"id":"P007","name":"Anker 140W充电器","category":"配件","price":399,"tags":["快充","便携"]},
{"id":"P008","name":"绿联氮化镓65W","category":"配件","price":129,"tags":["快充","性价比"]},
{"id":"P009","name":"大疆Mini 4 Pro","category":"无人机","price":4788,"tags":["航拍","便携"]},
{"id":"P010","name":"Switch 2","category":"游戏机","price":2499,"tags":["新品","游戏"]}]

选择最符合用户偏好的5个商品。
优先考虑用户偏好和价格范围。

只返回JSON数组，例如：
["P007","P008","P003","P006","P004"]"""


async def main():
    t = time.perf_counter()

    print("开始测试...")
    print("Prompt长度:", len(prompt))

    response = await llm.ainvoke([
        SystemMessage(content="你是电商推荐排序专家。"),
        HumanMessage(content=prompt),
    ])

    elapsed = time.perf_counter() - t

    print("完整Prompt耗时:", f"{elapsed:.3f}s")
    print("返回:", response.content)


asyncio.run(main())