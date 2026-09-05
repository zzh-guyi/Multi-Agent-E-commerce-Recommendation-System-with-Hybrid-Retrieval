from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


products = [
    {
        "product_id": "P016",
        "name": "Redmi Buds 6",
        "category": "耳机",
        "price": 299,
        "tags": ["无线", "蓝牙", "性价比"],
    },
    {
        "product_id": "P017",
        "name": "华为 FreeBuds 6",
        "category": "耳机",
        "price": 499,
        "tags": ["无线", "降噪", "音乐"],
    },
    {
        "product_id": "P018",
        "name": "小米平板7",
        "category": "平板",
        "price": 499,
        "tags": ["学习", "娱乐", "性价比"],
    },
    {
        "product_id": "P019",
        "name": "荣耀平板X9",
        "category": "平板",
        "price": 399,
        "tags": ["学习", "娱乐", "大屏"],
    },
    {
        "product_id": "P020",
        "name": "Redmi Note 14",
        "category": "手机",
        "price": 499,
        "tags": ["性价比", "拍照", "性能"],
    },
    {
        "product_id": "P021",
        "name": "荣耀畅玩50",
        "category": "手机",
        "price": 399,
        "tags": ["性价比", "大电池", "日常使用"],
    },
    {
        "product_id": "P022",
        "name": "绿联100W氮化镓充电器",
        "category": "配件",
        "price": 199,
        "tags": ["快充", "氮化镓", "便携"],
    },
    {
        "product_id": "P023",
        "name": "倍思65W充电器",
        "category": "配件",
        "price": 159,
        "tags": ["快充", "便携", "性价比"],
    },
    {
        "product_id": "P024",
        "name": "iPad Air M3",
        "category": "平板",
        "price": 4799,
        "tags": ["学习", "办公", "苹果"],
    },
    {
        "product_id": "P025",
        "name": "Sony WH-1000XM6",
        "category": "耳机",
        "price": 2499,
        "tags": ["降噪", "头戴", "无线"],
    },
    {
        "product_id": "P026",
        "name": "OPPO Enco X3",
        "category": "耳机",
        "price": 499,
        "tags": ["降噪", "无线", "蓝牙"],
    },
    {
        "product_id": "P027",
        "name": "Redmi Buds 6 Pro",
        "category": "耳机",
        "price": 399,
        "tags": ["降噪", "无线", "性价比"],
    },
    {
        "product_id": "P028",
        "name": "荣耀平板MagicPad",
        "category": "平板",
        "price": 499,
        "tags": ["学习", "办公", "大屏"],
    },
    {
        "product_id": "P029",
        "name": "小米平板6",
        "category": "平板",
        "price": 299,
        "tags": ["娱乐", "学习", "性价比"],
    },
    {
        "product_id": "P030",
        "name": "一加 Ace 5",
        "category": "手机",
        "price": 499,
        "tags": ["性能", "游戏", "性价比"],
    },
    {
        "product_id": "P031",
        "name": "Redmi Note 14 Pro",
        "category": "手机",
        "price": 399,
        "tags": ["拍照", "性能", "性价比"],
    },
    {
        "product_id": "P032",
        "name": "倍思100W充电器",
        "category": "配件",
        "price": 299,
        "tags": ["快充", "便携", "氮化镓"],
    },
    {
        "product_id": "P033",
        "name": "绿联65W充电器",
        "category": "配件",
        "price": 199,
        "tags": ["快充", "便携", "性价比"],
    },
    {
        "product_id": "P034",
        "name": "漫步者NeoBuds",
        "category": "耳机",
        "price": 459,
        "tags": ["无线", "降噪", "音乐"],
    },
    {
        "product_id": "P035",
        "name": "华为MatePad SE",
        "category": "平板",
        "price": 399,
        "tags": ["学习", "娱乐", "性价比"],
    },
]


def main():
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    texts = [
        (
            f"商品名称：{p['name']}；"
            f"商品类别：{p['category']}；"
            f"商品价格：{p['price']}；"
            f"商品标签：{', '.join(p['tags'])}"
        )
        for p in products
    ]

    print(f"准备插入 {len(products)} 个商品")

    embeddings = embedding_service.embed_batch(texts)

    print(f"Embedding数量: {len(embeddings)}")
    print(f"Embedding维度: {len(embeddings[0])}")

    product_ids = [p["product_id"] for p in products]

    vector_store.insert(
        product_ids=product_ids,
        embeddings=embeddings,
    )

    print("P016-P035 已插入 Milvus")


if __name__ == "__main__":
    main()