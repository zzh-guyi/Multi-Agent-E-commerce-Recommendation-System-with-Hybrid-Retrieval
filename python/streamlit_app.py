import streamlit as st
import requests
import json

st.set_page_config(page_title="AI 电商推荐系统", layout="wide")

st.title("🛒 多 Agent 电商推荐系统")

# 用户输入
user_id = st.text_input("用户 ID", value="user_001")
scene = st.selectbox("推荐场景", ["homepage", "detail", "cart"])
num_items = st.slider("推荐数量", 1, 10, 5)

# 上下文输入
recent_views = st.text_input("最近浏览品类（逗号分隔）", "手机, 耳机")
avg_order = st.number_input("平均客单价", value=500)

if st.button("🚀 获取推荐", type="primary"):
    with st.spinner("正在调用 4 个 Agent 协同推荐..."):
        try:
            payload = {
                "user_id": user_id,
                "scene": scene,
                "num_items": num_items,
                "context": {
                    "recent_views": [v.strip() for v in recent_views.split(",")],
                    "avg_order_amount": avg_order
                }
            }
            response = requests.post(
                "http://localhost:8000/api/v1/recommend",
                json=payload,
                timeout=120
            )
            data = response.json()

            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.metric("总耗时", f"{data.get('total_latency_ms', 0):.0f} ms")
                st.metric("推荐商品数", len(data.get("products", [])))

            with col2:
                st.metric("实验分组", data.get("experiment_group", "control"))
                st.metric("Agent 成功率",
                          f"{sum(1 for a in data.get('agent_results', {}).values() if a.get('success'))}/{len(data.get('agent_results', {}))}")

            # 商品展示
            st.subheader("📦 推荐商品")
            products = data.get("products", [])
            copies = {c.get("product_id"): c.get("copy") for c in data.get("marketing_copies", [])}

            cols = st.columns(min(4, len(products)))
            for i, product in enumerate(products[:4]):
                with cols[i % 4]:
                    # 这里用 st.markdown 搭配边框样式，模拟卡片效果
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:10px;">
                        <b>{product.get('name')}</b><br>
                        💰 ¥{product.get('price')}<br>
                        📂 {product.get('category')}<br>
                        🏷️ {', '.join(product.get('tags', []))}<br>
                        <hr style="margin:10px 0;">
                        📝 {copies.get(product.get('product_id'), '')}
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"请求失败: {e}")