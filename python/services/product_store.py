from __future__ import annotations

import re
from typing import Any

from sqlalchemy import create_engine, text

from config import get_settings


class ProductStore:
    """MySQL product data access layer.

    负责商品在 MySQL 中的 CRUD 操作。

    同时提供：
    - 商品关键词检索
    - 商品批量查询

    注意：
    - ProductStore 只负责 MySQL
    - 不负责 Embedding
    - 不负责 Milvus
    - 不负责推荐排序
    """

    def __init__(self):
        settings = get_settings()

        self.engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
        )

    # ============================================================
    # 查询全部商品
    # ============================================================

    def get_all_products(self) -> list[dict[str, Any]]:
        """Get all products from MySQL."""

        sql = text("""
            SELECT
                product_id,
                name,
                category,
                price,
                description,
                brand,
                seller_id,
                stock,
                tags
            FROM products
            WHERE stock >= 0
            ORDER BY product_id
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(sql).mappings().all()

        return [dict(row) for row in rows]

    # ============================================================
    # 关键词搜索
    # ============================================================

    def keyword_search_products(
        self,
        query: str,
        limit: int = 20,
        product_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        商品关键词检索。

        使用 MySQL LIKE 实现中文商品 lexical retrieval。

        检索策略：
        1. Query 规范化
        2. 中文领域词提取
        3. 同义词扩展
        4. 多字段 LIKE 匹配
        5. 字段加权
        6. Token coverage 加分
        7. Query phrase 命中加分

        字段权重：

            name        8
            category    6
            tags        5
            brand       3
            description 2

        同时支持 product_ids 过滤。

        这是 Hybrid Search 中的 lexical / keyword retrieval 部分。
        """

        if not query or limit <= 0:
            return []

        query = str(query).strip()

        if not query:
            return []

        # --------------------------------------------------------
        # 1. Query 规范化
        # --------------------------------------------------------

        normalized_query = query.lower()

        # 去除换行
        normalized_query = re.sub(r"\s+", " ", normalized_query)

        # 去除常见价格/预算表达。
        #
        # 例如：
        #   预算500元以内的高性价比手机
        #
        # 最终重点保留：
        #   高性价比手机
        normalized_query = re.sub(
            r"\d+(?:\.\d+)?\s*(?:元|块|人民币)?",
            " ",
            normalized_query,
        )

        normalized_query = re.sub(
            r"(?:预算|价格)\s*(?:以内|以下|不超过|小于|低于)?",
            " ",
            normalized_query,
        )

        # 去除常见无意义连接词
        normalized_query = re.sub(
            r"(?:适合|推荐|的|和|与|以及|以及适合|用户|想要|需要|一款|一些)",
            " ",
            normalized_query,
        )

        # 标点统一为空格
        normalized_query = re.sub(
            r"[，。！？；：、,.!?;:\"'（）()\[\]【】<>《》/\\|]+",
            " ",
            normalized_query,
        )

        normalized_query = re.sub(
            r"\s+",
            " ",
            normalized_query,
        ).strip()

        if not normalized_query:
            normalized_query = query.lower()

        # --------------------------------------------------------
        # 2. 中文领域词 + 同义词
        #
        # 长词必须优先于短词。
        # --------------------------------------------------------

        synonym_map: dict[str, tuple[str, ...]] = {
            "高性价比": ("性价比",),
            "性价比高": ("性价比",),
            "便宜": ("性价比",),
            "低价": ("性价比",),
            "实惠": ("性价比",),

            "高性能": ("性能",),
            "性能强": ("性能",),
            "性能好": ("性能",),

            "轻薄": ("轻薄",),
            "轻便": ("便携",),
            "轻巧": ("便携",),
            "便携": ("便携",),

            "降噪耳机": ("降噪", "耳机"),
            "降噪": ("降噪",),

            "蓝牙耳机": ("蓝牙", "耳机"),
            "蓝牙": ("蓝牙",),

            "无线耳机": ("无线", "耳机"),
            "无线": ("无线",),

            "充电器": ("充电器",),
            "充电设备": ("充电器",),
            "快充": ("快充",),
            "快速充电": ("快充",),

            "平板电脑": ("平板",),
            "平板": ("平板",),

            "智能手机": ("手机",),
            "手机": ("手机",),

            "办公": ("办公",),
            "工作": ("办公",),

            "学习": ("学习",),
            "学生": ("学习",),

            "游戏": ("游戏",),
            "娱乐": ("娱乐",),

            "拍照": ("拍照",),
            "影像": ("拍照",),

            "运动": ("运动",),
            "户外": ("户外",),

            "摄影": ("摄影",),
            "航拍": ("航拍",),

            "鼠标": ("鼠标",),
            "键盘": ("键盘",),

            "显示器": ("显示器",),
            "笔记本": ("笔记本",),

            "耳机": ("耳机",),
            "设备": ("设备",),
            "配件": ("配件",),
        }

        # 按长度降序，保证最长词优先识别
        domain_terms = sorted(
            synonym_map.keys(),
            key=len,
            reverse=True,
        )

        # --------------------------------------------------------
        # 3. 提取领域词
        # --------------------------------------------------------

        matched_terms: list[str] = []
        remaining_text = normalized_query

        for term in domain_terms:
            if term in remaining_text:
                matched_terms.append(term)
                remaining_text = remaining_text.replace(
                    term,
                    " ",
                )

        # --------------------------------------------------------
        # 4. 从领域词进行同义词扩展
        # --------------------------------------------------------

        expanded_tokens: list[str] = []

        def add_token(token: str) -> None:
            token = token.strip().lower()

            if not token:
                return

            if len(token) < 2:
                return

            if token not in expanded_tokens:
                expanded_tokens.append(token)

        for term in matched_terms:
            add_token(term)

            for synonym in synonym_map.get(term, ()):
                add_token(synonym)

        # --------------------------------------------------------
        # 5. 处理没有被领域词覆盖的文本
        #
        # 不做大规模 2/3/4 字滑窗。
        # 只保留连续中文片段，并对较长片段做有限切分。
        # --------------------------------------------------------

        remaining_chunks = re.findall(
            r"[\u4e00-\u9fff]{2,}",
            remaining_text,
        )

        english_tokens = re.findall(
            r"[a-zA-Z0-9]+",
            normalized_query,
        )

        for chunk in remaining_chunks:
            if len(chunk) <= 6:
                add_token(chunk)
            else:
                # 对长中文片段做有限 bigram 提取，
                # 避免原实现产生大量 2/3/4 字滑窗噪声。
                for i in range(len(chunk) - 1):
                    piece = chunk[i : i + 2]

                    # 忽略常见无意义词
                    if piece in {
                        "以内",
                        "以下",
                        "适合",
                        "用户",
                        "推荐",
                        "相关",
                        "商品",
                        "产品",
                        "一些",
                        "一个",
                        "当前",
                    }:
                        continue

                    add_token(piece)

        for token in english_tokens:
            add_token(token)

        # --------------------------------------------------------
        # 6. 如果 tokenizer 没有得到有效 token，
        #    回退到原始 query
        # --------------------------------------------------------

        if not expanded_tokens:
            fallback = re.sub(
                r"\s+",
                " ",
                query.lower(),
            ).strip()

            if fallback:
                expanded_tokens = [fallback]

        # 防止 SQL 过长
        expanded_tokens = expanded_tokens[:15]

        # --------------------------------------------------------
        # 7. Query phrase
        #
        # 用于额外 exact phrase bonus。
        # --------------------------------------------------------

        phrase = re.sub(
            r"\s+",
            "",
            normalized_query,
        ).strip()

        # phrase 太长时不直接参与 LIKE，
        # 避免长 query 导致完全匹配不到。
        phrase_for_search = (
            phrase
            if 2 <= len(phrase) <= 30
            else ""
        )

        # --------------------------------------------------------
        # 8. 动态 SQL
        # --------------------------------------------------------

        fields = [
            ("name", 8),
            ("category", 6),
            ("tags", 5),
            ("brand", 3),
            ("description", 2),
        ]

        keyword_conditions: list[str] = []
        score_parts: list[str] = []

        params: dict[str, Any] = {}

        for token_index, token in enumerate(expanded_tokens):

            token_param = f"token_{token_index}"

            params[token_param] = f"%{token}%"

            # --------------------------------------------
            # 一个 token 命中任意字段即可进入候选集
            # --------------------------------------------

            field_conditions = [
                f"{field} LIKE :{token_param}"
                for field, _weight in fields
            ]

            keyword_conditions.append(
                "("
                + " OR ".join(field_conditions)
                + ")"
            )

            # --------------------------------------------
            # 字段加权
            # --------------------------------------------

            field_scores = []

            for field, weight in fields:

                field_scores.append(
                    f"""
                    CASE
                        WHEN {field} LIKE :{token_param}
                        THEN {weight}
                        ELSE 0
                    END
                    """
                )

            field_score_expression = " + ".join(
                field_scores
            )

            # --------------------------------------------
            # Token coverage：
            #
            # 每个 token 命中一次就获得额外奖励。
            # --------------------------------------------

            coverage_expression = f"""
                CASE
                    WHEN (
                        {" OR ".join(field_conditions)}
                    )
                    THEN 4
                    ELSE 0
                END
            """

            score_parts.append(
                f"""
                (
                    {field_score_expression}
                    + {coverage_expression}
                )
                """
            )

        # --------------------------------------------------------
        # 9. Phrase bonus
        # --------------------------------------------------------

        phrase_score = "0"

        if phrase_for_search:

            params["query_phrase"] = (
                f"%{phrase_for_search}%"
            )

            phrase_score = """
                CASE
                    WHEN REPLACE(
                        LOWER(
                            CONCAT(
                                COALESCE(name, ''),
                                COALESCE(category, ''),
                                COALESCE(tags, ''),
                                COALESCE(brand, ''),
                                COALESCE(description, '')
                            )
                        ),
                        ' ',
                        ''
                    ) LIKE :query_phrase
                    THEN 12
                    ELSE 0
                END
            """

        # --------------------------------------------------------
        # 10. product_id 过滤
        # --------------------------------------------------------

        product_filter = ""

        if product_ids:

            placeholders = ", ".join(
                f":product_id_{i}"
                for i in range(len(product_ids))
            )

            product_filter = (
                f"AND product_id IN ({placeholders})"
            )

            for i, product_id in enumerate(product_ids):
                params[f"product_id_{i}"] = product_id

        # --------------------------------------------------------
        # 11. 最终 score
        # --------------------------------------------------------

        if score_parts:
            score_expression = " + ".join(
                score_parts
            )
        else:
            score_expression = "0"

        # --------------------------------------------------------
        # 12. SQL
        # --------------------------------------------------------

        sql = text(
            f"""
            SELECT
                product_id,
                name,
                category,
                price,
                description,
                brand,
                seller_id,
                stock,
                tags,

                (
                    {score_expression}
                    + ({phrase_score})
                ) AS keyword_score

            FROM products

            WHERE stock > 0

            {product_filter}

            AND (
                {" OR ".join(keyword_conditions)}
            )

            ORDER BY
                keyword_score DESC,
                product_id ASC

            LIMIT :limit
            """
        )

        params["limit"] = int(limit)

        # --------------------------------------------------------
        # 13. 执行查询
        # --------------------------------------------------------

        with self.engine.connect() as conn:
            rows = conn.execute(
                sql,
                params,
            ).mappings().all()

        return [dict(row) for row in rows]

    # ============================================================
    # 根据商品 ID 查询单个商品
    # ============================================================

    def get_product_by_id(
        self,
        product_id: str,
    ) -> dict[str, Any] | None:
        """Get one product by product ID."""

        if not product_id:
            return None

        sql = text("""
            SELECT
                product_id,
                name,
                category,
                price,
                description,
                brand,
                seller_id,
                stock,
                tags
            FROM products
            WHERE product_id = :product_id
        """)

        with self.engine.connect() as conn:
            row = conn.execute(
                sql,
                {
                    "product_id": product_id,
                },
            ).mappings().first()

        if row is None:
            return None

        return dict(row)

    # ============================================================
    # 根据 ID 批量查询商品
    # ============================================================

    def get_products_by_ids(
        self,
        product_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Get products by product IDs."""

        if not product_ids:
            return []

        placeholders = ", ".join(
            f":id_{i}"
            for i in range(len(product_ids))
        )

        params = {
            f"id_{i}": product_id
            for i, product_id in enumerate(product_ids)
        }

        sql = text(f"""
            SELECT
                product_id,
                name,
                category,
                price,
                description,
                brand,
                seller_id,
                stock,
                tags
            FROM products
            WHERE product_id IN ({placeholders})
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(
                sql,
                params,
            ).mappings().all()

        return [dict(row) for row in rows]

    # ============================================================
    # 新增商品
    # ============================================================

    def insert_product(
        self,
        product: dict[str, Any],
    ) -> None:
        """Insert one product into MySQL."""

        tags = product.get("tags", "")

        if isinstance(tags, list):
            tags = ", ".join(
                str(tag)
                for tag in tags
            )

        sql = text("""
            INSERT INTO products (
                product_id,
                name,
                category,
                price,
                description,
                brand,
                seller_id,
                stock,
                tags
            )
            VALUES (
                :product_id,
                :name,
                :category,
                :price,
                :description,
                :brand,
                :seller_id,
                :stock,
                :tags
            )
        """)

        with self.engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "product_id": product["product_id"],
                    "name": product["name"],
                    "category": product["category"],
                    "price": product["price"],
                    "description": product.get(
                        "description",
                        "",
                    ),
                    "brand": product.get(
                        "brand",
                        "",
                    ),
                    "seller_id": product.get(
                        "seller_id",
                        "",
                    ),
                    "stock": product.get(
                        "stock",
                        0,
                    ),
                    "tags": tags,
                },
            )

    # ============================================================
    # 更新商品
    # ============================================================

    def update_product(
        self,
        product_id: str,
        product: dict[str, Any],
    ) -> None:
        """Update one product in MySQL."""

        if not product_id:
            raise ValueError("商品ID不能为空")

        if not product:
            raise ValueError("没有需要更新的商品字段")

        update_data = dict(product)

        if "tags" in update_data:

            tags = update_data["tags"]

            if isinstance(tags, list):

                update_data["tags"] = ", ".join(
                    str(tag)
                    for tag in tags
                )

        allowed_fields = {
            "name",
            "category",
            "price",
            "description",
            "brand",
            "seller_id",
            "stock",
            "tags",
        }

        update_data = {
            key: value
            for key, value in update_data.items()
            if key in allowed_fields
        }

        if not update_data:

            raise ValueError(
                "没有有效的商品更新字段"
            )

        set_clause = ", ".join(
            f"{field} = :{field}"
            for field in update_data
        )

        update_data["product_id"] = product_id

        sql = text(
            f"""
            UPDATE products
            SET {set_clause}
            WHERE product_id = :product_id
            """
        )

        with self.engine.begin() as conn:

            result = conn.execute(
                sql,
                update_data,
            )

            if result.rowcount == 0:

                raise ValueError(
                    f"商品不存在: {product_id}"
                )

    # ============================================================
    # 删除商品
    # ============================================================

    def delete_product(
        self,
        product_id: str,
    ) -> None:
        """Delete one product from MySQL."""

        if not product_id:
            raise ValueError("商品ID不能为空")

        sql = text("""
            DELETE FROM products
            WHERE product_id = :product_id
        """)

        with self.engine.begin() as conn:

            result = conn.execute(
                sql,
                {
                    "product_id": product_id,
                },
            )

            if result.rowcount == 0:

                raise ValueError(
                    f"商品不存在: {product_id}"
                )