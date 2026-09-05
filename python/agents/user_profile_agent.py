"""
用户画像 Agent

职责：

1. 获取用户行为数据
   ├── Redis Feature Store
   └── context fallback

2. 程序化生成用户画像
   ├── RFM
   ├── 用户分群
   ├── 偏好类目
   ├── 价格区间
   └── 实时标签

3. 为 ProductRecAgent 提供 UserProfile

重要：

本版本不再使用 LLM 生成用户画像。

旧版本：

    用户行为
       ↓
    LLM
       ↓
    UserProfile

新版本：

    Redis / Context
         ↓
    Rule + RFM
         ↓
    UserProfile

这样可以避免用户画像阶段产生数秒甚至几十秒的 LLM 延迟。
"""

from __future__ import annotations

import math
import time
from collections import Counter
from datetime import datetime
from typing import Any

import structlog

from config import get_settings
from models.schemas import (
    UserProfile,
    UserProfileResult,
    UserSegment,
)

from .base_agent import BaseAgent


logger = structlog.get_logger()


class UserProfileAgent(BaseAgent):
    """
    用户画像 Agent。

    当前采用：

        Redis / Context
              ↓
        行为特征提取
              ↓
        RFM计算
              ↓
        用户分群
              ↓
        偏好类目
              ↓
        价格区间
              ↓
        实时标签
              ↓
        UserProfile

    不依赖 LLM。
    """

    def __init__(self):
        settings = get_settings()

        super().__init__(
            name="user_profile",
            timeout=settings.agent_timeout_user_profile,
        )

        # ----------------------------------------------------
        # Feature Store
        #
        # Phase 2 可以注入 Redis Feature Store。
        #
        # 例如：
        #
        # agent.feature_store = RedisFeatureStore()
        # ----------------------------------------------------

        self.feature_store: Any = None

    # ========================================================
    # Main Execute
    # ========================================================

    async def _execute(
        self,
        **kwargs: Any,
    ) -> UserProfileResult:

        start = time.perf_counter()

        user_id: str = kwargs["user_id"]

        context: dict[str, Any] = kwargs.get(
            "context",
            {},
        )

        logger.info(
            "user_profile.start",
            user_id=user_id,
        )

        # ====================================================
        # 1. 获取用户行为
        # ====================================================

        behavior_data = await self._collect_behavior(
            user_id=user_id,
            context=context,
        )

        logger.info(
            "user_profile.behavior_collected",
            user_id=user_id,
            behavior_keys=list(
                behavior_data.keys()
            ),
        )

        # ====================================================
        # 2. 判断是否为新用户
        # ====================================================

        is_new_user = self._is_new_user(
            behavior_data
        )

        # ====================================================
        # 3. RFM
        # ====================================================

        rfm_score = self._calculate_rfm(
            behavior_data
        )

        # ====================================================
        # 4. 用户分群
        # ====================================================

        segments = self._build_segments(
            behavior_data=behavior_data,
            rfm_score=rfm_score,
            is_new_user=is_new_user,
        )

        # ====================================================
        # 5. 偏好类目
        # ====================================================

        preferred_categories = (
            self._extract_preferred_categories(
                behavior_data
            )
        )

        # ====================================================
        # 6. 价格区间
        # ====================================================

        price_range = self._calculate_price_range(
            behavior_data
        )

        # ====================================================
        # 7. 实时标签
        # ====================================================

        real_time_tags = self._build_real_time_tags(
            behavior_data=behavior_data,
            segments=segments,
        )

        # ====================================================
        # 8. 最近浏览
        # ====================================================

        recent_views = self._extract_recent_items(
            behavior_data.get(
                "recent_views",
                [],
            )
        )

        # ====================================================
        # 9. 最近购买
        # ====================================================

        recent_purchases = self._extract_recent_items(
            behavior_data.get(
                "recent_purchases",
                [],
            )
        )

        # ====================================================
        # 10. 构造 UserProfile
        # ====================================================

        profile = UserProfile(
            user_id=user_id,

            segments=segments,

            preferred_categories=(
                preferred_categories
            ),

            price_range=price_range,

            recent_views=recent_views,

            recent_purchases=recent_purchases,

            rfm_score=rfm_score,

            real_time_tags=real_time_tags,
        )

        # ====================================================
        # 11. 统计耗时
        # ====================================================

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        logger.info(
            "user_profile.complete",
            user_id=user_id,
            latency_ms=round(
                latency_ms,
                1,
            ),
            segments=[
                s.value
                for s in segments
            ],
            preferred_categories=(
                preferred_categories
            ),
        )

        # ====================================================
        # 12. 返回
        # ====================================================

        return UserProfileResult(
            success=True,

            profile=profile,

            data={
                "behavior_source": (
                    behavior_data.get(
                        "_source",
                        "context",
                    )
                ),

                "is_new_user": (
                    is_new_user
                ),

                "behavior_summary": {
                    "view_count_7d": (
                        behavior_data.get(
                            "view_count_7d",
                            0,
                        )
                    ),

                    "click_count_7d": (
                        behavior_data.get(
                            "click_count_7d",
                            0,
                        )
                    ),

                    "purchase_count_30d": (
                        behavior_data.get(
                            "purchase_count_30d",
                            0,
                        )
                    ),

                    "avg_order_amount": (
                        behavior_data.get(
                            "avg_order_amount",
                            0.0,
                        )
                    ),
                },

                "rfm_score": rfm_score,

                "processing": (
                    "rule+rfm"
                ),
            },

            confidence=(
                0.6
                if is_new_user
                else 0.9
            ),
        )

    # ========================================================
    # Behavior Collection
    # ========================================================

    async def _collect_behavior(
        self,
        user_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:

        # ====================================================
        # 优先使用 Feature Store
        # ====================================================

        if self.feature_store:

            try:

                data = (
                    await self.feature_store
                    .get_user_features(
                        user_id
                    )
                )

                if data:

                    if not isinstance(
                        data,
                        dict,
                    ):
                        raise TypeError(
                            "Feature Store返回的数据必须是dict"
                        )

                    data = dict(data)

                    data["_source"] = (
                        "feature_store"
                    )

                    return data

            except Exception as exc:

                logger.warning(
                    "user_profile.feature_store_failed",
                    user_id=user_id,
                    error=str(exc),
                )

        # ====================================================
        # Feature Store没有数据
        #
        # 使用API context。
        #
        # 这对于测试和冷启动非常重要。
        # ====================================================

        return {
            "user_id": user_id,

            "recent_views": context.get(
                "recent_views",
                [],
            ),

            "recent_purchases": context.get(
                "recent_purchases",
                [],
            ),

            "view_count_7d": context.get(
                "view_count_7d",
                0,
            ),

            "click_count_7d": context.get(
                "click_count_7d",
                0,
            ),

            "favorite_count_30d": context.get(
                "favorite_count_30d",
                0,
            ),

            "purchase_count_30d": context.get(
                "purchase_count_30d",
                0,
            ),

            "avg_order_amount": context.get(
                "avg_order_amount",
                0.0,
            ),

            "total_spend_30d": context.get(
                "total_spend_30d",
                0.0,
            ),

            "recent_purchase_days_ago": context.get(
                "recent_purchase_days_ago",
                None,
            ),

            "active_hours": context.get(
                "active_hours",
                [],
            ),

            "view_categories": context.get(
                "view_categories",
                [],
            ),

            "purchase_categories": context.get(
                "purchase_categories",
                [],
            ),

            "view_prices": context.get(
                "view_prices",
                [],
            ),

            "purchase_prices": context.get(
                "purchase_prices",
                [],
            ),

            "_source": "context",
        }

    # ========================================================
    # New User
    # ========================================================

    def _is_new_user(
        self,
        behavior_data: dict[str, Any],
    ) -> bool:

        view_count = self._safe_int(
            behavior_data.get(
                "view_count_7d",
                0,
            )
        )

        click_count = self._safe_int(
            behavior_data.get(
                "click_count_7d",
                0,
            )
        )

        purchase_count = self._safe_int(
            behavior_data.get(
                "purchase_count_30d",
                0,
            )
        )

        recent_views = (
            behavior_data.get(
                "recent_views",
                [],
            )
            or []
        )

        recent_purchases = (
            behavior_data.get(
                "recent_purchases",
                [],
            )
            or []
        )

        return (
            view_count == 0
            and click_count == 0
            and purchase_count == 0
            and not recent_views
            and not recent_purchases
        )

    # ========================================================
    # RFM
    # ========================================================

    def _calculate_rfm(
        self,
        behavior_data: dict[str, Any],
    ) -> dict[str, float]:

        # ----------------------------------------------------
        # Recency
        #
        # 最近购买距离现在越近，R越高。
        # ----------------------------------------------------

        recent_days = behavior_data.get(
            "recent_purchase_days_ago",
            None,
        )

        if recent_days is None:

            recency_score = 0.3

        else:

            recent_days = max(
                0,
                self._safe_int(
                    recent_days
                ),
            )

            # 指数衰减
            #
            # 0天 -> 1.0
            # 7天 -> ~0.50
            # 30天 -> ~0.05
            #
            recency_score = math.exp(
                -recent_days / 10
            )

        # ----------------------------------------------------
        # Frequency
        #
        # 30天购买次数
        # ----------------------------------------------------

        purchase_count = self._safe_int(
            behavior_data.get(
                "purchase_count_30d",
                0,
            )
        )

        # 使用log避免极高频用户把分数拉爆

        frequency_score = min(
            1.0,
            math.log1p(
                purchase_count
            )
            / math.log1p(20),
        )

        # ----------------------------------------------------
        # Monetary
        #
        # 30天消费金额
        # ----------------------------------------------------

        total_spend = (
            behavior_data.get(
                "total_spend_30d",
                None,
            )
        )

        if total_spend is None:

            avg_order_amount = (
                self._safe_float(
                    behavior_data.get(
                        "avg_order_amount",
                        0,
                    )
                )
            )

            total_spend = (
                avg_order_amount
                * purchase_count
            )

        total_spend = self._safe_float(
            total_spend
        )

        monetary_score = min(
            1.0,
            math.log1p(
                max(
                    0.0,
                    total_spend,
                )
            )
            / math.log1p(10000),
        )

        return {
            "recency": round(
                recency_score,
                4,
            ),

            "frequency": round(
                frequency_score,
                4,
            ),

            "monetary": round(
                monetary_score,
                4,
            ),
        }

    # ========================================================
    # User Segments
    # ========================================================

    def _build_segments(
        self,
        behavior_data: dict[str, Any],
        rfm_score: dict[str, float],
        is_new_user: bool,
    ) -> list[UserSegment]:

        # ====================================================
        # 新用户
        # ====================================================

        if is_new_user:

            return [
                UserSegment.NEW_USER
            ]

        segments: list[UserSegment] = []

        recency = rfm_score.get(
            "recency",
            0.0,
        )

        frequency = rfm_score.get(
            "frequency",
            0.0,
        )

        monetary = rfm_score.get(
            "monetary",
            0.0,
        )

        # ====================================================
        # ACTIVE
        # ====================================================

        view_count = self._safe_int(
            behavior_data.get(
                "view_count_7d",
                0,
            )
        )

        click_count = self._safe_int(
            behavior_data.get(
                "click_count_7d",
                0,
            )
        )

        if (
            recency >= 0.45
            or view_count >= 5
            or click_count >= 3
        ):

            segments.append(
                UserSegment.ACTIVE
            )

        # ====================================================
        # HIGH VALUE
        # ====================================================

        if (
            frequency >= 0.55
            and monetary >= 0.55
        ):

            segments.append(
                UserSegment.HIGH_VALUE
            )

        # ====================================================
        # PRICE SENSITIVE
        # ====================================================

        avg_order_amount = (
            self._safe_float(
                behavior_data.get(
                    "avg_order_amount",
                    0,
                )
            )
        )

        if (
            avg_order_amount > 0
            and avg_order_amount <= 500
        ):

            segments.append(
                UserSegment.PRICE_SENSITIVE
            )

        # ====================================================
        # CHURN RISK
        #
        # 最近购买很久
        # 且活跃度较低
        # ====================================================

        recent_days = behavior_data.get(
            "recent_purchase_days_ago",
            None,
        )

        if recent_days is not None:

            recent_days = self._safe_int(
                recent_days
            )

            if (
                recent_days >= 30
                and view_count <= 2
            ):

                segments.append(
                    UserSegment.CHURN_RISK
                )

        # ====================================================
        # 没有命中任何标签
        # ====================================================

        if not segments:

            segments.append(
                UserSegment.ACTIVE
            )

        return segments

    # ========================================================
    # Preferred Categories
    # ========================================================

    def _extract_preferred_categories(
        self,
        behavior_data: dict[str, Any],
    ) -> list[str]:

        categories: list[str] = []

        # ----------------------------------------------------
        # 购买类目权重最高
        # ----------------------------------------------------

        purchase_categories = (
            behavior_data.get(
                "purchase_categories",
                [],
            )
            or []
        )

        # ----------------------------------------------------
        # 浏览类目
        # ----------------------------------------------------

        view_categories = (
            behavior_data.get(
                "view_categories",
                [],
            )
            or []
        )

        # ----------------------------------------------------
        # 某些项目可能直接把类目放在recent_views中
        #
        # 例如：
        #
        # recent_views = ["手机", "耳机", "平板"]
        #
        # 因此如果没有结构化category字段，
        # 也兼容这种情况。
        # ----------------------------------------------------

        if not purchase_categories:
            purchase_categories = []

        if not view_categories:

            recent_views = (
                behavior_data.get(
                    "recent_views",
                    [],
                )
                or []
            )

            view_categories = (
                self._extract_category_like_items(
                    recent_views
                )
            )

        counter = Counter()

        # 购买权重 × 3

        for category in purchase_categories:

            if category:

                counter[str(category)] += 3

        # 浏览权重 × 1

        for category in view_categories:

            if category:

                counter[str(category)] += 1

        preferred = [
            category
            for category, _ in counter.most_common(
                5
            )
        ]

        return preferred

    # ========================================================
    # Price Range
    # ========================================================

    def _calculate_price_range(
        self,
        behavior_data: dict[str, Any],
    ) -> tuple[float, float]:

        prices: list[float] = []

        # ----------------------------------------------------
        # 优先使用购买价格
        # ----------------------------------------------------

        purchase_prices = (
            behavior_data.get(
                "purchase_prices",
                [],
            )
            or []
        )

        for price in purchase_prices:

            try:

                value = float(price)

                if value > 0:

                    prices.append(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

        # ----------------------------------------------------
        # 再使用浏览价格
        # ----------------------------------------------------

        view_prices = (
            behavior_data.get(
                "view_prices",
                [],
            )
            or []
        )

        for price in view_prices:

            try:

                value = float(price)

                if value > 0:

                    prices.append(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

        # ----------------------------------------------------
        # 如果没有价格数据
        # ----------------------------------------------------

        if not prices:

            avg_order_amount = (
                self._safe_float(
                    behavior_data.get(
                        "avg_order_amount",
                        0,
                    )
                )
            )

            if avg_order_amount > 0:

                return (
                    round(
                        max(
                            0,
                            avg_order_amount
                            * 0.5,
                        ),
                        2,
                    ),

                    round(
                        avg_order_amount
                        * 1.5,
                        2,
                    ),
                )

            # 冷启动默认价格范围

            return (
                0.0,
                10000.0,
            )

        # ----------------------------------------------------
        # 根据行为价格计算区间
        # ----------------------------------------------------

        min_price = min(prices)

        max_price = max(prices)

        # 防止范围太窄
        #
        # 例如用户只买过：
        #
        # 399元
        #
        # 不应该变成：
        #
        # 399 ~ 399
        #
        # 而应该给一定上下浮动。
        # ----------------------------------------------------

        lower = max(
            0.0,
            min_price * 0.7,
        )

        upper = max(
            max_price * 1.3,
            lower + 100,
        )

        return (
            round(
                lower,
                2,
            ),

            round(
                upper,
                2,
            ),
        )

    # ========================================================
    # Real Time Tags
    # ========================================================

    def _build_real_time_tags(
        self,
        behavior_data: dict[str, Any],
        segments: list[UserSegment],
    ) -> dict[str, Any]:

        tags: dict[str, Any] = {}

        # ====================================================
        # 活跃时段
        # ====================================================

        active_hours = (
            behavior_data.get(
                "active_hours",
                [],
            )
            or []
        )

        if active_hours:

            valid_hours = []

            for hour in active_hours:

                try:

                    hour = int(hour)

                    if 0 <= hour <= 23:

                        valid_hours.append(
                            hour
                        )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

            if valid_hours:

                min_hour = min(
                    valid_hours
                )

                max_hour = max(
                    valid_hours
                )

                tags["活跃时段"] = (
                    f"{min_hour:02d}-"
                    f"{max_hour:02d}点"
                )

        # ====================================================
        # 偏好风格
        # ====================================================

        if (
            UserSegment.PRICE_SENSITIVE
            in segments
        ):

            tags["偏好风格"] = (
                "高性价比"
            )

        elif (
            UserSegment.HIGH_VALUE
            in segments
        ):

            tags["偏好风格"] = (
                "高端品质"
            )

        else:

            tags["偏好风格"] = (
                "综合型"
            )

        # ====================================================
        # 行为强度
        # ====================================================

        view_count = self._safe_int(
            behavior_data.get(
                "view_count_7d",
                0,
            )
        )

        click_count = self._safe_int(
            behavior_data.get(
                "click_count_7d",
                0,
            )
        )

        if view_count >= 20:

            tags["浏览活跃度"] = (
                "高"
            )

        elif view_count >= 5:

            tags["浏览活跃度"] = (
                "中"
            )

        else:

            tags["浏览活跃度"] = (
                "低"
            )

        if click_count >= 10:

            tags["点击活跃度"] = (
                "高"
            )

        elif click_count >= 3:

            tags["点击活跃度"] = (
                "中"
            )

        else:

            tags["点击活跃度"] = (
                "低"
            )

        # ====================================================
        # 用户类型
        # ====================================================

        segment_names = [
            segment.value
            for segment in segments
        ]

        tags["用户类型"] = (
            segment_names
        )

        return tags

    # ========================================================
    # Recent Items
    # ========================================================

    def _extract_recent_items(
        self,
        items: Any,
    ) -> list[str]:

        if not items:

            return []

        if not isinstance(
            items,
            (list, tuple),
        ):

            return []

        result: list[str] = []

        for item in items:

            if item is None:

                continue

            value = str(item).strip()

            if value:

                result.append(
                    value
                )

        # 去重但保持原顺序

        seen = set()

        unique_items = []

        for item in result:

            if item in seen:

                continue

            seen.add(item)

            unique_items.append(
                item
            )

        return unique_items[:20]

    # ========================================================
    # Category Compatibility
    # ========================================================

    def _extract_category_like_items(
        self,
        items: list[Any],
    ) -> list[str]:

        """
        兼容测试数据：

            recent_views = [
                "手机",
                "耳机",
                "平板"
            ]

        如果项目后续使用真正的商品ID，
        则应该优先在 Feature Store 中直接保存
        view_categories。
        """

        result: list[str] = []

        for item in items:

            if item is None:

                continue

            value = str(item).strip()

            if not value:

                continue

            # 简单保留
            #
            # 后续如果 recent_views 保存的是 P001，
            # 可以通过 ProductStore 查询商品category。
            result.append(value)

        return result

    # ========================================================
    # Safe Number Helpers
    # ========================================================

    @staticmethod
    def _safe_int(
        value: Any,
        default: int = 0,
    ) -> int:

        try:

            return int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):

            return default

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default