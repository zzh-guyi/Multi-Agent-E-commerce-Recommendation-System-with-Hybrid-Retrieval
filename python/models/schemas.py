from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserSegment(str, Enum):
    NEW_USER = "new_user"
    ACTIVE = "active"
    HIGH_VALUE = "high_value"
    PRICE_SENSITIVE = "price_sensitive"
    CHURN_RISK = "churn_risk"


class UserProfile(BaseModel):
    user_id: str
    age: int | None = None
    gender: str | None = None
    city: str | None = None
    segments: list[UserSegment] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 10000.0)
    recent_views: list[str] = Field(default_factory=list)
    recent_purchases: list[str] = Field(default_factory=list)
    rfm_score: dict[str, float] = Field(default_factory=dict)
    real_time_tags: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    """
    系统内部使用的完整商品模型。

    MySQL中的商品信息经过转换后，
    最终会使用这个模型进入推荐流程。
    """

    product_id: str
    name: str
    category: str
    price: float
    description: str = ""
    brand: str = ""
    seller_id: str = ""
    stock: int = 0
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    image_url: str = ""


class ProductCreateRequest(BaseModel):
    """
    新增商品 API 请求模型。

    用于：
        POST /api/v1/products
    """

    product_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="商品ID",
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="商品名称",
    )

    category: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="商品类别",
    )

    price: float = Field(
        ...,
        ge=0,
        description="商品价格",
    )

    description: str = Field(
        default="",
        description="商品描述",
    )

    brand: str = Field(
        default="",
        max_length=100,
        description="品牌",
    )

    seller_id: str = Field(
        default="",
        max_length=64,
        description="商家ID",
    )

    stock: int = Field(
        default=0,
        ge=0,
        description="库存数量",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="商品标签",
    )

    image_url: str = Field(
        default="",
        description="商品图片URL",
    )


class ProductUpdateRequest(BaseModel):
    """
    修改商品 API 请求模型。

    所有字段都是可选的。

    例如只修改库存：

        {
            "stock": 100
        }

    就不会触发 Embedding 更新。

    如果修改：
        name
        category
        description
        brand
        tags

    则 ProductService 后续会重新生成 Embedding。
    """

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="商品名称",
    )

    category: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="商品类别",
    )

    price: float | None = Field(
        default=None,
        ge=0,
        description="商品价格",
    )

    description: str | None = Field(
        default=None,
        description="商品描述",
    )

    brand: str | None = Field(
        default=None,
        max_length=100,
        description="品牌",
    )

    seller_id: str | None = Field(
        default=None,
        max_length=64,
        description="商家ID",
    )

    stock: int | None = Field(
        default=None,
        ge=0,
        description="库存数量",
    )

    tags: list[str] | None = Field(
        default=None,
        description="商品标签",
    )

    image_url: str | None = Field(
        default=None,
        description="商品图片URL",
    )


class ProductResponse(Product):
    """
    商品 API 返回模型。

    当前直接继承 Product，
    后续如果需要增加创建时间、更新时间等字段，
    可以继续扩展。
    """

    pass


class RecommendationRequest(BaseModel):
    user_id: str
    scene: str = "homepage"
    num_items: int = 10
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class UserProfileResult(AgentResult):
    agent_name: str = "user_profile"
    profile: UserProfile | None = None


class ProductRecResult(AgentResult):
    agent_name: str = "product_rec"
    products: list[Product] = Field(default_factory=list)
    recall_strategy: str = ""


class MarketingCopyResult(AgentResult):
    agent_name: str = "marketing_copy"
    copies: list[dict[str, str]] = Field(default_factory=list)
    prompt_template_used: str = ""


class InventoryResult(AgentResult):
    agent_name: str = "inventory"
    available_products: list[str] = Field(default_factory=list)
    low_stock_alerts: list[dict[str, Any]] = Field(default_factory=list)
    purchase_limits: dict[str, int] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    products: list[Product] = Field(default_factory=list)
    marketing_copies: list[dict[str, str]] = Field(default_factory=list)
    experiment_group: str = "control"
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    stage_latencies: dict[str, float] = Field(default_factory=dict)
    supervisor_routing_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
