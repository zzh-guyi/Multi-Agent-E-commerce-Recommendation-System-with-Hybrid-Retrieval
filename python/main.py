from __future__ import annotations
"""
Multi-Agent E-Commerce Recommendation System — FastAPI Entry Point

Endpoints:
  POST   /api/v1/recommend
  POST   /api/v1/recommend/graph

  GET    /api/v1/products
  GET    /api/v1/products/{product_id}
  POST   /api/v1/products
  PUT    /api/v1/products/{product_id}
  DELETE /api/v1/products/{product_id}

  GET    /api/v1/experiments
  POST   /api/v1/experiments/{experiment_id}/outcome

  POST   /api/v1/events/exposure
  POST   /api/v1/events/click
  POST   /api/v1/events/purchase

  GET    /api/v1/metrics

  GET    /health
"""


import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_settings

from models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
)

from orchestrator.supervisor import SupervisorOrchestrator
from orchestrator.graph import build_recommendation_graph

from services.ab_test import ABTestEngine
from services.metrics import metrics_collector
from services.product_service import ProductService


logger = structlog.get_logger()

settings = get_settings()

ab_engine = ABTestEngine()

supervisor = SupervisorOrchestrator(
    ab_engine=ab_engine
)

product_service = ProductService()


# ============================================================
# 推荐请求上下文
# ============================================================

"""
保存最近一次推荐请求与 A/B 实验组之间的对应关系。

结构：

    request_id
        ↓
    {
        user_id: U001,
        experiment_id: rec_strategy,
        experiment_group: control
    }

这样后面的：

    exposure
    click
    purchase

只需要携带 request_id，
就可以找到这次推荐属于哪个实验组。

注意：
当前项目是内存版本。

生产环境可以替换成：

    Redis
        或
    MySQL
        或
    Kafka / ClickHouse
"""

recommendation_context: dict[str, dict[str, str]] = {}


# ============================================================
# 商品请求模型
# ============================================================


class ProductCreateRequest(BaseModel):
    """
    新增商品请求。

    通过：

        POST /api/v1/products
    """

    product_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="商品ID，例如 P036",
    )

    name: str = Field(
        ...,
        min_length=1,
        description="商品名称",
    )

    category: str = Field(
        ...,
        min_length=1,
        description="商品类目",
    )

    price: float = Field(
        ...,
        ge=0,
        description="商品价格",
    )

    description: str = ""

    brand: str = ""

    seller_id: str = ""

    stock: int = Field(
        default=0,
        ge=0,
        description="库存",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="商品标签",
    )

    image_url: str = ""


class ProductUpdateRequest(BaseModel):
    """
    修改商品请求。
    """

    name: str | None = None

    category: str | None = None

    price: float | None = Field(
        default=None,
        ge=0,
    )

    description: str | None = None

    brand: str | None = None

    seller_id: str | None = None

    stock: int | None = Field(
        default=None,
        ge=0,
    )

    tags: list[str] | None = None

    image_url: str | None = None


# ============================================================
# 用户行为事件请求模型
# ============================================================


class ExposureEventRequest(BaseModel):
    """
    商品曝光事件。

    推荐：

        /recommend
            ↓
        request_id
            ↓
        exposure

    推荐接口返回的 request_id 可以直接传进来。

    示例：

        {
            "user_id": "U001",
            "product_id": "P036",
            "request_id": "REQ001"
        }

    experiment_id 和 experiment_group
    都可以不传。

    系统会优先根据 request_id 自动找到实验组。
    """

    user_id: str = Field(
        ...,
        min_length=1,
        description="用户ID",
    )

    product_id: str = Field(
        ...,
        min_length=1,
        description="商品ID",
    )

    request_id: str = ""

    experiment_id: str = "rec_strategy"

    experiment_group: str = ""


class ClickEventRequest(BaseModel):
    """
    商品点击事件。
    """

    user_id: str = Field(
        ...,
        min_length=1,
        description="用户ID",
    )

    product_id: str = Field(
        ...,
        min_length=1,
        description="商品ID",
    )

    request_id: str = ""

    experiment_id: str = "rec_strategy"

    experiment_group: str = ""


class PurchaseEventRequest(BaseModel):
    """
    商品购买事件。

    amount：
        本次购买产生的 GMV。

    quantity：
        本次购买数量。
    """

    user_id: str = Field(
        ...,
        min_length=1,
        description="用户ID",
    )

    product_id: str = Field(
        ...,
        min_length=1,
        description="商品ID",
    )

    amount: float = Field(
        ...,
        ge=0,
        description="本次购买总金额",
    )

    quantity: int = Field(
        default=1,
        ge=1,
        description="购买数量",
    )

    request_id: str = ""

    experiment_id: str = "rec_strategy"

    experiment_group: str = ""


# ============================================================
# FastAPI 生命周期
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "app.startup",
        model=settings.llm_model,
    )

    # ====================================================
    # 启动时初始化 Milvus 商品向量库
    # 确保 product_embeddings Collection 存在且所有商品已有 Embedding
    # 后续每次推荐请求无需重复建库
    # ====================================================

    # ====================================================
    # 等待 Milvus collection 完成 recovery
    # Milvus 刚启动时 collection 处于 recovering 状态，
    # 需要重试直到可以正常 query。
    # ====================================================
    milvus_ready = False
    for _retry in range(10):
        try:
            product_service.vector_store.exists()
            milvus_ready = True
            break
        except Exception:
            logger.warning(
                "app.startup.milvus_recovering",
                attempt=_retry + 1,
            )
            time.sleep(2)

    if not milvus_ready:
        logger.warning(
            "app.startup.milvus_timeout",
            reason="collection still recovering after 10 retries",
        )

    try:

        all_products = product_service.get_all_products()

        if all_products:

            product_service.vector_store.ensure_products_indexed(
                all_products,
                product_service.embedding_service,
            )

            logger.info(
                "app.startup.milvus_ready",
                total=len(all_products),
            )

        else:

            logger.info(
                "app.startup.milvus_empty",
                reason="no products in database",
            )

    except Exception as exc:

        logger.warning(
            "app.startup.milvus_init_failed",
            error=str(exc),
        )

    yield

    logger.info(
        "app.shutdown"
    )


# ============================================================
# FastAPI
# ============================================================


app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description=(
        "用户画像Agent + 商品推荐Agent + "
        "营销文案Agent + 库存决策Agent，"
        "支持Supervisor与LangGraph编排，"
        "并提供商品CRUD接口、"
        "用户行为事件统计、"
        "A/B测试以及推荐业务指标监控。"
    ),
    version="1.3.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Health Check
# ============================================================


@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model": settings.llm_model,
    }


# ============================================================
# Recommendation API
# ============================================================


@app.post(
    "/api/v1/recommend",
    response_model=RecommendationResponse,
)
async def recommend(
    request: RecommendationRequest,
):
    """
    使用 Supervisor 编排器进行推荐。

    流程：

        用户请求
            ↓
        Supervisor
            ↓
        ABTestEngine
            ↓
        control / treatment
            ↓
        Agent推荐
            ↓
        返回 request_id + experiment_group
    """

    response = await supervisor.recommend(
        request
    )

    # --------------------------------------------------------
    # Agent Metrics
    # --------------------------------------------------------

    _collect_metrics(response)

    # --------------------------------------------------------
    # 保存推荐请求与 A/B 实验组的关系
    # --------------------------------------------------------

    _save_recommendation_context(
        response=response,
        user_id=request.user_id,
    )

    return response


@app.post("/api/v1/recommend/graph")
async def recommend_via_graph(
    request: RecommendationRequest,
):
    """
    使用 LangGraph 状态图进行推荐（统一入口）。

    与 /api/v1/recommend 使用相同的 Supervisor + LangGraph 链路，
    仅作为独立端点保留用于对比测试。
    """

    response = await supervisor.recommend(
        request
    )

    # --------------------------------------------------------
    # 保存推荐请求与 A/B 实验组关系
    # --------------------------------------------------------

    _save_recommendation_context(
        response=response,
        user_id=request.user_id,
    )

    # --------------------------------------------------------
    # 返回与 /api/v1/recommend 相同格式
    # --------------------------------------------------------

    return {
        "request_id": response.request_id,
        "user_id": response.user_id,
        "products": [
            p.model_dump()
            for p in response.products
        ],
        "marketing_copies": response.marketing_copies,
        "experiment_group": response.experiment_group,
        "total_latency_ms": round(response.total_latency_ms, 1),
        "agent_results": {
            name: {
                "success": r.success,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "confidence": r.confidence,
                "data": r.data,
            }
            for name, r in response.agent_results.items()
        },
    }


# ============================================================
# Product CRUD
# ============================================================


@app.get("/api/v1/products")
async def get_products():
    """
    查询全部商品。
    """

    products = product_service.get_all_products()

    return {
        "total": len(products),
        "products": products,
    }


@app.get("/api/v1/products/{product_id}")
async def get_product(
    product_id: str,
):
    """
    查询单个商品。
    """

    product = product_service.get_product(
        product_id
    )

    if product is None:

        raise HTTPException(
            status_code=404,
            detail=f"商品不存在: {product_id}",
        )

    return product


@app.post("/api/v1/products")
async def create_product(
    request: ProductCreateRequest,
):
    """
    新增商品。
    """

    product = request.model_dump()

    try:

        result = product_service.add_product(
            product
        )

        return {
            "status": "created",
            "message": "商品创建成功",
            "product": result,
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


        logger.exception(
            "product.create_failed",
            product_id=request.product_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"商品创建失败: {str(e)}",
        )


@app.put("/api/v1/products/{product_id}")
async def update_product(
    product_id: str,
    request: ProductUpdateRequest,
):
    """
    修改商品。
    """

    product = request.model_dump(
        exclude_unset=True,
        exclude_none=True,
    )

    if not product:

        raise HTTPException(
            status_code=400,
            detail="没有需要更新的字段",
        )

    product.pop(
        "image_url",
        None,
    )

    try:

        result = product_service.update_product(
            product_id=product_id,
            product=product,
        )

        return {
            "status": "updated",
            "message": "商品更新成功",
            "product": result,
        }

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )


        logger.exception(
            "product.update_failed",
            product_id=product_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"商品更新失败: {str(e)}",
        )


@app.delete("/api/v1/products/{product_id}")
async def delete_product(
    product_id: str,
):
    """
    删除商品。
    """

    try:

        product_service.delete_product(
            product_id
        )

        return {
            "status": "deleted",
            "message": "商品删除成功",
            "product_id": product_id,
        }

    except ValueError as e:

        raise HTTPException(
            status_code=404,
            detail=str(e),
        )


        logger.exception(
            "product.delete_failed",
            product_id=product_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"商品删除失败: {str(e)}",
        )


# ============================================================
# A/B Testing
# ============================================================


@app.get("/api/v1/experiments")
async def get_experiments():
    """
    查看所有 A/B 实验状态。

    包含：

        实验配置
        Thompson Sampling 状态
        曝光
        点击
        CTR
        购买
        CVR
        GMV
    """

    experiments = {}

    for exp_id, exp in ab_engine.experiments.items():

        experiments[exp_id] = {
            "name": exp.name,

            "enabled": exp.enabled,

            "groups": [
                {
                    "name": g.name,
                    "weight": g.weight,
                    "config": g.config,
                    "successes": g.successes,
                    "failures": g.failures,
                }
                for g in exp.groups
            ],

            # 原有统计
            "stats": ab_engine.get_stats(
                exp_id
            ),

            # 新增行为统计
            "behavior_stats": (
                ab_engine.get_behavior_stats(
                    exp_id
                )
            ),
        }

    return experiments


@app.post(
    "/api/v1/experiments/{experiment_id}/outcome"
)
async def record_outcome(
    experiment_id: str,
    group: str,
    success: bool,
):
    """
    手动记录 A/B 测试结果。

    该接口保留，用于兼容原来的测试方式。

    实际业务中：
        click
            ↓
        自动更新 Thompson Sampling
    """

    ab_engine.record_outcome(
        experiment_id,
        group,
        success,
    )

    return {
        "status": "recorded"
    }


# ============================================================
# User Behavior Events
# ============================================================


@app.post("/api/v1/events/exposure")
async def record_exposure(
    request: ExposureEventRequest,
):
    """
    记录商品曝光事件。

    推荐情况下只需要：

        user_id
        product_id
        request_id

    系统会通过 request_id 自动找到：

        experiment_id
        experiment_group

    如果 request_id 找不到，
    则使用请求中显式传入的 experiment_group。

    如果仍然没有 experiment_group，
    则根据 user_id 自动进行 A/B 分组。
    """

    (
        experiment_id,
        experiment_group,
    ) = _resolve_experiment_context(
        request_id=request.request_id,
        user_id=request.user_id,
        experiment_id=request.experiment_id,
        experiment_group=request.experiment_group,
    )

    # --------------------------------------------------------
    # 1. 全局业务指标
    # --------------------------------------------------------

    metrics_collector.record_exposure(
        user_id=request.user_id,
        product_id=request.product_id,
        request_id=request.request_id,
        experiment_group=experiment_group,
    )

    # --------------------------------------------------------
    # 2. A/B 实验指标
    # --------------------------------------------------------

    if experiment_group:

        ab_engine.record_behavior_event(
            experiment_id=experiment_id,
            group_name=experiment_group,
            event_type="exposure",
            user_id=request.user_id,
            product_id=request.product_id,
            request_id=request.request_id,
        )

    return {
        "status": "recorded",
        "event": "exposure",
        "experiment_id": experiment_id,
        "experiment_group": experiment_group,
    }


@app.post("/api/v1/events/click")
async def record_click(
    request: ClickEventRequest,
):
    """
    记录商品点击事件。

    点击会自动作为 Thompson Sampling
    的 success 信号。
    """

    (
        experiment_id,
        experiment_group,
    ) = _resolve_experiment_context(
        request_id=request.request_id,
        user_id=request.user_id,
        experiment_id=request.experiment_id,
        experiment_group=request.experiment_group,
    )

    # --------------------------------------------------------
    # 1. 全局业务指标
    # --------------------------------------------------------

    metrics_collector.record_click(
        user_id=request.user_id,
        product_id=request.product_id,
        request_id=request.request_id,
        experiment_group=experiment_group,
    )

    # --------------------------------------------------------
    # 2. A/B 实验指标
    # --------------------------------------------------------

    if experiment_group:

        ab_engine.record_behavior_event(
            experiment_id=experiment_id,
            group_name=experiment_group,
            event_type="click",
            user_id=request.user_id,
            product_id=request.product_id,
            request_id=request.request_id,
        )

    return {
        "status": "recorded",
        "event": "click",
        "experiment_id": experiment_id,
        "experiment_group": experiment_group,
    }


@app.post("/api/v1/events/purchase")
async def record_purchase(
    request: PurchaseEventRequest,
):
    """
    记录商品购买事件。

    同时记录：

        purchase
        GMV

    A/B 实验也会同步记录。
    """

    (
        experiment_id,
        experiment_group,
    ) = _resolve_experiment_context(
        request_id=request.request_id,
        user_id=request.user_id,
        experiment_id=request.experiment_id,
        experiment_group=request.experiment_group,
    )

    # --------------------------------------------------------
    # 1. 全局业务指标
    # --------------------------------------------------------

    metrics_collector.record_purchase(
        user_id=request.user_id,
        product_id=request.product_id,
        amount=request.amount,
        quantity=request.quantity,
        request_id=request.request_id,
        experiment_group=experiment_group,
    )

    # --------------------------------------------------------
    # 2. A/B 实验指标
    # --------------------------------------------------------

    if experiment_group:

        ab_engine.record_behavior_event(
            experiment_id=experiment_id,
            group_name=experiment_group,
            event_type="purchase",
            user_id=request.user_id,
            product_id=request.product_id,
            amount=request.amount,
            quantity=request.quantity,
            request_id=request.request_id,
        )

    return {
        "status": "recorded",
        "event": "purchase",
        "amount": request.amount,
        "quantity": request.quantity,
        "experiment_id": experiment_id,
        "experiment_group": experiment_group,
    }


# ============================================================
# Metrics
# ============================================================


@app.get("/api/v1/metrics")
async def get_metrics():
    """
    查看系统监控指标。

    包含：

        Agent:
            - 调用次数
            - 成功率
            - 平均延迟

        Business:
            - 曝光量
            - 点击量
            - CTR
            - 购买量
            - CVR
            - GMV
    """

    return {
        "agents": (
            metrics_collector.get_agent_stats()
        ),

        "business": (
            metrics_collector.get_business_stats()
        ),

        "supervisor": (
            metrics_collector.get_supervisor_stats()
        ),

        "recall": (
            metrics_collector.get_recall_stats()
        ),

        "tools": (
            metrics_collector.get_tool_stats()
        ),

        "rerank": (
            metrics_collector.get_rerank_stats()
        ),

        "marketing": (
            metrics_collector.get_marketing_stats()
        ),

        "request": (
            metrics_collector.get_request_stats()
        ),

        "recent_requests": (
            metrics_collector.get_recent_requests(limit=5)
        ),
    }


# ============================================================
# Metrics Helper
# ============================================================


def _collect_metrics(
    response: RecommendationResponse,
):
    """
    收集推荐请求中的 Agent 指标。
    """

    for name, result in response.agent_results.items():

        metrics_collector.record_agent_call(
            agent_name=name,

            success=result.success,

            latency_ms=result.latency_ms,
        )

    # Record request-level metrics
    agent_data = getattr(getattr(response, "agent_results", {}).get("product_recall"), "data", {})
    metrics_collector.record_request(
        request_id=response.request_id,
        total_latency_ms=response.total_latency_ms,
        experiment_group=response.experiment_group,
        product_count=len(response.products),
        recall_latency_ms=response.stage_latencies.get("recall", agent_data.get("tool_latency_ms", 0.0)),
        rerank_latency_ms=response.stage_latencies.get("rerank", agent_data.get("rerank_latency_ms", 0.0)),
        supervisor_latency_ms=response.stage_latencies.get("supervisor", 0.0),
        inventory_latency_ms=response.stage_latencies.get("inventory", 0.0),
        filter_latency_ms=response.stage_latencies.get("filter", 0.0),
        marketing_latency_ms=response.stage_latencies.get("marketing", 0.0),
        supervisor_routing_count=response.supervisor_routing_count,
    )
# ============================================================
# Recommendation Context Helper
# ============================================================


def _save_recommendation_context(
    response: RecommendationResponse,
    user_id: str,
):
    """
    保存：

        request_id
            ↓
        user_id
        experiment_id
        experiment_group

    这样用户后续产生曝光、点击、购买时，
    可以通过 request_id 找回 A/B 实验归属。
    """

    request_id = getattr(
        response,
        "request_id",
        "",
    )

    if not request_id:
        return

    experiment_group = getattr(
        response,
        "experiment_group",
        "control",
    )

    recommendation_context[
        request_id
    ] = {
        "user_id": user_id,
        "experiment_id": "rec_strategy",
        "experiment_group": experiment_group,
    }


def _resolve_experiment_context(
    request_id: str,
    user_id: str,
    experiment_id: str,
    experiment_group: str,
) -> tuple[str, str]:
    """
    解析用户行为对应的 A/B 实验。

    优先级：

        1. request_id 对应的推荐上下文
        2. 请求显式传入 experiment_group
        3. 根据 user_id 自动分组

    最终返回：

        experiment_id
        experiment_group
    """

    # ========================================================
    # 1. 优先从 request_id 找
    # ========================================================

    if request_id:

        context = recommendation_context.get(
            request_id
        )

        if context:

            return (
                context.get(
                    "experiment_id",
                    experiment_id,
                ),
                context.get(
                    "experiment_group",
                    experiment_group,
                ),
            )

    # ========================================================
    # 2. 如果 API 显式传了实验组
    # ========================================================

    if experiment_group:

        return (
            experiment_id,
            experiment_group,
        )

    # ========================================================
    # 3. 最后根据 user_id 自动分组
    # ========================================================

    assignment = ab_engine.assign(
        user_id=user_id,
        experiment_id=experiment_id,
    )

    return (
        assignment.get(
            "experiment_id",
            experiment_id,
        ),
        assignment.get(
            "group",
            "control",
        ),
    )


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )









