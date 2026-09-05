"""
Supervisor 编排器（薄层路由器）

职责：
1. A/B 实验分组
2. 构建 PipelineState（包含路由初始字段）
3. 调用 LangGraph 执行完整推荐流程
4. 将结果转换为 RecommendationResponse
"""
from __future__ import annotations

import time
import uuid

import structlog

from models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    AgentResult,
)

from services.ab_test import ABTestEngine
from .graph import build_recommendation_graph, PipelineState


logger = structlog.get_logger()


class SupervisorOrchestrator:
    def __init__(self, ab_engine=None):
        self.ab_engine = ab_engine or ABTestEngine()
        self._graph = None

    def _get_graph(self):
        if self._graph is None:
            self._graph = build_recommendation_graph()
        return self._graph

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        experiment = self.ab_engine.assign(request.user_id)
        experiment_group = experiment.get("group", "control")

        initial_state: PipelineState = {
            "request_id": request_id,
            "user_id": request.user_id,
            "scene": request.scene,
            "num_items": max(1, int(request.num_items)),
            "context": dict(request.context) if request.context else {},
            "experiment_group": experiment_group,
            "user_profile": None,
            "raw_products": [],
            "ranked_products": [],
            "available_ids": set(),
            "final_products": [],
            "marketing_copies": [],
            "agent_results": {},
            "total_latency_ms": 0.0,
            "_start_time": start,
            # 路由初始字段
            "current_node": "init",
            "next_action": "supervisor",
            "iteration_count": 0,
            "agent_visit_counts": {},
        }

        graph = self._get_graph()
        result_state = await graph.ainvoke(initial_state)

        products = list(result_state.get("final_products", []))
        copies = result_state.get("marketing_copies", [])
        agent_results_raw = result_state.get("agent_results", {})

        agent_results = {}
        for name, res in agent_results_raw.items():
            if res is not None:
                agent_results[name] = AgentResult(
                    agent_name=name,
                    success=getattr(res, "success", True),
                    latency_ms=getattr(res, "latency_ms", 0.0),
                    error=getattr(res, "error", None),
                    data=getattr(res, "data", {}),
                    confidence=getattr(res, "confidence", 1.0),
                )

        total_latency = (time.perf_counter() - start) * 1000

        pr_result = result_state.get("agent_results", {}).get("product_recall")
        stage_latencies = {
            "supervisor": result_state.get("_supervisor_total_latency_ms", 0.0),
            "inventory": getattr(result_state.get("agent_results", {}).get("inventory"), "latency_ms", 0.0),
            "filter": result_state.get("filter_latency_ms", 0.0),
            "marketing": getattr(result_state.get("agent_results", {}).get("marketing_copy"), "latency_ms", 0.0),
        }
        if pr_result:
            _pr_data = getattr(pr_result, "data", {}) or {}
            if isinstance(_pr_data, dict):
                stage_latencies["recall"] = _pr_data.get("tool_latency_ms", getattr(pr_result, "latency_ms", 0.0))
        # Read rerank latency from rerank agent result
        rerank_result = result_state.get("agent_results", {}).get("rerank")
        if rerank_result:
            _rr_data = getattr(rerank_result, "data", {}) or {}
            if isinstance(_rr_data, dict):
                stage_latencies["rerank"] = _rr_data.get("rerank_latency_ms", 0.0)

        logger.info(
            "supervisor.complete",
            request_id=request_id,
            product_count=len(products),
            total_latency_ms=round(total_latency, 1),
            stage_latencies=stage_latencies,
            experiment_group=experiment_group,
        )

        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            products=products,
            marketing_copies=copies,
            experiment_group=experiment_group,
            agent_results=agent_results,
            total_latency_ms=round(total_latency, 1),
            stage_latencies=stage_latencies,
            supervisor_routing_count=len(result_state.get("_routing_decisions", [])),
        )
