from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# ============================================================
# Internal Data Classes
# ============================================================


@dataclass
class AgentMetric:
    call_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    recent_errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.call_count if self.call_count > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.call_count if self.call_count > 0 else 0.0


@dataclass
class BusinessMetric:
    exposure_count: int = 0
    click_count: int = 0
    purchase_count: int = 0
    gmv: float = 0.0

    @property
    def ctr(self) -> float:
        return self.click_count / self.exposure_count if self.exposure_count > 0 else 0.0

    @property
    def cvr(self) -> float:
        return self.purchase_count / self.click_count if self.click_count > 0 else 0.0


@dataclass
class ToolMetric:
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.call_count if self.call_count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.call_count if self.call_count > 0 else 0.0


# ============================================================
# Metrics Collector
# ============================================================


class MetricsCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._agent_metrics: dict[str, AgentMetric] = {}
        self._business_metrics = BusinessMetric()
        self._tool_metrics: dict[str, ToolMetric] = {}
        self._request_metrics: list[dict[str, Any]] = []

        self._recall_metrics: dict[str, Any] = {
            "call_count": 0, "success_count": 0, "error_count": 0,
            "total_latency_ms": 0.0, "total_iterations": 0,
            "total_tool_calls": 0, "total_search_products_calls": 0,
            "total_get_product_detail_calls": 0,
        }
        self._supervisor_metrics: dict[str, Any] = {
            "call_count": 0, "total_latency_ms": 0.0,
            "total_routing_decisions": 0, "request_count": 0,
        }
        self._rerank_metrics: dict[str, Any] = {
            "call_count": 0, "success_count": 0, "error_count": 0,
            "timeout_count": 0, "fallback_count": 0,
            "total_latency_ms": 0.0, "total_candidate_count": 0,
            "total_output_count": 0,
        }
        self._marketing_metrics: dict[str, Any] = {
            "call_count": 0, "success_count": 0, "error_count": 0,
            "fallback_count": 0, "total_latency_ms": 0.0,
        }
    # ========================================================
    # Agent Metrics
    # ========================================================

    def record_agent_call(
        self,
        agent_name: str,
        success: bool,
        latency_ms: float,
        error: str = "",
    ):
        with self._lock:
            if agent_name not in self._agent_metrics:
                self._agent_metrics[agent_name] = AgentMetric()
            m = self._agent_metrics[agent_name]
            m.call_count += 1
            m.total_latency_ms += latency_ms
            if success:
                m.success_count += 1
            else:
                m.recent_errors.append(error)

    def get_agent_stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result = {}
            for name, m in self._agent_metrics.items():
                result[name] = {
                    "call_count": m.call_count,
                    "success_rate": round(m.success_rate, 4),
                    "avg_latency_ms": round(m.avg_latency_ms, 1),
                    "recent_errors": m.recent_errors[-5:],
                }
            return result

    # ========================================================
    # Business Metrics
    # ========================================================

    def record_business_event(self, event_type: str, **kwargs):
        with self._lock:
            if event_type == "exposure":
                self._business_metrics.exposure_count += 1
            elif event_type == "click":
                self._business_metrics.click_count += 1
            elif event_type == "purchase":
                self._business_metrics.purchase_count += 1
                self._business_metrics.gmv += kwargs.get("amount", 0.0)

    def record_exposure(self, user_id: str, product_id: str, request_id: str, experiment_group: str = ""):
        self.record_business_event("exposure")

    def record_click(self, user_id: str, product_id: str, request_id: str, experiment_group: str = ""):
        self.record_business_event("click")

    def record_purchase(self, user_id: str, product_id: str, amount: float, quantity: int, request_id: str, experiment_group: str = ""):
        self.record_business_event("purchase", amount=amount)

    def get_business_stats(self) -> dict[str, Any]:
        with self._lock:
            bm = self._business_metrics
            return {
                "exposure_count": bm.exposure_count,
                "click_count": bm.click_count,
                "ctr": round(bm.ctr, 4),
                "purchase_count": bm.purchase_count,
                "cvr": round(bm.cvr, 4),
                "gmv": round(bm.gmv, 2),
            }

    # ========================================================
    # Tool Metrics
    # ========================================================

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
        skipped: bool = False,
    ):
        with self._lock:
            if tool_name not in self._tool_metrics:
                self._tool_metrics[tool_name] = ToolMetric()
            m = self._tool_metrics[tool_name]
            m.call_count += 1
            m.total_latency_ms += latency_ms
            if skipped:
                m.skipped_count += 1
            elif success:
                m.success_count += 1
            else:
                m.error_count += 1

    def get_tool_stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for name, metric in self._tool_metrics.items():
                result[name] = {
                    "call_count": metric.call_count,
                    "success_count": metric.success_count,
                    "error_count": metric.error_count,
                    "skipped_count": metric.skipped_count,
                    "avg_latency_ms": round(metric.avg_latency_ms, 1),
                    "success_rate": round(metric.success_rate, 4),
                }
            return result

    # ========================================================
    # Recall Metrics
    # ========================================================

    def record_recall(
        self,
        latency_ms: float,
        success: bool = True,
        iterations: int = 0,
        tool_calls: int = 0,
        search_products_calls: int = 0,
        get_product_detail_calls: int = 0,
    ):
        with self._lock:
            m = self._recall_metrics
        m = self._recall_metrics
        m["total_latency_ms"] += latency_ms
        if success:
            m["success_count"] += 1
        else:
            m["error_count"] += 1
        m["call_count"] += 1
        m["total_iterations"] += iterations
        m["total_tool_calls"] += tool_calls
        m["total_search_products_calls"] += search_products_calls
        m["total_get_product_detail_calls"] += get_product_detail_calls

    def get_recall_stats(self) -> dict[str, Any]:
        with self._lock:
            m = getattr(self, "_recall_metrics", None)
        if not m:
            return {
                "call_count": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_latency_ms": 0.0,
                "avg_iterations": 0.0,
                "avg_tool_calls": 0.0,
                "avg_search_products_calls": 0.0,
                "avg_get_product_detail_calls": 0.0,
            }
        cnt = m["call_count"]
        return {
            "call_count": cnt,
            "success_count": m["success_count"],
            "error_count": m["error_count"],
            "avg_latency_ms": round(m["total_latency_ms"] / cnt, 1),
            "avg_iterations": round(m["total_iterations"] / cnt, 1),
            "avg_tool_calls": round(m["total_tool_calls"] / cnt, 1),
            "avg_search_products_calls": round(m["total_search_products_calls"] / cnt, 1),
            "avg_get_product_detail_calls": round(m["total_get_product_detail_calls"] / cnt, 1),
        }

    # ========================================================
    # Supervisor Metrics
    # ========================================================

    def record_supervisor(
        self,
        latency_ms: float,
        routing_decisions: list[str] | dict[str, int] | None = None,
    ):
        with self._lock:
            m = self._supervisor_metrics
        m = self._supervisor_metrics
        m["total_latency_ms"] += latency_ms
        if routing_decisions:
            m["call_count"] += 1
            if isinstance(routing_decisions, list):
                m["total_routing_decisions"] += len(routing_decisions)
            else:
                m["total_routing_decisions"] += sum(routing_decisions.values())
            m["request_count"] += 1

    def get_supervisor_stats(self) -> dict[str, Any]:
        with self._lock:
            m = getattr(self, "_supervisor_metrics", None)
        if not m:
            return {
                "call_count": 0,
                "avg_latency_ms": 0.0,
                "avg_routing_decisions": 0.0,
            }
        cnt = m["call_count"]
        req_cnt = max(m.get("request_count", 0), 1)
        return {
            "call_count": cnt,
            "avg_latency_ms": round(m["total_latency_ms"] / cnt, 1) if cnt else 0.0,
            "avg_routing_decisions": round(m["total_routing_decisions"] / req_cnt, 1),
        }

    # ========================================================
    # Rerank Metrics
    # ========================================================

    def record_rerank(
        self,
        latency_ms: float,
        success: bool = True,
        timeout: bool = False,
        fallback: bool = False,
        candidate_count: int = 0,
        output_count: int = 0,
    ):
        with self._lock:
            m = self._rerank_metrics
        m = self._rerank_metrics
        m["call_count"] += 1
        if timeout:
            m["timeout_count"] += 1
        if fallback:
            m["fallback_count"] += 1
        if success:
            m["success_count"] += 1
        else:
            m["error_count"] += 1
        m["total_latency_ms"] += latency_ms
        m["total_candidate_count"] += candidate_count
        m["total_output_count"] += output_count

    def get_rerank_stats(self) -> dict[str, Any]:
        with self._lock:
            m = getattr(self, "_rerank_metrics", None)
        if not m:
            return {
                "call_count": 0,
                "success_count": 0,
                "error_count": 0,
                "timeout_count": 0,
                "fallback_count": 0,
                "avg_latency_ms": 0.0,
                "avg_candidate_count": 0.0,
                "avg_output_count": 0.0,
            }
        cnt = m["call_count"]
        return {
            "call_count": cnt,
            "success_count": m["success_count"],
            "error_count": m["error_count"],
            "timeout_count": m["timeout_count"],
            "fallback_count": m["fallback_count"],
            "avg_latency_ms": round(m["total_latency_ms"] / cnt, 1),
            "avg_candidate_count": round(m["total_candidate_count"] / cnt, 1),
            "avg_output_count": round(m["total_output_count"] / cnt, 1),
        }

    # ========================================================
    # Marketing Metrics
    # ========================================================

    def record_marketing(
        self,
        latency_ms: float,
        success: bool = True,
        fallback: bool = False,
    ):
        with self._lock:
            m = self._marketing_metrics
            m["call_count"] += 1
            if fallback:
                m["fallback_count"] += 1
            if success:
                m["success_count"] += 1
            else:
                m["error_count"] += 1
            m["total_latency_ms"] += latency_ms

    def get_marketing_stats(self) -> dict[str, Any]:
        with self._lock:
            m = getattr(self, "_marketing_metrics", None)
        if not m:
            return {
                "call_count": 0,
                "success_count": 0,
                "error_count": 0,
                "fallback_count": 0,
                "avg_latency_ms": 0.0,
            }
        cnt = m["call_count"]
        return {
            "call_count": cnt,
            "success_count": m["success_count"],
            "error_count": m["error_count"],
            "fallback_count": m["fallback_count"],
            "avg_latency_ms": round(m["total_latency_ms"] / cnt, 1),
        }

    # ========================================================
    # Request Metrics
    # ========================================================

    def record_request(
        self,
        request_id: str,
        total_latency_ms: float,
        experiment_group: str,
        product_count: int,
        supervisor_latency_ms: float = 0.0,
        recall_latency_ms: float = 0.0,
        rerank_latency_ms: float = 0.0,
        inventory_latency_ms: float = 0.0,
        filter_latency_ms: float = 0.0,
        marketing_latency_ms: float = 0.0,
        supervisor_routing_count: int = 0,
    ):
        with self._lock:
            self._request_metrics.append({
                "request_id": request_id,
                "total_latency_ms": round(total_latency_ms, 1),
                "experiment_group": experiment_group,
                "product_count": product_count,
                "supervisor_latency_ms": round(supervisor_latency_ms, 1),
                "recall_latency_ms": round(recall_latency_ms, 1),
                "rerank_latency_ms": round(rerank_latency_ms, 1),
                "inventory_latency_ms": round(inventory_latency_ms, 1),
                "filter_latency_ms": round(filter_latency_ms, 1),
                "marketing_latency_ms": round(marketing_latency_ms, 1),
                "supervisor_routing_count": supervisor_routing_count,
            })

    def get_request_stats(self) -> dict[str, Any]:
        with self._lock:
            requests = getattr(self, "_request_metrics", [])
        if not requests:
            return {"count": 0}
        total = sum(r["total_latency_ms"] for r in requests)
        return {
            "count": len(requests),
            "avg_total_latency_ms": round(total / len(requests), 1),
            "avg_supervisor_latency_ms": round(
                sum(r["supervisor_latency_ms"] for r in requests) / len(requests), 1
            ),
            "avg_recall_latency_ms": round(
                sum(r["recall_latency_ms"] for r in requests) / len(requests), 1
            ),
            "avg_rerank_latency_ms": round(
                sum(r["rerank_latency_ms"] for r in requests) / len(requests), 1
            ),
            "avg_marketing_latency_ms": round(
                sum(r["marketing_latency_ms"] for r in requests) / len(requests), 1
            ),
            "total_products_recommended": sum(r["product_count"] for r in requests),
        }

    def get_recent_requests(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            requests = getattr(self, "_request_metrics", [])
        return list(requests[-limit:])


# ============================================================
# Global Instance
# ============================================================

metrics_collector = MetricsCollector()
