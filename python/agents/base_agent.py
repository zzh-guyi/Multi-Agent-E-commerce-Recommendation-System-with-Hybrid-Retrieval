from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import AgentResult

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    所有 Agent 的基类。

    负责：
    1. Agent 调用次数统计
    2. Agent 执行耗时统计
    3. Retry
    4. Fallback
    5. 日志输出
    """

    def __init__(
        self,
        name: str,
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries

        self._call_count = 0
        self._error_count = 0

    # ========================================================
    # 子类必须实现
    # ========================================================

    @abstractmethod
    async def _execute(
        self,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Agent真正的业务逻辑。

        例如：

        UserProfileAgent
            ↓
        _execute()

        ProductRecAgent
            ↓
        _execute()
        """

    # ========================================================
    # Agent统一入口
    # ========================================================

    async def run(
        self,
        **kwargs: Any,
    ) -> AgentResult:

        start = time.perf_counter()

        self._call_count += 1

        # ----------------------------------------------------
        # 开始执行
        # ----------------------------------------------------

        print(
            f"[{self.name}] 开始执行"
        )

        try:

            # ------------------------------------------------
            # 执行Agent
            # ------------------------------------------------

            result = await self._retry_execute(
                **kwargs
            )

            # ------------------------------------------------
            # 计算耗时
            # ------------------------------------------------

            result.latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            # ------------------------------------------------
            # 控制台打印
            # ------------------------------------------------

            print(
                f"[{self.name}] 完成，"
                f"耗时 {result.latency_ms:.1f} ms"
            )

            # ------------------------------------------------
            # structlog
            # ------------------------------------------------

            logger.info(
                "agent.success",
                agent=self.name,
                latency_ms=round(
                    result.latency_ms,
                    1,
                ),
            )

            return result

        except Exception as exc:

            self._error_count += 1

            latency_ms = (
                time.perf_counter()
                - start
            ) * 1000

            # ------------------------------------------------
            # 控制台打印失败信息
            # ------------------------------------------------

            print(
                f"[{self.name}] 失败，"
                f"耗时 {latency_ms:.1f} ms，"
                f"error={exc}"
            )

            # ------------------------------------------------
            # structlog
            # ------------------------------------------------

            logger.error(
                "agent.failed",
                agent=self.name,
                error=str(exc),
            )

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            return self._fallback(
                latency_ms,
                exc,
            )

    # ========================================================
    # Retry
    # ========================================================

    async def _retry_execute(
        self,
        **kwargs: Any,
    ) -> AgentResult:

        @retry(
            stop=stop_after_attempt(
                self.max_retries
            ),
            wait=wait_exponential(
                multiplier=0.5,
                min=0.5,
                max=4,
            ),
            reraise=True,
        )
        async def _inner():

            return await self._execute(
                **kwargs
            )

        return await _inner()

    # ========================================================
    # Fallback
    # ========================================================

    def _fallback(
        self,
        latency_ms: float,
        exc: Exception,
    ) -> AgentResult:

        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
        )

    # ========================================================
    # Error Rate
    # ========================================================

    @property
    def error_rate(self) -> float:

        if self._call_count == 0:
            return 0.0

        return (
            self._error_count
            / self._call_count
        )