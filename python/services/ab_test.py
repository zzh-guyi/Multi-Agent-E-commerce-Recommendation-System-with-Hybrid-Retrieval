"""
A/B测试引擎

功能：
- 流量分桶：用户ID哈希取模分桶
- 实验层：Agent级别 / 模型级别 / Prompt级别实验
- MAB算法：Thompson Sampling动态分配流量
- 行为指标：曝光 / 点击 / 购买
- 业务指标：CTR / CVR / GMV
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ============================================================
# Experiment Group
# ============================================================

@dataclass
class ExperimentGroup:
    """
    A/B实验中的一个实验组。
    """

    name: str

    weight: int = 50

    config: dict[str, Any] = field(
        default_factory=dict
    )

    # --------------------------------------------------------
    # Thompson Sampling
    #
    # Beta(successes, failures)
    # --------------------------------------------------------

    successes: int = 1

    failures: int = 1


# ============================================================
# Experiment
# ============================================================

@dataclass
class Experiment:
    """
    一个完整的A/B实验。
    """

    id: str

    name: str

    groups: list[ExperimentGroup]

    enabled: bool = True

    start_time: float = 0.0

    end_time: float = 0.0


# ============================================================
# A/B Test Engine
# ============================================================

class ABTestEngine:
    """
    Bucket-based A/B test engine with optional Thompson Sampling.

    支持：

        1. 用户稳定分桶
        2. Thompson Sampling
        3. 曝光 / 点击 / 购买事件
        4. CTR / CVR / GMV统计
    """

    def __init__(
        self,
        bucket_count: int = 100,
    ):

        self.bucket_count = bucket_count

        self.experiments: dict[
            str,
            Experiment,
        ] = {}

        # ----------------------------------------------------
        # 原来的通用指标
        # ----------------------------------------------------

        self._metrics: list[
            dict[str, Any]
        ] = []

        # ----------------------------------------------------
        # 行为事件
        #
        # experiment_id
        #     ↓
        # group
        #     ↓
        # exposure / click / purchase
        # ----------------------------------------------------

        self._behavior_events: list[
            dict[str, Any]
        ] = []

        self._init_default_experiments()

    # ========================================================
    # 初始化默认实验
    # ========================================================

    def _init_default_experiments(self):

        self.register_experiment(
            Experiment(
                id="rec_strategy",
                name="推荐策略实验",
                groups=[
                    ExperimentGroup(
                        name="control",
                        weight=50,
                        config={
                            "rerank": "rule_based"
                        },
                    ),
                    ExperimentGroup(
                        name="treatment_llm",
                        weight=50,
                        config={
                            "rerank": "llm"
                        },
                    ),
                ],
            )
        )

        self.register_experiment(
            Experiment(
                id="copy_style",
                name="文案风格实验",
                groups=[
                    ExperimentGroup(
                        name="formal",
                        weight=50,
                        config={
                            "style": "formal"
                        },
                    ),
                    ExperimentGroup(
                        name="casual",
                        weight=50,
                        config={
                            "style": "casual"
                        },
                    ),
                ],
            )
        )

    # ========================================================
    # 注册实验
    # ========================================================

    def register_experiment(
        self,
        exp: Experiment,
    ):
        self.experiments[exp.id] = exp

    # ========================================================
    # 用户分组
    # ========================================================

    def assign(
        self,
        user_id: str,
        experiment_id: str = "rec_strategy",
    ) -> dict[str, Any]:
        """
        使用一致性哈希进行用户分组。

        同一个：

            user_id
            +
            experiment_id

        会稳定进入同一个实验组。
        """

        exp = self.experiments.get(
            experiment_id
        )

        if not exp or not exp.enabled:

            return {
                "experiment_id": experiment_id,
                "group": "control",
                "config": {},
            }

        bucket = self._hash_bucket(
            user_id,
            experiment_id,
        )

        group = self._bucket_to_group(
            bucket,
            exp.groups,
        )

        return {
            "experiment_id": experiment_id,
            "group": group.name,
            "config": group.config,
        }

    # ========================================================
    # Thompson Sampling
    # ========================================================

    def assign_thompson(
        self,
        user_id: str,
        experiment_id: str = "rec_strategy",
    ) -> dict[str, Any]:
        """
        使用 Thompson Sampling 动态选择实验组。

        注意：

        user_id 在这里主要用于记录归属。
        动态选择本身基于各实验组当前的 Beta 后验分布。
        """

        exp = self.experiments.get(
            experiment_id
        )

        if not exp or not exp.enabled:

            return {
                "experiment_id": experiment_id,
                "group": "control",
                "config": {},
            }

        samples = []

        for group in exp.groups:

            sample = np.random.beta(
                group.successes,
                group.failures,
            )

            samples.append(
                (
                    sample,
                    group,
                )
            )

        best = max(
            samples,
            key=lambda x: x[0],
        )[1]

        return {
            "experiment_id": experiment_id,
            "group": best.name,
            "config": best.config,
        }

    # ========================================================
    # Thompson Sampling Outcome
    # ========================================================

    def record_outcome(
        self,
        experiment_id: str,
        group_name: str,
        success: bool,
    ):
        """
        更新 Thompson Sampling 后验分布。
        """

        exp = self.experiments.get(
            experiment_id
        )

        if not exp:
            return

        for group in exp.groups:

            if group.name == group_name:

                if success:
                    group.successes += 1
                else:
                    group.failures += 1

                break

    # ========================================================
    # 通用 Metric
    # ========================================================

    def record_metric(
        self,
        experiment_id: str,
        group_name: str,
        metric_name: str,
        value: float,
        user_id: str = "",
    ):
        """
        记录一个通用实验指标。

        这个接口保留，兼容原来的代码。
        """

        self._metrics.append(
            {
                "experiment_id": experiment_id,
                "group": group_name,
                "metric": metric_name,
                "value": value,
                "user_id": user_id,
                "timestamp": time.time(),
            }
        )

    # ========================================================
    # Behavior Event
    # ========================================================

    def record_behavior_event(
        self,
        experiment_id: str,
        group_name: str,
        event_type: str,
        user_id: str = "",
        product_id: str = "",
        amount: float = 0.0,
        quantity: int = 1,
        request_id: str = "",
    ):
        """
        记录推荐系统用户行为。

        event_type：

            exposure
            click
            purchase

        同时：

            - 保存行为事件
            - 转换成实验Metric
            - click 更新 Thompson Sampling
        """

        valid_events = {
            "exposure",
            "click",
            "purchase",
        }

        if event_type not in valid_events:

            raise ValueError(
                f"不支持的行为事件: {event_type}"
            )

        # ----------------------------------------------------
        # 保存原始行为事件
        # ----------------------------------------------------

        event = {
            "experiment_id": experiment_id,
            "group": group_name,
            "event_type": event_type,
            "user_id": user_id,
            "product_id": product_id,
            "amount": float(amount),
            "quantity": int(quantity),
            "request_id": request_id,
            "timestamp": time.time(),
        }

        self._behavior_events.append(
            event
        )

        # ----------------------------------------------------
        # 同步记录通用 Metric
        # ----------------------------------------------------

        if event_type == "exposure":

            self.record_metric(
                experiment_id=experiment_id,
                group_name=group_name,
                metric_name="exposure",
                value=1.0,
                user_id=user_id,
            )

        elif event_type == "click":

            self.record_metric(
                experiment_id=experiment_id,
                group_name=group_name,
                metric_name="click",
                value=1.0,
                user_id=user_id,
            )

            # ------------------------------------------------
            # 点击作为 Thompson Sampling 的成功反馈
            # ------------------------------------------------

            self.record_outcome(
                experiment_id=experiment_id,
                group_name=group_name,
                success=True,
            )

        elif event_type == "purchase":

            self.record_metric(
                experiment_id=experiment_id,
                group_name=group_name,
                metric_name="purchase",
                value=1.0,
                user_id=user_id,
            )

            self.record_metric(
                experiment_id=experiment_id,
                group_name=group_name,
                metric_name="gmv",
                value=float(amount),
                user_id=user_id,
            )

    # ========================================================
    # Experiment Behavior Stats
    # ========================================================

    def get_behavior_stats(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """
        获取某个实验的行为指标。

        返回：

            control:
                exposure_count
                click_count
                ctr
                purchase_count
                cvr
                gmv

            treatment:
                ...
        """

        exp = self.experiments.get(
            experiment_id
        )

        if not exp:
            return {}

        # ----------------------------------------------------
        # 初始化所有实验组
        # ----------------------------------------------------

        stats: dict[
            str,
            dict[str, Any]
        ] = {}

        for group in exp.groups:

            stats[group.name] = {
                "exposure_count": 0,
                "click_count": 0,
                "ctr": 0.0,
                "purchase_count": 0,
                "cvr": 0.0,
                "gmv": 0.0,
            }

        # ----------------------------------------------------
        # 聚合行为事件
        # ----------------------------------------------------

        for event in self._behavior_events:

            if event["experiment_id"] != experiment_id:
                continue

            group_name = event["group"]

            if group_name not in stats:
                continue

            event_type = event["event_type"]

            if event_type == "exposure":

                stats[group_name][
                    "exposure_count"
                ] += 1

            elif event_type == "click":

                stats[group_name][
                    "click_count"
                ] += 1

            elif event_type == "purchase":

                stats[group_name][
                    "purchase_count"
                ] += 1

                stats[group_name][
                    "gmv"
                ] += float(
                    event.get(
                        "amount",
                        0.0,
                    )
                )

        # ----------------------------------------------------
        # 计算 CTR / CVR
        # ----------------------------------------------------

        for group_name, data in stats.items():

            exposure_count = data[
                "exposure_count"
            ]

            click_count = data[
                "click_count"
            ]

            if exposure_count:

                data["ctr"] = round(
                    click_count
                    / exposure_count,
                    4,
                )

            if click_count:

                data["cvr"] = round(
                    data["purchase_count"]
                    / click_count,
                    4,
                )

            data["gmv"] = round(
                data["gmv"],
                2,
            )

        return stats

    # ========================================================
    # 原来的统计接口
    # ========================================================

    def get_stats(
        self,
        experiment_id: str,
    ) -> dict[str, Any]:
        """
        获取实验的通用Metric统计。

        同时附加：

            CTR
            CVR
            GMV
        """

        exp = self.experiments.get(
            experiment_id
        )

        if not exp:
            return {}

        relevant = [
            metric
            for metric in self._metrics
            if metric["experiment_id"]
            == experiment_id
        ]

        stats: dict[
            str,
            dict[str, list[float]]
        ] = {}

        for metric in relevant:

            group = metric["group"]

            metric_name = metric["metric"]

            if group not in stats:

                stats[group] = {}

            if metric_name not in stats[group]:

                stats[group][metric_name] = []

            stats[group][metric_name].append(
                metric["value"]
            )

        result: dict[str, Any] = {}

        for group, metrics in stats.items():

            result[group] = {}

            for metric_name, values in metrics.items():

                arr = np.array(values)

                result[group][metric_name] = {
                    "count": len(values),
                    "mean": float(
                        arr.mean()
                    ),
                    "std": float(
                        arr.std()
                    ),
                    "min": float(
                        arr.min()
                    ),
                    "max": float(
                        arr.max()
                    ),
                }

        # ----------------------------------------------------
        # 附加行为统计
        # ----------------------------------------------------

        behavior_stats = self.get_behavior_stats(
            experiment_id
        )

        for group_name, behavior in behavior_stats.items():

            if group_name not in result:

                result[group_name] = {}

            result[group_name][
                "behavior"
            ] = behavior

        return result

    # ========================================================
    # Hash Bucket
    # ========================================================

    def _hash_bucket(
        self,
        user_id: str,
        experiment_id: str,
    ) -> int:

        raw = (
            f"{user_id}:{experiment_id}"
        )

        h = hashlib.md5(
            raw.encode()
        ).hexdigest()

        return int(
            h[:8],
            16,
        ) % self.bucket_count

    # ========================================================
    # Bucket → Group
    # ========================================================

    def _bucket_to_group(
        self,
        bucket: int,
        groups: list[ExperimentGroup],
    ) -> ExperimentGroup:

        total_weight = sum(
            group.weight
            for group in groups
        )

        cumulative = 0

        normalized_bucket = (
            bucket
            * total_weight
            / self.bucket_count
        )

        for group in groups:

            cumulative += group.weight

            if normalized_bucket < cumulative:

                return group

        return groups[-1]