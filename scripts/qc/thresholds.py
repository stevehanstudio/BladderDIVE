"""Threshold evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Threshold:
    warn: float
    fail: float
    direction: str = "high"  # high | low | abs

    def status(self, value: float) -> str:
        if value is None or (isinstance(value, float) and value != value):
            return "unknown"
        v = abs(value) if self.direction == "abs" else value
        if self.direction == "low":
            if v < self.fail:
                return "fail"
            if v < self.warn:
                return "warn"
            return "pass"
        if v > self.fail:
            return "fail"
        if v > self.warn:
            return "warn"
        return "pass"


def metric(value: float, thr: Threshold | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": float(value)}
    if thr is not None:
        out.update(
            {
                "warn": thr.warn,
                "fail": thr.fail,
                "direction": thr.direction,
                "status": thr.status(value),
            }
        )
    return out


def worst_status(statuses: list[str]) -> str:
    order = {"fail": 3, "warn": 2, "pass": 1, "unknown": 0, "skip": 0}
    if not statuses:
        return "unknown"
    return max(statuses, key=lambda s: order.get(s, 0))


def module_status(metric_dicts: list[dict[str, Any]]) -> str:
    statuses: list[str] = []
    for val in metric_dicts:
        if isinstance(val, dict) and "status" in val:
            statuses.append(val["status"])
        elif isinstance(val, dict):
            statuses.extend(
                s for s in (v.get("status") for v in val.values() if isinstance(v, dict)) if s
            )
    return worst_status(statuses)
