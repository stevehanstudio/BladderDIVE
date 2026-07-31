"""QC metric registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

REGISTRY: dict[str, dict[str, Callable[..., dict[str, Any]]]] = {}


def register_qc(module: str, name: str):
    """Decorator to register a QC metric function under a module namespace."""

    def decorator(fn: Callable[..., dict[str, Any]]):
        REGISTRY.setdefault(module, {})[name] = fn
        return fn

    return decorator


def list_modules() -> list[str]:
    return list(REGISTRY.keys())


def list_metrics(module: str) -> list[str]:
    return list(REGISTRY.get(module, {}).keys())
