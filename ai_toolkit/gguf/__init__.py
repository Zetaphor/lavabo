from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .loader import GGUFModelLoader, InferenceResult  # noqa: F401

__all__ = ["GGUFModelLoader", "InferenceResult"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy import helper
    if name in __all__:
        from .loader import GGUFModelLoader, InferenceResult

        return {"GGUFModelLoader": GGUFModelLoader, "InferenceResult": InferenceResult}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
