from .config import ProjectPaths

__all__ = ["ProjectPaths", "MealPredictor"]


def __getattr__(name: str):
    if name == "MealPredictor":
        from .pipeline import MealPredictor

        return MealPredictor
    raise AttributeError(f"module 'food_cv' has no attribute '{name}'")
