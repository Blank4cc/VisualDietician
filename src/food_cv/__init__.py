from .config import ProjectPaths

__all__ = ["ProjectPaths", "MealPredictor", "export_report_figures", "ReportVizConfig"]


def __getattr__(name: str):
    if name == "MealPredictor":
        from .pipeline import MealPredictor

        return MealPredictor
    if name == "export_report_figures":
        from .visualization import export_report_figures

        return export_report_figures
    if name == "ReportVizConfig":
        from .visualization import ReportVizConfig

        return ReportVizConfig
    raise AttributeError(f"module 'food_cv' has no attribute '{name}'")
