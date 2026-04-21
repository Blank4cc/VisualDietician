from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ReportVizConfig:
    """Configuration for report-oriented plots.

    Designed for concise report usage (4-page limit): prioritize merged charts
    and avoid redundant figures.
    """

    confidence_threshold: float = 0.35
    top_error_classes: int = 12
    top_class_table_size: int = 20
    dpi: int = 220
    dashboard_size: tuple[float, float] = (14.0, 9.0)
    calibration_size: tuple[float, float] = (10.0, 5.2)
    class_detail_size: tuple[float, float] = (12.0, 7.0)
    style: str = "seaborn-v0_8-whitegrid"


_REQUIRED_COLUMNS = {
    "gt_label",
    "pred_top1",
    "pred_top1_confidence",
    "top1_hit",
    "top3_hit",
    "total_calories",
}


def load_scheme_a_results(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate Scheme A batch result CSV.

    Args:
        csv_path: Path to `scheme_a_test_results.csv`.

    Returns:
        Cleaned DataFrame with normalized numeric columns.

    Raises:
        FileNotFoundError: If CSV does not exist.
        ValueError: If required columns are missing or file is empty.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"结果文件不存在: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError as exc:
        raise ValueError(f"CSV 解析失败: {path}") from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(f"读取结果文件失败: {path}") from exc

    if df.empty:
        raise ValueError(f"结果文件为空: {path}")
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"结果文件缺少必要字段: {sorted(missing)}")

    # Normalize numeric columns for robust metric computation.
    for col in ["pred_top1_confidence", "top1_hit", "top3_hit", "total_calories"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["gt_label"] = df["gt_label"].astype(str)
    df["pred_top1"] = df["pred_top1"].astype(str)
    return df


def _summary_metrics(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    image_count = float(len(df))
    if image_count == 0:
        return {
            "image_count": 0.0,
            "top1_acc": 0.0,
            "top3_acc": 0.0,
            "nonzero_total_calorie_rate": 0.0,
            "avg_total_calories": 0.0,
            "low_confidence_rate": 0.0,
            "trusted_nonzero_total_calorie_rate": 0.0,
        }

    trusted = df[df["pred_top1_confidence"] >= threshold]
    trusted_count = float(len(trusted))
    trusted_nonzero = float((trusted["total_calories"] > 0).mean()) if trusted_count > 0 else 0.0
    return {
        "image_count": image_count,
        "top1_acc": float(df["top1_hit"].mean()),
        "top3_acc": float(df["top3_hit"].mean()),
        "nonzero_total_calorie_rate": float((df["total_calories"] > 0).mean()),
        "avg_total_calories": float(df["total_calories"].mean()),
        "low_confidence_rate": float((df["pred_top1_confidence"] < threshold).mean()),
        "trusted_nonzero_total_calorie_rate": trusted_nonzero,
    }


def _class_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("gt_label")
        .agg(
            samples=("gt_label", "count"),
            top1_acc=("top1_hit", "mean"),
            top3_acc=("top3_hit", "mean"),
            avg_conf=("pred_top1_confidence", "mean"),
            avg_calories=("total_calories", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["top1_acc", "samples"], ascending=[True, False]).reset_index(drop=True)


def plot_report_dashboard(df: pd.DataFrame, cfg: ReportVizConfig | None = None) -> plt.Figure:
    """Create one merged dashboard figure for report main page.

    Subplots:
    - KPI bar chart
    - confidence distribution
    - calorie distribution
    - hardest classes (low top1 accuracy)
    """

    config = cfg or ReportVizConfig()
    metrics = _summary_metrics(df, threshold=config.confidence_threshold)
    class_df = _class_table(df)
    hardest = class_df.head(config.top_error_classes)

    plt.style.use(config.style)
    fig, axes = plt.subplots(2, 2, figsize=config.dashboard_size, dpi=config.dpi)
    ax_kpi, ax_conf, ax_cal, ax_hard = axes.flatten()

    kpi_labels = ["Top1", "Top3", "Calorie>0", "Trusted Calorie>0"]
    kpi_values = [
        metrics["top1_acc"],
        metrics["top3_acc"],
        metrics["nonzero_total_calorie_rate"],
        metrics["trusted_nonzero_total_calorie_rate"],
    ]
    bars = ax_kpi.bar(kpi_labels, kpi_values, color=["#3b82f6", "#6366f1", "#10b981", "#14b8a6"])
    ax_kpi.set_ylim(0.0, 1.0)
    ax_kpi.set_title("Core Metrics")
    for bar, value in zip(bars, kpi_values):
        ax_kpi.text(bar.get_x() + bar.get_width() / 2.0, min(value + 0.03, 0.98), f"{value:.3f}", ha="center")

    ax_conf.hist(df["pred_top1_confidence"], bins=20, color="#8b5cf6", alpha=0.85)
    ax_conf.axvline(config.confidence_threshold, color="#ef4444", linestyle="--", linewidth=1.2)
    ax_conf.set_xlim(0.0, 1.0)
    ax_conf.set_title("Top1 Confidence Distribution")
    ax_conf.set_xlabel("confidence")
    ax_conf.set_ylabel("count")

    ax_cal.hist(df["total_calories"], bins=24, color="#f59e0b", alpha=0.85)
    ax_cal.set_title("Total Calories Distribution")
    ax_cal.set_xlabel("kcal")
    ax_cal.set_ylabel("count")

    if hardest.empty:
        ax_hard.text(0.5, 0.5, "No class data", ha="center", va="center")
        ax_hard.set_axis_off()
    else:
        labels = hardest["gt_label"].tolist()[::-1]
        values = hardest["top1_acc"].tolist()[::-1]
        ax_hard.barh(labels, values, color="#ef4444")
        ax_hard.set_xlim(0.0, 1.0)
        ax_hard.set_title(f"Hardest Classes (Top-{config.top_error_classes})")
        ax_hard.set_xlabel("top1 accuracy")

    fig.suptitle(
        (
            "Scheme A Evaluation Dashboard | "
            f"images={int(metrics['image_count'])}, "
            f"avg_cal={metrics['avg_total_calories']:.1f}, "
            f"low_conf={metrics['low_confidence_rate']:.3f}"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def plot_confidence_calibration(df: pd.DataFrame, cfg: ReportVizConfig | None = None) -> plt.Figure:
    """Create one compact figure for threshold policy trade-off.

    X-axis is confidence threshold; two lines:
    - retained coverage
    - retained top1 accuracy
    """

    config = cfg or ReportVizConfig()
    thresholds = [x / 100.0 for x in range(5, 96, 5)]
    coverage: list[float] = []
    retained_top1: list[float] = []

    for th in thresholds:
        retained = df[df["pred_top1_confidence"] >= th]
        if retained.empty:
            coverage.append(0.0)
            retained_top1.append(0.0)
        else:
            coverage.append(float(len(retained)) / float(len(df)))
            retained_top1.append(float(retained["top1_hit"].mean()))

    plt.style.use(config.style)
    fig, ax = plt.subplots(1, 1, figsize=config.calibration_size, dpi=config.dpi)
    ax.plot(thresholds, retained_top1, marker="o", color="#2563eb", label="retained_top1_acc")
    ax.plot(thresholds, coverage, marker="s", color="#16a34a", label="coverage")
    ax.axvline(config.confidence_threshold, color="#ef4444", linestyle="--", label="current_threshold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("ratio")
    ax.set_title("Confidence Threshold Trade-off")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_class_accuracy_detail(df: pd.DataFrame, cfg: ReportVizConfig | None = None) -> plt.Figure:
    """Create optional detailed class-level accuracy figure."""

    config = cfg or ReportVizConfig()
    class_df = _class_table(df).head(config.top_class_table_size)

    plt.style.use(config.style)
    fig, ax = plt.subplots(1, 1, figsize=config.class_detail_size, dpi=config.dpi)
    if class_df.empty:
        ax.text(0.5, 0.5, "No class data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels = class_df["gt_label"].tolist()[::-1]
    values = class_df["top1_acc"].tolist()[::-1]
    ax.barh(labels, values, color="#f97316")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("top1 accuracy")
    ax.set_title(f"Class Accuracy Detail (Worst {config.top_class_table_size})")
    fig.tight_layout()
    return fig


def export_report_figures(
    scheme_csv_path: str | Path,
    output_dir: str | Path,
    cfg: ReportVizConfig | None = None,
    include_class_detail: bool = False,
) -> dict[str, str]:
    """Export report-ready figures to disk.

    Args:
        scheme_csv_path: Path to Scheme A CSV result file.
        output_dir: Directory for figure outputs.
        cfg: Optional plot configuration.
        include_class_detail: Export third optional figure for appendix.

    Returns:
        Dictionary of generated file paths.
    """

    config = cfg or ReportVizConfig()
    if not (0.0 <= config.confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold 必须在 [0, 1] 区间内")
    if config.top_error_classes <= 0 or config.top_class_table_size <= 0:
        raise ValueError("top_error_classes 与 top_class_table_size 必须大于 0")

    df = load_scheme_a_results(scheme_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}
    dashboard_path = out_dir / "report_dashboard.png"
    calibration_path = out_dir / "report_confidence_tradeoff.png"

    fig_dashboard = plot_report_dashboard(df, config)
    fig_dashboard.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig_dashboard)
    saved["dashboard"] = str(dashboard_path)

    fig_calibration = plot_confidence_calibration(df, config)
    fig_calibration.savefig(calibration_path, bbox_inches="tight")
    plt.close(fig_calibration)
    saved["confidence_tradeoff"] = str(calibration_path)

    if include_class_detail:
        detail_path = out_dir / "report_class_detail.png"
        fig_detail = plot_class_accuracy_detail(df, config)
        fig_detail.savefig(detail_path, bbox_inches="tight")
        plt.close(fig_detail)
        saved["class_detail"] = str(detail_path)

    return saved


def quick_report_plan() -> dict[str, Any]:
    """Return concise chart planning guidance for a 4-page report."""

    return {
        "main_figures": [
            {
                "name": "report_dashboard.png",
                "purpose": "主图，合并展示 KPI/置信度/热量分布/困难类别",
                "placement": "正文主结果页",
            },
            {
                "name": "report_confidence_tradeoff.png",
                "purpose": "展示阈值策略的准确率-覆盖率权衡",
                "placement": "正文方法与可靠性页",
            },
        ],
        "optional_figure": {
            "name": "report_class_detail.png",
            "purpose": "类别级细节，建议放附录或答辩备份",
            "placement": "附录（可选）",
        },
    }

