"""Plot the sealed first-pass risk–coverage evaluation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
COLORS = {"gap_only": "#a3482a", "uncertainty_aware": "#14786f"}
METHOD_LABELS = {
    "gap_only": "Predicted gap",
    "uncertainty_aware": "Joint resampling",
}


def main():
    results = json.loads((HERE / "SCORE_SUMMARY.json").read_text())
    fig, (fixed, matched) = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.81, wspace=0.13)
    fig.suptitle("Named slow-set selection on 8,192 independent windows", x=0.09,
                 y=0.962, ha="left", fontsize=15, fontweight="bold", color="#202b34")
    for axis in (fixed, matched):
        axis.set_xlim(0, 0.835)
        axis.set_ylim(0, 0.335)
        axis.set_xlabel("Test coverage")
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        axis.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        axis.grid(axis="y", color="#e5e9e9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#a6afb1")
        axis.axhline(0.05, color="#536674", linestyle=":", linewidth=1.4)

    fixed.set_title("Frozen development thresholds", loc="left", fontsize=11)
    fixed.set_ylabel("Incorrect among answered windows")
    matched.set_title("Matched test coverage · descriptive", loc="left", fontsize=11)
    for method, color in COLORS.items():
        points = results["fixed_rule_results"][method]
        coverages = [point["coverage"] for point in points]
        observed = [point["selective_error"] for point in points]
        upper = [point["upper"] for point in points]
        fixed.plot(coverages, observed, color=color, marker="o", ms=4, lw=1.9)
        fixed.plot(coverages, upper, color=color, linestyle="--", alpha=0.85, lw=1.4)
        comparison = results["matched_realized_coverage_descriptive"]
        matched.plot(
            [point[method]["coverage"] for point in comparison],
            [point[method]["selective_error"] for point in comparison],
            color=color, marker="o", ms=4, lw=1.9,
        )
    matched.annotate("8.0% vs 15.7%\nat 20% coverage",
                     xy=(0.20, 0.08), xytext=(0.26, 0.04),
                     arrowprops={"arrowstyle": "-", "color": "#14786f", "lw": 0.9},
                     fontsize=9, color="#14786f", va="top")
    fig.legend(
        handles=[
            *[
                Line2D([0], [0], color=color, lw=2, label=METHOD_LABELS[method])
                for method, color in COLORS.items()
            ],
            Line2D([0], [0], color="#536674", linestyle="--", label="95% simultaneous upper bound"),
            Line2D([0], [0], color="#536674", linestyle=":", label="5% risk target"),
        ],
        loc="lower center", bbox_to_anchor=(0.51, 0.04), frameon=False, ncols=4, fontsize=9,
    )
    fig.savefig(HERE / "risk_coverage.png", dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
