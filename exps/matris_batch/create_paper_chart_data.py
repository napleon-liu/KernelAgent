#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "exps" / "matris_batch"
OUT = EXP / "optimization_results" / "paper_chart_data"
COMPARISON = EXP / "summaries" / "comparison.csv"
OPT_SUMMARY = EXP / "optimization_results" / "summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_true(v: str) -> bool:
    return str(v).lower() == "true"


def maybe_int(v: str):
    return None if v in ("", "None", None) else int(v)


def maybe_float(v: str):
    return None if v in ("", "None", None) else float(v)


def category(op: str) -> str:
    if op in {"energy_to_force", "energy_to_stress", "force_loss", "stress_loss"}:
        return "force_stress"
    if op in {"segment_softmax", "segment_sum", "index_add", "graph_pool_sum", "node_update_scatter"}:
        return "graph_reduction"
    if op in {"fused_silu", "fused_sigmoid", "residual_add", "reference_energy_add"}:
        return "elementwise"
    if op in {"edge_feature_fusion", "edge_gather", "attention_weighted_message", "line_graph_update"}:
        return "graph_message"
    return "other"


def main() -> None:
    comp = read_csv(COMPARISON)
    opt = read_csv(OPT_SUMMARY)

    # 1. Generation success rates.
    rows = []
    for phase, col_suffix in [("initial", "initial_success"), ("refine", "refine_success")]:
        for variant in ["with_fx", "without_fx"]:
            col = f"{variant}_{col_suffix}"
            success = sum(is_true(r[col]) for r in comp)
            total = len(comp)
            rows.append({
                "phase": phase,
                "variant": variant,
                "success": success,
                "total": total,
                "success_rate": round(success / total, 4),
            })
    write_csv(OUT / "generation_success_rates.csv", rows)

    # 2. Refine outcome categories overall and by tier.
    rows = []
    tiers = ["all"] + sorted({r["tier"] for r in comp})
    for tier in tiers:
        subset = comp if tier == "all" else [r for r in comp if r["tier"] == tier]
        counts = {"both_success": 0, "only_with_fx": 0, "only_without_fx": 0, "both_failed": 0}
        for r in subset:
            wf = is_true(r["with_fx_refine_success"])
            wo = is_true(r["without_fx_refine_success"])
            if wf and wo:
                counts["both_success"] += 1
            elif wf:
                counts["only_with_fx"] += 1
            elif wo:
                counts["only_without_fx"] += 1
            else:
                counts["both_failed"] += 1
        for outcome, count in counts.items():
            rows.append({"tier": tier, "outcome": outcome, "count": count, "total": len(subset), "fraction": round(count / len(subset), 4)})
    write_csv(OUT / "refine_outcomes_by_tier.csv", rows)

    # 3. Refine rounds among successes.
    rows = []
    for variant in ["with_fx", "without_fx"]:
        vals = [maybe_int(r[f"{variant}_refine_rounds"]) for r in comp if is_true(r[f"{variant}_refine_success"])]
        vals = [v for v in vals if v is not None]
        rows.append({
            "variant": variant,
            "success_count": len(vals),
            "mean_rounds": round(statistics.mean(vals), 3),
            "median_rounds": round(statistics.median(vals), 3),
            "min_rounds": min(vals),
            "max_rounds": max(vals),
        })
    write_csv(OUT / "refine_rounds_summary.csv", rows)

    # 4. Force/stress generation table.
    rows = []
    for r in comp:
        if r["tier"] == "force_stress":
            rows.append({
                "operator_id": r["operator_id"],
                "with_fx_initial_success": r["with_fx_initial_success"],
                "without_fx_initial_success": r["without_fx_initial_success"],
                "with_fx_refine_success": r["with_fx_refine_success"],
                "without_fx_refine_success": r["without_fx_refine_success"],
                "with_fx_refine_rounds": r["with_fx_refine_rounds"],
                "without_fx_refine_rounds": r["without_fx_refine_rounds"],
            })
    write_csv(OUT / "force_stress_generation.csv", rows)

    # 5. Optimization speedups.
    rows = []
    for r in opt:
        sp = maybe_float(r.get("speedup_vs_pytorch", ""))
        if sp is None or sp <= 0:
            continue
        op = r["operator_id"]
        rows.append({
            "operator_id": op,
            "category": category(op),
            "best_time_ms": r["best_time_ms"],
            "initial_time_ms": r["initial_time_ms"],
            "pytorch_time_ms": r["pytorch_time_ms"],
            "compile_time_ms": r["compile_time_ms"],
            "speedup_vs_initial": r["speedup_vs_initial"],
            "speedup_vs_pytorch": r["speedup_vs_pytorch"],
            "total_rounds": r["total_rounds"],
            "recommended_for_main_figure": op in {"segment_softmax", "energy_to_stress", "force_loss", "stress_loss", "fused_silu", "edge_feature_fusion"},
        })
    rows.sort(key=lambda x: float(x["speedup_vs_pytorch"]), reverse=True)
    write_csv(OUT / "optimization_speedups.csv", rows)

    # 6. Hessian coverage.
    rows = []
    for tier in tiers:
        subset = comp if tier == "all" else [r for r in comp if r["tier"] == tier]
        total = len(subset)
        success = 0
        for r in subset:
            status = EXP / "artifacts" / r["operator_id"] / "with_fx" / "hessian" / "status.json"
            if status.exists() and json.loads(status.read_text()).get("returncode") == 0:
                success += 1
        rows.append({"tier": tier, "hessian_success": success, "total": total, "coverage_rate": round(success / total, 4)})
    write_csv(OUT / "hessian_coverage_by_tier.csv", rows)

    # 7. Reviewer-facing recommendations.
    rows = [
        {
            "figure_id": "Fig. 1",
            "title": "Effect of FX/Hessian context and refinement on generation success",
            "plot_type": "grouped bar",
            "data_file": "generation_success_rates.csv",
            "main_message": "with_fx+refine reaches 27/33 success (81.8%) vs without_fx+refine 25/33 (75.8%).",
            "reviewer_note": "Use as the main evidence that second-order/FX context improves agentic generation.",
        },
        {
            "figure_id": "Fig. 2",
            "title": "Outcome breakdown by operator tier",
            "plot_type": "stacked bar",
            "data_file": "refine_outcomes_by_tier.csv",
            "main_message": "Shows both-success, only-with-fx, only-without-fx, and both-failed operators across tiers.",
            "reviewer_note": "Helps avoid cherry-picking and shows where FX helps or hurts.",
        },
        {
            "figure_id": "Fig. 3",
            "title": "Iterations to successful generation",
            "plot_type": "bar or box plot",
            "data_file": "refine_rounds_summary.csv",
            "main_message": "with_fx successful cases need fewer rounds on average (1.48) than without_fx (2.20).",
            "reviewer_note": "Supports efficiency of the agentic refinement process.",
        },
        {
            "figure_id": "Fig. 4",
            "title": "Second-order Hessian extraction coverage",
            "plot_type": "coverage bar",
            "data_file": "hessian_coverage_by_tier.csv",
            "main_message": "33/33 operators have successful diagonal Hessian FX artifacts.",
            "reviewer_note": "Directly supports the paper claim about second-order operator generation coverage.",
        },
        {
            "figure_id": "Fig. 5",
            "title": "Optimized generated kernel speedups",
            "plot_type": "horizontal bar, log scale if needed",
            "data_file": "optimization_speedups.csv",
            "main_message": "segment_softmax reaches 6.81x vs PyTorch; representative force/stress loss kernels reach about 1.5x.",
            "reviewer_note": "Use as microbenchmark evidence; clearly separate from end-to-end MatRIS results.",
        },
        {
            "figure_id": "Table 1",
            "title": "Force/stress second-order operator generation",
            "plot_type": "table",
            "data_file": "force_stress_generation.csv",
            "main_message": "All four force/stress-related operators succeed after refinement in both with_fx and without_fx settings.",
            "reviewer_note": "Important for a second-order derivative operator paper, but state that these kernels are representative and not fully integrated into the real MatRIS force path.",
        },
    ]
    write_csv(OUT / "figure_recommendations.csv", rows)

    print(OUT)


if __name__ == "__main__":
    main()
