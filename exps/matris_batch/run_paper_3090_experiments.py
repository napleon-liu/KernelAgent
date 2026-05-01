#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "exps" / "matris_batch" / "optimization_results"
BENCHMARK = ROOT / "exps" / "matris_batch" / "benchmark_matris_generated_kernel.py"

MODELS = ["matris_10m_oam", "matris_10m_mp"]
INFERENCE_TASKS = ["e", "ef", "efsm"]
TRAIN_TASKS = ["e", "ef", "efs"]


def run_one(model: str, mode: str, task: str, warmup: int, repeat: int) -> Path:
    model_tag = model.replace("matris_10m_", "")
    out = RESULT_DIR / f"paper_3090_{model_tag}_{mode}_{task}_sc1.json"
    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--backend",
        "all",
        "--mode",
        mode,
        "--device",
        "cuda",
        "--model-name",
        model,
        "--batch-size",
        "1",
        "--supercell",
        "1",
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--json",
        str(out),
    ]
    if mode == "inference":
        cmd += ["--infer-task", task]
    else:
        cmd += ["--train-task", task, "--optimizer-step"]

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    return out


def collect(json_paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in json_paths:
        if not path.exists():
            rows.append({"source_json": str(path), "status": "missing"})
            continue
        data = json.loads(path.read_text())
        speedups = data.get("speedups_vs_torch_eager", {})
        for result in data.get("results", []):
            backend = result.get("backend")
            common = {
                "model_name": result.get("model_name"),
                "backend": backend,
                "batch_size": result.get("batch_size"),
                "supercell": result.get("supercell"),
                "infer_task": result.get("infer_task"),
                "train_task": result.get("train_task"),
                "warmup": result.get("warmup"),
                "repeat": result.get("repeat"),
                "optimizer_step": result.get("optimizer_step"),
                "compile_mode": result.get("compile_mode"),
                "compile_fullgraph": result.get("compile_fullgraph"),
                "status": result.get("status", "ok"),
                "returncode": result.get("returncode", ""),
                "source_json": str(path),
            }
            benchmarks = result.get("benchmarks", {})
            if benchmarks:
                for mode, stats in benchmarks.items():
                    rows.append(
                        {
                            **common,
                            "mode": mode,
                            "ms_per_iter": stats.get("ms_per_iter"),
                            "speedup_vs_torch_eager": speedups.get(backend, {}).get(mode, ""),
                        }
                    )
            else:
                rows.append({**common, "mode": result.get("mode"), "ms_per_iter": "", "speedup_vs_torch_eager": ""})
    return rows


def write_summary(rows: list[dict[str, object]]) -> tuple[Path, Path]:
    csv_path = RESULT_DIR / "paper_3090_matris_dual_model_sweep.csv"
    json_path = RESULT_DIR / "paper_3090_matris_dual_model_sweep_summary.json"
    fieldnames = [
        "model_name",
        "backend",
        "mode",
        "batch_size",
        "supercell",
        "infer_task",
        "train_task",
        "warmup",
        "repeat",
        "optimizer_step",
        "compile_mode",
        "compile_fullgraph",
        "ms_per_iter",
        "speedup_vs_torch_eager",
        "status",
        "returncode",
        "source_json",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    return csv_path, json_path


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_paths: list[Path] = []
    for model in MODELS:
        for task in INFERENCE_TASKS:
            json_paths.append(run_one(model, "inference", task, warmup=3, repeat=8))
        for task in TRAIN_TASKS:
            json_paths.append(run_one(model, "train", task, warmup=3, repeat=8))
    rows = collect(json_paths)
    csv_path, json_path = write_summary(rows)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
