#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
MATRIS_ROOT = ROOT / "MatRIS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark generated Triton kernels inside MatRIS train/inference steps.")
    parser.add_argument(
        "--backend",
        choices=["baseline", "torch_eager", "torch_compile", "generated_triton", "both", "all"],
        default="both",
    )
    parser.add_argument("--mode", choices=["inference", "train", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default=None, help="Use pretrained MatRIS.load(model_name); default builds a small random model.")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--supercell", type=int, default=1)
    parser.add_argument("--infer-task", default="ef")
    parser.add_argument("--train-task", default="ef")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--enable-residual-add", action="store_true", help="Also route MatRIS residual adds through generated Triton residual_add.")
    parser.add_argument("--json", default=None, help="Optional path to write benchmark results as JSON.")
    return parser.parse_args()


def _canonical_backend(backend: str) -> str:
    return "torch_eager" if backend == "baseline" else backend


def _backend_suite(backend: str) -> tuple[str, ...]:
    if backend == "both":
        return ("torch_eager", "generated_triton")
    if backend == "all":
        return ("torch_eager", "torch_compile", "generated_triton")
    return (_canonical_backend(backend),)


def run_backend_suite(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    for backend in _backend_suite(args.backend):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            backend,
            "--mode",
            args.mode,
            "--device",
            args.device,
            "--layers",
            str(args.layers),
            "--dim",
            str(args.dim),
            "--batch-size",
            str(args.batch_size),
            "--supercell",
            str(args.supercell),
            "--infer-task",
            args.infer_task,
            "--train-task",
            args.train_task,
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--json",
            tmp_path,
        ]
        if args.model_name:
            cmd += ["--model-name", args.model_name]
        if args.optimizer_step:
            cmd.append("--optimizer-step")
        if args.compile_fullgraph:
            cmd.append("--compile-fullgraph")
        if args.enable_residual_add:
            cmd.append("--enable-residual-add")
        cmd += ["--compile-mode", args.compile_mode]

        env = os.environ.copy()
        env["MATRIS_KERNEL_BACKEND"] = "generated_triton" if backend == "generated_triton" else "jiterator"
        env["MATRIS_ENABLE_GENERATED_RESIDUAL_ADD"] = "1" if backend == "generated_triton" and args.enable_residual_add else "0"
        completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
        if completed.returncode == 0:
            with open(tmp_path) as f:
                results.append(json.load(f))
        else:
            results.append(
                {
                    "backend": backend,
                    "mode": args.mode,
                    "device": args.device,
                    "model_name": args.model_name,
                    "batch_size": args.batch_size,
                    "supercell": args.supercell,
                    "infer_task": args.infer_task,
                    "train_task": args.train_task,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "optimizer_step": args.optimizer_step,
                    "compile_mode": args.compile_mode,
                    "compile_fullgraph": args.compile_fullgraph,
                    "enable_residual_add": args.enable_residual_add,
                    "benchmarks": {},
                    "status": "failed",
                    "returncode": completed.returncode,
                    "command": cmd,
                }
            )
        os.unlink(tmp_path)

    summary: dict[str, Any] = {"results": results, "speedups_vs_torch_eager": {}}
    baseline = next((item for item in results if item["backend"] == "torch_eager"), None)
    if baseline is not None:
        for result in results:
            if result["backend"] == "torch_eager":
                continue
            for mode, base_stats in baseline["benchmarks"].items():
                stats = result["benchmarks"].get(mode)
                if stats and stats["ms_per_iter"] > 0:
                    summary["speedups_vs_torch_eager"].setdefault(result["backend"], {})[mode] = (
                        base_stats["ms_per_iter"] / stats["ms_per_iter"]
                    )
    return summary


def setup_matris_import(backend: str, enable_residual_add: bool) -> None:
    backend = _canonical_backend(backend)
    os.environ["MATRIS_KERNEL_BACKEND"] = "generated_triton" if backend == "generated_triton" else "jiterator"
    os.environ["MATRIS_ENABLE_GENERATED_RESIDUAL_ADD"] = "1" if backend == "generated_triton" and enable_residual_add else "0"
    sys.path.insert(0, str(MATRIS_ROOT))


def build_model(args: argparse.Namespace):
    from matris.model import MatRIS

    if args.model_name:
        return MatRIS.load(args.model_name, device=args.device)

    model = MatRIS(
        num_layers=args.layers,
        node_feat_dim=args.dim,
        edge_feat_dim=args.dim,
        three_body_feat_dim=args.dim,
        mlp_hidden_dims=(args.dim, args.dim),
        num_radial=5,
        num_angular=5,
        max_l=2,
        max_n=2,
        graph_conv_mlp="GateMLP",
        norm_type="rms",
    )
    return model.to(args.device)


def build_graphs(model, args: argparse.Namespace):
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        Lattice.cubic(3.61),
        ["Cu", "Cu", "Cu", "Cu"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
    )
    if args.supercell != 1:
        structure.make_supercell([args.supercell, args.supercell, args.supercell])
    graph = model.graph_converter(structure).to(args.device)
    return [graph for _ in range(args.batch_size)]


def sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def prediction_loss(prediction: dict[str, Any]) -> torch.Tensor:
    loss = prediction["e"].float().sum()
    for force in prediction.get("f", []):
        loss = loss + force.float().square().mean() * 1.0e-3
    for stress in prediction.get("s", []):
        loss = loss + stress.float().square().mean() * 1.0e-6
    return loss


def time_loop(step_fn, warmup: int, repeat: int, device: str) -> float:
    for _ in range(warmup):
        step_fn()
    sync_if_needed(device)
    start = time.perf_counter()
    for _ in range(repeat):
        step_fn()
    sync_if_needed(device)
    return (time.perf_counter() - start) * 1000.0 / repeat


def prepare_model_for_backend(model, args: argparse.Namespace):
    if _canonical_backend(args.backend) != "torch_compile":
        return model
    return torch.compile(model, mode=args.compile_mode, fullgraph=args.compile_fullgraph)


def benchmark_inference(model, graphs, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    grad_needed = "f" in args.infer_task or "s" in args.infer_task

    def step() -> None:
        ctx = torch.enable_grad() if grad_needed else torch.no_grad()
        with ctx:
            prediction = model(graphs, task=args.infer_task, is_training=False)
            prediction_loss(prediction).detach()

    return {"ms_per_iter": time_loop(step, args.warmup, args.repeat, args.device)}


def benchmark_train(model, graphs, args: argparse.Namespace) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5) if args.optimizer_step else None

    def step() -> None:
        model.zero_grad(set_to_none=True)
        prediction = model(graphs, task=args.train_task, is_training=True)
        loss = prediction_loss(prediction)
        loss.backward()
        if optimizer is not None:
            optimizer.step()

    return {"ms_per_iter": time_loop(step, args.warmup, args.repeat, args.device)}


def run_single_backend(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generated Triton kernel benchmarking")

    args.backend = _canonical_backend(args.backend)
    torch.manual_seed(0)
    setup_matris_import(args.backend, args.enable_residual_add)
    model = build_model(args)
    graphs = build_graphs(model, args)
    model = prepare_model_for_backend(model, args)

    benchmarks: dict[str, Any] = {}
    if args.mode in {"inference", "both"}:
        benchmarks["inference"] = benchmark_inference(model, graphs, args)
    if args.mode in {"train", "both"}:
        benchmarks["train"] = benchmark_train(model, graphs, args)

    return {
        "backend": args.backend,
        "mode": args.mode,
        "device": args.device,
        "model_name": args.model_name,
        "layers": args.layers,
        "dim": args.dim,
        "batch_size": args.batch_size,
        "supercell": args.supercell,
        "infer_task": args.infer_task,
        "train_task": args.train_task,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "optimizer_step": args.optimizer_step,
        "compile_mode": args.compile_mode,
        "compile_fullgraph": args.compile_fullgraph,
        "enable_residual_add": args.enable_residual_add,
        "benchmarks": benchmarks,
    }


def emit(result: dict[str, Any], json_path: str | None) -> None:
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if json_path:
        Path(json_path).write_text(text + "\n")


def main() -> None:
    args = parse_args()
    if args.backend in {"both", "all"}:
        result = run_backend_suite(args)
    else:
        result = run_single_backend(args)
    emit(result, args.json)


if __name__ == "__main__":
    main()
