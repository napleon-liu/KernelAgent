#!/usr/bin/env python3
"""Extract Hessian FX graph for arbitrary operators/models.

Thin wrapper around `extract_hessian_graph` from `hessian_graph_utils.py`.
The caller provides a builder callable that returns `(model, example_inputs)`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from hessian_graph_utils import extract_hessian_graph


@dataclass
class RunMetadata:
    mode: str
    dtype: str
    device: str
    builder: str
    artifacts_dir: str
    fx_graph_code_path: str


def _load_builder(builder: str, builder_file: str | None) -> Callable[[], tuple[nn.Module, list[Any] | tuple[Any, ...]]]:
    if ":" not in builder:
        raise ValueError("--builder must be in format module:function")

    module_name, func_name = builder.split(":", 1)

    if builder_file:
        file_path = Path(builder_file).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"builder file not found: {file_path}")
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load module from file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    if not hasattr(module, func_name):
        raise AttributeError(
            f"builder function '{func_name}' not found in module '{module.__name__}'"
        )

    builder_fn = getattr(module, func_name)
    if not callable(builder_fn):
        raise TypeError(f"builder '{builder}' is not callable")

    return builder_fn


def _normalize_example_inputs(example_inputs: list[Any] | tuple[Any, ...]) -> list[Any]:
    if isinstance(example_inputs, tuple):
        return list(example_inputs)
    if isinstance(example_inputs, list):
        return example_inputs
    raise TypeError("builder must return example_inputs as list or tuple")


def _cast_move_and_require_grad(
    example_inputs: list[Any],
    dtype: torch.dtype,
    device: torch.device,
) -> list[Any]:
    normalized: list[Any] = []
    for inp in example_inputs:
        if isinstance(inp, torch.Tensor):
            if inp.dtype.is_floating_point:
                t = inp.to(device=device, dtype=dtype)
                if not t.requires_grad:
                    t = t.detach().requires_grad_(True)
                normalized.append(t)
            else:
                normalized.append(inp.to(device=device))
        else:
            normalized.append(inp)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Hessian FX graph for arbitrary operator/model"
    )
    parser.add_argument(
        "--builder",
        required=True,
        help=(
            "Builder callable in format module:function. "
            "Callable must return (model, example_inputs)."
        ),
    )
    parser.add_argument(
        "--builder-file",
        default=None,
        help="Optional Python file path containing the builder callable.",
    )
    parser.add_argument(
        "--mode",
        default="diagonal",
        choices=["diagonal", "full", "hvp"],
        help="Hessian mode",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Floating tensor dtype used for tracing inputs",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used during graph extraction",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Artifacts output directory (default: .hessian/<builder_name>_<mode>)",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    builder_fn = _load_builder(args.builder, args.builder_file)
    built = builder_fn()

    if not isinstance(built, tuple) or len(built) != 2:
        raise TypeError("builder must return a 2-tuple: (model, example_inputs)")

    model, example_inputs = built
    if not isinstance(model, nn.Module):
        raise TypeError("builder must return torch.nn.Module as first element")

    example_inputs = _normalize_example_inputs(example_inputs)
    example_inputs = _cast_move_and_require_grad(example_inputs, dtype=dtype, device=device)

    fx_graph = extract_hessian_graph(model, example_inputs, mode=args.mode)

    builder_name = args.builder.replace(":", "_").replace(".", "_")
    artifacts_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(".hessian") / f"{builder_name}_{args.mode}"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    graph_path = artifacts_dir / "fx_graph_code.py"
    graph_path.write_text(fx_graph.code)

    metadata = RunMetadata(
        mode=args.mode,
        dtype=args.dtype,
        device=args.device,
        builder=args.builder,
        artifacts_dir=str(artifacts_dir),
        fx_graph_code_path=str(graph_path),
    )

    metadata_path = artifacts_dir / "metadata.json"
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2))

    print(json.dumps(asdict(metadata), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
