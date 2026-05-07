#!/usr/bin/env python3
"""Run resumable MatRIS batch operator generation experiments.

Artifacts are written under exps/matris_batch/artifacts/<operator>/<variant>/.
The runner intentionally keeps initial generation and refinement as separate
phases so the experiment can compare with_fx vs without_fx and preserve all
intermediate prompts, FX graphs, logs, and summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "exps" / "matris_batch"
CONFIG_DIR = EXP_DIR / "configs"
TEMPLATE_DIR = EXP_DIR / "templates"
ARTIFACT_DIR = EXP_DIR / "artifacts"
SUMMARY_DIR = EXP_DIR / "summaries"
LOCK_FILE = EXP_DIR / ".batch_runner.lock"


@dataclass(frozen=True)
class OperatorSpec:
    operator_id: str
    tier: str
    priority: int
    enabled: bool
    spec_text: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonish(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_config() -> dict[str, Any]:
    return read_jsonish(CONFIG_DIR / "run_config.yaml")


OPERATOR_DETAILS: dict[str, str] = {
    "edge_vector_distance": "Forward: r_ij = pos_j + offset - pos_i; dist = sqrt(sum(r_ij^2)). Backward returns gradients wrt pos_i, pos_j, and offset. Double backward returns second-order geometry VJP terms.",
    "unit_vector": "Forward: u_ij = r_ij / (norm(r_ij) + eps). Backward wrt r_ij. Double backward covers the normalized-vector Jacobian derivative.",
    "cutoff_envelope": "Forward: smooth cosine cutoff envelope c(dist)=0.5*(cos(pi*dist/rc)+1) for dist < rc and 0 otherwise. Backward and double backward wrt dist.",
    "bessel_rbf": "Forward: out[i, j] = sin(freq[j] * dist[i]) / dist[i]. Backward and double backward wrt dist.",
    "gaussian_rbf": "Forward: out[i, j] = exp(-gamma * (dist[i] - center[j])^2). Backward and double backward wrt dist.",
    "angle_basis": "Forward: compute dot/norm based angle features from two edge vectors, including cos(theta), sin(theta), and low-order powers. Backward and double backward wrt both edge vectors.",
    "strain_transform": "Forward: transform positions under a small strain tensor, pos_strained = pos + pos @ strain.T. Backward wrt pos and strain; double backward for stress-related losses.",
    "segment_sum": "Forward: grouped sum aggregation out[g] = sum_{i: segment[i]=g} x[i]. Backward broadcasts grad_out to source rows. Double backward is gather/scatter passthrough.",
    "index_add": "Forward: index_add grouped accumulation out[index[i]] += x[i]. Backward gathers grad_out[index]. Double backward is scatter/gather passthrough.",
    "segment_max": "Forward: grouped max aggregation. Backward scatters gradients to argmax entries. Double backward is piecewise-zero except tie/argmax changes.",
    "segment_softmax": "Forward: softmax normalized independently within each segment. Backward uses softmax Jacobian per group. Double backward covers softmax Hessian-vector terms.",
    "graph_pool_sum": "Forward: atom-to-graph sum pooling using batch graph ids. Backward broadcasts graph gradients to atoms. Double backward is gather passthrough.",
    "edge_gather": "Forward: gather node features for edge endpoints by index. Backward scatters edge gradients back to nodes. Double backward is gather/scatter passthrough.",
    "edge_feature_fusion": "Forward: fuse edge features by concatenating inputs and elementwise products. Backward and double backward wrt all feature tensors.",
    "attention_weighted_message": "Forward: msg = value * attn * basis with broadcasting. Backward wrt value, attention, and basis. Double backward covers elementwise product Hessian terms.",
    "node_update_scatter": "Forward: scatter edge messages into destination node states. Backward gathers node gradients to edges. Double backward is scatter/gather passthrough.",
    "line_graph_update": "Forward: line-graph message update by gathering neighboring edge features and combining them. Backward and double backward through gather/add/product terms.",
    "gated_mlp": "Forward: gated activation y = x1 * sigmoid(x2) on split hidden channels. Backward and double backward wrt x.",
    "fused_silu": "Forward: y = x * sigmoid(x). Backward and double backward wrt x.",
    "fused_sigmoid": "Forward: y = sigmoid(x). Backward and double backward wrt x.",
    "rmsnorm": "Forward: y = x * rsqrt(mean(x^2) + eps) * weight. Backward and double backward wrt x and weight.",
    "layernorm": "Forward: y = (x - mean(x)) / sqrt(var(x)+eps) * weight + bias. Backward and double backward wrt x, weight, and bias.",
    "linear": "Forward: y = x @ W + b. Backward wrt x, W, b. Double backward is mostly linear passthrough / zero second derivative.",
    "gemm": "Forward: C = A @ B. Backward wrt A and B. Double backward is bilinear passthrough.",
    "mlp_block": "Forward: two-layer MLP block with sigmoid activation: sigmoid(x @ W1 + b1) @ W2 + b2. Backward and double backward wrt inputs and parameters.",
    "residual_add": "Forward: y = x + residual. Backward passthrough. Double backward zero/passthrough.",
    "dropout": "Forward: deterministic mask multiply y = x * mask / keep_prob for the experiment. Backward multiplies by mask. Double backward is zero/passthrough.",
    "energy_readout": "Forward: per-atom energy readout from hidden features followed by per-atom scalar energy. Backward and double backward for force-training paths.",
    "reference_energy_add": "Forward: add element reference correction to predicted energy. Backward passthrough; double backward zero/passthrough.",
    "energy_to_force": "Forward intent: force = -dE/dpos. Experiment uses a quadratic energy representative whose force is analytically -2*pos. Backward/double backward model force-loss second-order flow.",
    "energy_to_stress": "Forward intent: stress = dE/dstrain / volume. Experiment uses a quadratic strain-energy representative. Backward/double backward model stress-loss second-order flow.",
    "force_loss": "Forward: mean squared error between predicted and target forces. Backward wrt predicted force. Double backward is the MSE Hessian.",
    "stress_loss": "Forward: mean squared error between predicted and target stress. Backward wrt predicted stress. Double backward is the MSE Hessian.",
}


BUILDER_FORWARD: dict[str, str] = {
    "edge_vector_distance": """    def forward(self, pos_i, pos_j, offset):\n        r = pos_j + offset - pos_i\n        return torch.sqrt((r * r).sum(dim=-1) + 1.0e-12)\n""",
    "unit_vector": """    def forward(self, r):\n        dist = torch.sqrt((r * r).sum(dim=-1, keepdim=True) + 1.0e-12)\n        return r / dist\n""",
    "cutoff_envelope": """    def forward(self, dist):\n        rc = 5.0\n        x = math.pi * dist / rc\n        env = 0.5 * (torch.cos(x) + 1.0)\n        return torch.where(dist < rc, env, torch.zeros_like(env))\n""",
    "bessel_rbf": """    def forward(self, dist):\n        freq = torch.arange(1, 5, device=dist.device, dtype=dist.dtype)\n        z = dist[:, None] * freq[None, :]\n        return torch.sin(z) / dist[:, None]\n""",
    "gaussian_rbf": """    def forward(self, dist):\n        centers = torch.linspace(0.0, 5.0, steps=4, device=dist.device, dtype=dist.dtype)\n        return torch.exp(-2.0 * (dist[:, None] - centers[None, :]) ** 2)\n""",
    "angle_basis": """    def forward(self, a, b):\n        dot = (a * b).sum(dim=-1)\n        na = torch.sqrt((a * a).sum(dim=-1) + 1.0e-12)\n        nb = torch.sqrt((b * b).sum(dim=-1) + 1.0e-12)\n        cos_t = torch.clamp(dot / (na * nb), -0.999, 0.999)\n        sin_t = torch.sqrt(torch.clamp(1.0 - cos_t * cos_t, min=0.0))\n        return torch.stack([cos_t, sin_t, cos_t * cos_t], dim=-1)\n""",
    "strain_transform": """    def forward(self, pos, strain):\n        return pos + pos @ strain.transpose(-1, -2)\n""",
    "segment_sum": """    def forward(self, x):\n        out = torch.zeros(3, x.shape[1], device=x.device, dtype=x.dtype)\n        return out.index_add(0, self.segment, x)\n""",
    "index_add": """    def forward(self, x):\n        out = torch.zeros(3, x.shape[1], device=x.device, dtype=x.dtype)\n        return out.index_add(0, self.segment, x)\n""",
    "segment_max": """    def forward(self, x):\n        idx = self.segment[:, None].expand_as(x)\n        out = torch.full((3, x.shape[1]), -1.0e20, device=x.device, dtype=x.dtype)\n        return out.scatter_reduce(0, idx, x, reduce='amax', include_self=True)\n""",
    "segment_softmax": """    def forward(self, logits):\n        shifted = logits - logits.max()\n        expv = torch.exp(shifted)\n        denom = torch.zeros(3, device=logits.device, dtype=logits.dtype).index_add(0, self.segment, expv)\n        return expv / denom[self.segment]\n""",
    "graph_pool_sum": """    def forward(self, x):\n        out = torch.zeros(2, x.shape[1], device=x.device, dtype=x.dtype)\n        return out.index_add(0, self.graph_id, x)\n""",
    "edge_gather": """    def forward(self, node_feat):\n        return node_feat[self.edge_index]\n""",
    "edge_feature_fusion": """    def forward(self, a, b):\n        return torch.cat([a, b, a * b], dim=-1)\n""",
    "attention_weighted_message": """    def forward(self, value, attn, basis):\n        return value * attn[:, None] * basis\n""",
    "node_update_scatter": """    def forward(self, msg):\n        out = torch.zeros(4, msg.shape[1], device=msg.device, dtype=msg.dtype)\n        return out.index_add(0, self.dst, msg)\n""",
    "line_graph_update": """    def forward(self, edge_feat):\n        left = edge_feat[self.src]\n        right = edge_feat[self.dst_line]\n        return left + right + left * right\n""",
    "gated_mlp": """    def forward(self, x):\n        a, b = x.chunk(2, dim=-1)\n        return a * torch.sigmoid(b)\n""",
    "fused_silu": """    def forward(self, x):\n        return x * torch.sigmoid(x)\n""",
    "fused_sigmoid": """    def forward(self, x):\n        return torch.sigmoid(x)\n""",
    "rmsnorm": """    def forward(self, x, weight):\n        inv = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1.0e-5)\n        return x * inv * weight\n""",
    "layernorm": """    def forward(self, x, weight, bias):\n        mean = x.mean(dim=-1, keepdim=True)\n        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)\n        return (x - mean) * torch.rsqrt(var + 1.0e-5) * weight + bias\n""",
    "linear": """    def forward(self, x, weight, bias):\n        return x @ weight + bias\n""",
    "gemm": """    def forward(self, a, b):\n        return a @ b\n""",
    "mlp_block": """    def forward(self, x, w1, b1, w2, b2):\n        h = torch.sigmoid(x @ w1 + b1)\n        return h @ w2 + b2\n""",
    "residual_add": """    def forward(self, x, y):\n        return x + y\n""",
    "dropout": """    def forward(self, x):\n        return x * self.mask / 0.75\n""",
    "energy_readout": """    def forward(self, x, weight, bias):\n        return (x @ weight + bias).squeeze(-1)\n""",
    "reference_energy_add": """    def forward(self, energy, ref):\n        return energy + ref\n""",
    "energy_to_force": """    def forward(self, pos):\n        return -2.0 * pos\n""",
    "energy_to_stress": """    def forward(self, strain):\n        return 2.0 * strain / 10.0\n""",
    "force_loss": """    def forward(self, pred, target):\n        d = pred - target\n        return (d * d).mean()\n""",
    "stress_loss": """    def forward(self, pred, target):\n        d = pred - target\n        return (d * d).mean()\n""",
}


BUILDER_INPUTS: dict[str, str] = {
    "edge_vector_distance": "[torch.randn(8, 3) + 0.2, torch.randn(8, 3), torch.randn(8, 3) * 0.1]",
    "unit_vector": "[torch.randn(8, 3) + 0.2]",
    "cutoff_envelope": "[torch.linspace(0.2, 4.8, steps=16)]",
    "bessel_rbf": "[torch.linspace(0.2, 5.0, steps=16)]",
    "gaussian_rbf": "[torch.linspace(0.0, 5.0, steps=16)]",
    "angle_basis": "[torch.randn(8, 3) + 0.1, torch.randn(8, 3) - 0.1]",
    "strain_transform": "[torch.randn(8, 3), torch.randn(3, 3) * 0.01]",
    "segment_sum": "[torch.randn(8, 4)]",
    "index_add": "[torch.randn(8, 4)]",
    "segment_max": "[torch.randn(8, 4)]",
    "segment_softmax": "[torch.randn(8)]",
    "graph_pool_sum": "[torch.randn(8, 4)]",
    "edge_gather": "[torch.randn(6, 4)]",
    "edge_feature_fusion": "[torch.randn(8, 4), torch.randn(8, 4)]",
    "attention_weighted_message": "[torch.randn(8, 4), torch.randn(8), torch.randn(8, 4)]",
    "node_update_scatter": "[torch.randn(8, 4)]",
    "line_graph_update": "[torch.randn(6, 4)]",
    "gated_mlp": "[torch.randn(8, 8)]",
    "fused_silu": "[torch.randn(16)]",
    "fused_sigmoid": "[torch.randn(16)]",
    "rmsnorm": "[torch.randn(8, 8), torch.randn(8)]",
    "layernorm": "[torch.randn(8, 8), torch.randn(8), torch.randn(8)]",
    "linear": "[torch.randn(8, 6), torch.randn(6, 4), torch.randn(4)]",
    "gemm": "[torch.randn(8, 6), torch.randn(6, 4)]",
    "mlp_block": "[torch.randn(8, 6), torch.randn(6, 8), torch.randn(8), torch.randn(8, 4), torch.randn(4)]",
    "residual_add": "[torch.randn(8, 4), torch.randn(8, 4)]",
    "dropout": "[torch.randn(8, 4)]",
    "energy_readout": "[torch.randn(8, 6), torch.randn(6, 1), torch.randn(1)]",
    "reference_energy_add": "[torch.randn(8), torch.randn(8)]",
    "energy_to_force": "[torch.randn(8, 3)]",
    "energy_to_stress": "[torch.randn(3, 3) * 0.01]",
    "force_loss": "[torch.randn(8, 3), torch.randn(8, 3)]",
    "stress_loss": "[torch.randn(3, 3), torch.randn(3, 3)]",
}


BUFFER_INIT = """        self.register_buffer('segment', torch.tensor([0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long))\n        self.register_buffer('graph_id', torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long))\n        self.register_buffer('edge_index', torch.tensor([0, 2, 1, 3, 4, 5, 0, 1], dtype=torch.long))\n        self.register_buffer('dst', torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.long))\n        self.register_buffer('src', torch.tensor([0, 1, 2, 3], dtype=torch.long))\n        self.register_buffer('dst_line', torch.tensor([1, 2, 3, 4], dtype=torch.long))\n        self.register_buffer('mask', torch.tensor([[1., 1., 0., 1.], [1., 0., 1., 1.], [1., 1., 1., 0.], [0., 1., 1., 1.], [1., 1., 0., 1.], [1., 0., 1., 1.], [1., 1., 1., 0.], [0., 1., 1., 1.]]))\n"""


def load_operators() -> list[OperatorSpec]:
    raw_ops = read_jsonish(CONFIG_DIR / "operators.yaml")
    ops = []
    for item in raw_ops:
        op_id = item["id"]
        ops.append(
            OperatorSpec(
                operator_id=op_id,
                tier=item["tier"],
                priority=int(item["priority"]),
                enabled=bool(item.get("enabled", True)),
                spec_text=OPERATOR_DETAILS.get(op_id, f"MatRIS operator {op_id}."),
            )
        )
    return sorted([op for op in ops if op.enabled], key=lambda op: op.priority)


def render_prompt(op: OperatorSpec, variant: str, appendix: str) -> str:
    template = (TEMPLATE_DIR / "prompt_base.md.j2").read_text()
    return template.format(
        operator_id=op.operator_id,
        tier=op.tier,
        priority=op.priority,
        variant=variant,
        operator_spec_text=op.spec_text,
        variant_appendix=appendix,
    )


def write_builder(op: OperatorSpec, out_dir: Path) -> Path:
    if op.operator_id not in BUILDER_FORWARD or op.operator_id not in BUILDER_INPUTS:
        raise KeyError(f"missing builder for {op.operator_id}")
    code = f"""from __future__ import annotations

import math
import torch
import torch.nn as nn


class MatRISOperatorCase(nn.Module):
    def __init__(self):
        super().__init__()
{BUFFER_INIT}

{BUILDER_FORWARD[op.operator_id]}


def build_case():
    torch.manual_seed(0)
    model = MatRISOperatorCase()
    example_inputs = {BUILDER_INPUTS[op.operator_id]}
    return model, example_inputs
"""
    path = out_dir / "builder.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code)
    return path


def extract_fx(op: OperatorSpec, op_dir: Path, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    hessian_dir = op_dir / "hessian"
    hessian_dir.mkdir(parents=True, exist_ok=True)
    builder_path = write_builder(op, hessian_dir)
    status_path = hessian_dir / "status.json"
    fx_path = hessian_dir / "fx_graph_code.py"
    metadata_path = hessian_dir / "metadata.json"
    if fx_path.exists() and metadata_path.exists():
        return fx_path.read_text(), json.loads(metadata_path.read_text())

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'skills' / 'hessian-extract-graph'}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        str(ROOT / "skills" / "hessian-extract-graph" / "extract_hessian_graph.py"),
        "--builder",
        "builder:build_case",
        "--builder-file",
        str(builder_path),
        "--mode",
        str(config["hessian"].get("mode", "diagonal")),
        "--device",
        str(config["hessian"].get("device", "cpu")),
        "--dtype",
        str(config["hessian"].get("dtype", "float32")),
        "--out-dir",
        str(hessian_dir),
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
    status = {
        "operator_id": op.operator_id,
        "phase": "hessian",
        "started_at": utc_now(),
        "elapsed_s": time.time() - started,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
    write_json(status_path, status)
    if proc.returncode != 0:
        raise RuntimeError(f"FX extraction failed for {op.operator_id}: {proc.stderr[-1000:]}")
    return fx_path.read_text(), json.loads(metadata_path.read_text())


def variant_appendix(op: OperatorSpec, variant: str, op_dir: Path, config: dict[str, Any]) -> str:
    if variant == "without_fx":
        return "No FX graph is provided in this variant. Infer the implementation from the mathematical operator specification only."
    fx_code, metadata = extract_fx(op, op_dir, config)
    template = (TEMPLATE_DIR / "prompt_fx_appendix.md.j2").read_text()
    return template.format(
        fx_graph_code=fx_code,
        fx_metadata_json=json.dumps(metadata, indent=2, ensure_ascii=False),
    )


def run_generation(
    op: OperatorSpec,
    variant: str,
    phase: str,
    rounds: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    from triton_kernel_agent import TritonKernelAgent

    variant_dir = ARTIFACT_DIR / op.operator_id / variant
    phase_dir = variant_dir / phase
    status_path = phase_dir / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("completed") and not is_retryable_failure(status):
            return status

    variant_dir.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir(parents=True, exist_ok=True)
    appendix = variant_appendix(op, variant, variant_dir, config)
    prompt = render_prompt(op, variant, appendix)
    (variant_dir / "operator_spec.md").write_text(op.spec_text)
    (phase_dir / "prompt.md").write_text(prompt)

    started = time.time()
    status: dict[str, Any] = {
        "operator_id": op.operator_id,
        "variant": variant,
        "phase": phase,
        "rounds_requested": rounds,
        "started_at": utc_now(),
        "completed": False,
    }
    write_json(status_path, status)

    try:
        agent = TritonKernelAgent(
            num_workers=int(config.get("num_workers", 1)),
            max_rounds=rounds,
            log_dir=str(phase_dir / "kernelagent_logs"),
            model_name=config.get("model_name"),
            high_reasoning_effort=bool(config.get("high_reasoning_effort", True)),
            test_timeout_s=int(config.get("test_timeout_s", 60)),
        )
        result = agent.generate_kernel(prompt, test_code=None, generate_default_test=True)
        status.update(
            {
                "completed": True,
                "success": bool(result.get("success")),
                "result": result,
                "elapsed_s": time.time() - started,
                "completed_at": utc_now(),
            }
        )
        if result.get("success") and result.get("kernel_code"):
            (phase_dir / "kernel.py").write_text(result["kernel_code"])
        session_dir = result.get("session_dir")
        if config.get("copy_sessions", True) and session_dir:
            session_path = Path(session_dir)
            if session_path.exists():
                dest = phase_dir / "session"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(session_path, dest)
        try:
            agent.cleanup()
        except Exception:
            pass
    except Exception as exc:
        status.update(
            {
                "completed": True,
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc()[-8000:],
                "elapsed_s": time.time() - started,
                "completed_at": utc_now(),
            }
        )
    write_json(status_path, status)
    return status


def is_retryable_failure(status: dict[str, Any]) -> bool:
    text = "\n".join(
        str(status.get(key, ""))
        for key in ("error", "traceback")
    ).lower()
    return any(
        marker in text
        for marker in (
            "504 gateway timeout",
            "timeout",
            "temporarily unavailable",
            "connection",
            "server error",
            "404 page not found",
            "server returned status 404",
        )
    )


def phase_complete(ops: list[OperatorSpec], variants: list[str], phase: str) -> bool:
    for op in ops:
        for variant in variants:
            path = ARTIFACT_DIR / op.operator_id / variant / phase / "status.json"
            if not path.exists():
                return False
            try:
                status = json.loads(path.read_text())
                if not status.get("completed") or is_retryable_failure(status):
                    return False
            except json.JSONDecodeError:
                return False
    return True


def iter_missing(ops: list[OperatorSpec], variants: list[str], phase: str):
    for op in ops:
        for variant in variants:
            path = ARTIFACT_DIR / op.operator_id / variant / phase / "status.json"
            if not path.exists():
                yield op, variant
                continue
            try:
                status = json.loads(path.read_text())
            except json.JSONDecodeError:
                yield op, variant
                continue
            if not status.get("completed") or is_retryable_failure(status):
                yield op, variant


def status_for(op: OperatorSpec, variant: str, phase: str) -> dict[str, Any] | None:
    path = ARTIFACT_DIR / op.operator_id / variant / phase / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"completed": False, "success": False, "error": "invalid status json"}


def write_summary(ops: list[OperatorSpec], variants: list[str]) -> dict[str, Any]:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for op in ops:
        by_variant = {}
        for variant in variants:
            initial = status_for(op, variant, "initial")
            refine = status_for(op, variant, "refine")
            by_variant[variant] = {
                "initial_completed": bool(initial and initial.get("completed")),
                "initial_success": bool(initial and initial.get("success")),
                "initial_rounds": (initial or {}).get("result", {}).get("rounds"),
                "initial_elapsed_s": (initial or {}).get("elapsed_s"),
                "refine_completed": bool(refine and refine.get("completed")),
                "refine_success": bool(refine and refine.get("success")),
                "refine_rounds": (refine or {}).get("result", {}).get("rounds"),
                "refine_elapsed_s": (refine or {}).get("elapsed_s"),
                "error": (refine or initial or {}).get("error"),
            }
        rows.append(
            {
                "operator_id": op.operator_id,
                "tier": op.tier,
                "priority": op.priority,
                "variants": by_variant,
                "comparison": {
                    "with_fx_refine_success": by_variant.get("with_fx", {}).get("refine_success"),
                    "without_fx_refine_success": by_variant.get("without_fx", {}).get("refine_success"),
                    "with_fx_initial_success": by_variant.get("with_fx", {}).get("initial_success"),
                    "without_fx_initial_success": by_variant.get("without_fx", {}).get("initial_success"),
                },
            }
        )

    complete = all(
        row["variants"][variant]["initial_completed"] and row["variants"][variant]["refine_completed"]
        for row in rows
        for variant in variants
    )
    summary = {
        "updated_at": utc_now(),
        "complete": complete,
        "num_operators": len(ops),
        "variants": variants,
        "rows": rows,
    }
    write_json(SUMMARY_DIR / "summary.json", summary)

    lines = [
        "operator_id,tier,priority,with_fx_initial_success,without_fx_initial_success,with_fx_refine_success,without_fx_refine_success,with_fx_refine_rounds,without_fx_refine_rounds"
    ]
    for row in rows:
        wf = row["variants"].get("with_fx", {})
        nf = row["variants"].get("without_fx", {})
        lines.append(
            ",".join(
                str(x)
                for x in [
                    row["operator_id"],
                    row["tier"],
                    row["priority"],
                    wf.get("initial_success"),
                    nf.get("initial_success"),
                    wf.get("refine_success"),
                    nf.get("refine_success"),
                    wf.get("refine_rounds"),
                    nf.get("refine_rounds"),
                ]
            )
        )
    (SUMMARY_DIR / "comparison.csv").write_text("\n".join(lines) + "\n")
    return summary


@contextmanager
def lock_runner():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            data = json.loads(LOCK_FILE.read_text())
            pid = int(data.get("pid", -1))
            os.kill(pid, 0)
            raise RuntimeError(f"batch runner already active with pid {pid}")
        except ProcessLookupError:
            pass
        except ValueError:
            pass
    LOCK_FILE.write_text(json.dumps({"pid": os.getpid(), "started_at": utc_now()}, indent=2))
    try:
        yield
    finally:
        try:
            if LOCK_FILE.exists():
                data = json.loads(LOCK_FILE.read_text())
                if int(data.get("pid", -1)) == os.getpid():
                    LOCK_FILE.unlink()
        except Exception:
            pass


def run_phase(phase: str, limit: int | None = None) -> dict[str, Any]:
    config = load_config()
    ops = load_operators()
    variants = list(config.get("variants", ["with_fx", "without_fx"]))
    rounds = int(config["initial_rounds"] if phase == "initial" else config["refine_rounds"])
    done = 0
    for op, variant in iter_missing(ops, variants, phase):
        print(f"[{utc_now()}] {phase}: {op.operator_id}/{variant} rounds={rounds}", flush=True)
        run_generation(op, variant, phase, rounds, config)
        done += 1
        write_summary(ops, variants)
        if limit is not None and done >= limit:
            break
    return write_summary(ops, variants)


def print_status() -> dict[str, Any]:
    config = load_config()
    ops = load_operators()
    variants = list(config.get("variants", ["with_fx", "without_fx"]))
    summary = write_summary(ops, variants)
    total = len(ops) * len(variants)
    initial_done = sum(1 for op in ops for v in variants if status_for(op, v, "initial") and status_for(op, v, "initial").get("completed"))
    refine_done = sum(1 for op in ops for v in variants if status_for(op, v, "refine") and status_for(op, v, "refine").get("completed"))
    initial_success = sum(1 for op in ops for v in variants if status_for(op, v, "initial") and status_for(op, v, "initial").get("success"))
    refine_success = sum(1 for op in ops for v in variants if status_for(op, v, "refine") and status_for(op, v, "refine").get("success"))
    print(json.dumps({
        "total_operator_variants": total,
        "initial_done": initial_done,
        "initial_success": initial_success,
        "refine_done": refine_done,
        "refine_success": refine_success,
        "complete": summary["complete"],
        "summary": str(SUMMARY_DIR / "summary.json"),
        "comparison_csv": str(SUMMARY_DIR / "comparison.csv"),
    }, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["initial", "refine", "all", "status"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Run at most N missing operator/variant jobs in the selected phase")
    args = parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase == "status":
        print_status()
        return 0

    with lock_runner():
        config = load_config()
        ops = load_operators()
        variants = list(config.get("variants", ["with_fx", "without_fx"]))
        if args.phase in {"initial", "all"}:
            run_phase("initial", args.limit)
        if args.phase == "all" and args.limit is not None:
            print_status()
            return 0
        if args.phase in {"refine", "all"}:
            if args.phase == "all" and not phase_complete(ops, variants, "initial"):
                print("Initial phase is not complete; skipping refine for now.", flush=True)
            else:
                run_phase("refine", args.limit)
        print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
