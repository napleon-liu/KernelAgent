---
name: hessian-extract-graph
description: Extract Hessian FX graph for arbitrary operators by wrapping model with HessianOperator.
argument-hint: --builder <module:function> [--builder-file path.py] [--mode diagonal|full|hvp] [--device cpu|cuda] [--dtype float32|float16|bfloat16] [--out-dir PATH]
---

This skill follows `hessian_graph_utils.extract_hessian_graph(...)` directly.

## Builder contract
`--builder` must resolve to a callable that returns:

`(model, example_inputs)`

- `model`: `torch.nn.Module`
- `example_inputs`: `list` or `tuple`

Floating tensor inputs are moved/cast per `--device`/`--dtype` and set to `requires_grad=True` if needed.

## Run
`python skills/hessian-extract-graph/extract_hessian_graph.py $ARGUMENTS`

Example (module import):
`python skills/hessian-extract-graph/extract_hessian_graph.py --builder my_pkg.builders:build_case --mode diagonal --device cpu`

Example (from file):
`python skills/hessian-extract-graph/extract_hessian_graph.py --builder custom_builders:build_case --builder-file ./custom_builders.py --mode diagonal`

## Artifacts
- `fx_graph_code.py`
- `metadata.json`
