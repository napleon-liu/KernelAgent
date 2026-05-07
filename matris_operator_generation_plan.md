# MatRIS 算子生成 Plan

## P0：必须支持 forward / backward / double backward

| 算子 | Forward | Backward | Double backward |
|---|---|---|--- | `edge_vector_distance` | `r_ij = pos_j + offset - pos_i`, `dist = norm(r_ij)` | `dL/dpos`, `dL/doffset` | `d²L/dpos²`, force loss 必需 |
| `unit_vector` | `u_ij = r_ij / dist` | `dL/dr_ij` | 必需 |
| `cutoff_envelope` | smooth cutoff polynomial / cosine cutoff | `dL/ddist` | 必需 |
| `bessel_rbf` | `sin(freq * dist) / dist` | `dL/ddist` | 必需 |
| `gaussian_rbf` | `exp(coeff * dist²)` | `dL/ddist` | 必需 |
| `angle_basis` | dot / norm / `acos` / `sin` / `cos` | `dL/dedge_vec` | 必需 |
| `strain_transform` | `pos/cell` under strain | `dL/dstrain` | stress loss 必需 |
| `segment_sum` / `index_add` | grouped sum aggregation | scatter grad | gather/scatter double backward |
| `segment_max` | grouped max | argmax scatter grad | 通常可选，若 softmax 使用则需要 |
| `segment_softmax` | grouped softmax | softmax backward | 必需 |
| `graph_pool_sum` | atom → graph sum | broadcast grad | gather double backward |

## P1：建议支持 forward / backward / double backward

| 算子 | Forward | Backward | Double backward |
|---|---|---|---|
| `edge_gather` | gather node/edge features by index | scatter grad | gather/scatter double backward |
| `edge_feature_fusion` | concat / multiply basis / attention feature | elementwise grad | 必需 |
| `attention_weighted_message` | `msg = value * attn * basis` | grads wrt value/attn/basis | 必需 |
| `node_update_scatter` | edge msg → node hidden | scatter/index_add backward | 必需 |
| `line_graph_update` | three-body / line-edge message passing | gather/scatter backward | 必需 |
| `gated_mlp` | `x1 * sigmoid(x2)` / gated activation | activation + matmul backward | 若 fused 则必须 |
| `fused_silu` | `x * sigmoid(x)` | SiLU backward | 必须 |
| `fused_sigmoid` | sigmoid | sigmoid backward | 必须 |
| `rmsnorm` / `layernorm` | normalize hidden states | norm backward | 若 fused 则必须 |

## P2：可先用 PyTorch/cuBLAS，后续再融合

| 算子 | Forward | Backward | Double backward |
|---|---|---|---|
| `linear` / GEMM | `x @ W + b` | `dW`, `db`, `dx` | PyTorch 已支持 |
| `mlp_block` | Linear + activation + Linear | standard backward | 若自定义 fusion 则需要 |
| `residual_add` | `x + y` | passthrough grad | trivial |
| `dropout` | mask multiply | mask grad | 训练用，double backward 通常不重要 |
| `energy_readout` | per-atom energy MLP | standard backward | force training 必需 |
| `reference_energy_add` | element reference correction | grad passthrough | trivial |

## Force / Stress 专用导数算子

| 算子 | Forward | Backward | Double backward |
|---|---|---|---|
| `energy_to_force` | `F = -dE/dpos` | `d loss_F / dθ` | 本质需要二阶导 |
| `energy_to_stress` | `S = dE/dstrain / volume` | `d loss_S / dθ` | 本质需要二阶导 |
| `force_loss` | MSE/MAE force loss | backward to force | 触发 `d²E/dpos dθ` |
| `stress_loss` | MSE/MAE stress loss | backward to stress | 触发 `d²E/dstrain dθ` |

## 生成顺序

1. `segment_sum` / `segment_softmax` / `graph_pool_sum`
2. `edge_vector_distance` + `cutoff` + `rbf`
3. `angle_basis`
4. `edge_gather` + `message_scatter`
5. `fused_silu` / `fused_sigmoid` / `gated_mlp`
6. `energy_to_force`
7. `energy_to_stress`
8. double backward 全链路补齐：geometry → basis → attention → scatter → energy readout
