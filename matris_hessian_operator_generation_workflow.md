# MatRIS 二阶导算子生成流程

## 目标

从 MatRIS 算子生成计划中选择一个需要 forward / backward / double backward 的算子，利用 `skills/hessian-extract-graph` 提取二阶导 FX graph，并将该 graph 插入 KernelAgent prompt，用于生成 Triton 算子。

本次验证案例选择：

```text
bessel_rbf: out = sin(freq * dist) / dist
```

原因：

- 属于 MatRIS P0 算子；
- 包含 `sin / div` 等 MLIP basis 常见计算；
- forward、backward、double backward 公式清晰；
- 适合作为二阶导算子生成 smoke test。

---

## 流程总览

```text
1. 从算子 plan 中选择目标算子
2. 编写 PyTorch builder
3. 使用 @skills/hessian-extract-graph 提取 Hessian FX graph
4. 将 FX graph 插入 KernelAgent prompt
5. 生成一个混合版 kernel 做 smoke test
6. 将混合版拆成真实 autograd 三阶段 kernel
7. 分别验证 forward / backward / double backward
```

---

## 1. 选择算子

来自：

```text
matris_operator_generation_plan.md
```

选择：

```text
bessel_rbf
```

算子定义：

```python
out[i, j] = sin(freq[j] * dist[i]) / dist[i]
```

本次固定频率：

```python
freq = [1, 2, 3, 4]
```

---

## 2. 编写 Hessian extraction builder

文件：

```text
matris_bessel_rbf_builder.py
```

核心接口：

```python
def build_bessel_rbf_case():
    model = BesselRBFCase(num_freq=4)
    dist = torch.linspace(0.2, 5.0, steps=16, dtype=torch.float32)
    return model, [dist]
```

`skills/hessian-extract-graph` 要求 builder 返回：

```python
(model, example_inputs)
```

其中：

- `model` 是 `torch.nn.Module`
- `example_inputs` 是 `list` 或 `tuple`
- 浮点输入会被自动设为 `requires_grad=True`

---

## 3. 提取二阶导 FX graph

命令：

```bash
PYTHONPATH="/share/home/liuyao/workspace/KernelAgent:/share/home/liuyao/workspace/KernelAgent/skills/hessian-extract-graph:${PYTHONPATH}" \
python /share/home/liuyao/workspace/KernelAgent/skills/hessian-extract-graph/extract_hessian_graph.py \
  --builder matris_bessel_rbf_builder:build_bessel_rbf_case \
  --builder-file /share/home/liuyao/workspace/KernelAgent/matris_bessel_rbf_builder.py \
  --mode diagonal \
  --device cpu \
  --dtype float32 \
  --out-dir /share/home/liuyao/workspace/KernelAgent/matris_bessel_rbf_hessian_artifacts
```

输出：

```text
matris_bessel_rbf_hessian_artifacts/fx_graph_code.py
matris_bessel_rbf_hessian_artifacts/metadata.json
```

其中 `fx_graph_code.py` 包含：

```python
return (div, [add], [add_4])
```

含义：

```text
div    -> forward output
add    -> first derivative wrt dist
add_4  -> second derivative wrt dist
```

---

## 4. 构造生成 prompt

文件：

```text
matris_bessel_rbf_generation_prompt.txt
```

prompt 需要包含：

1. 算子名称；
2. forward 定义；
3. backward 目标；
4. double backward 目标；
5. analytic formulas；
6. 从 `hessian-extract-graph` 得到的 FX graph；
7. Triton 生成约束。

本次 smoke test 的目标接口：

```python
def kernel_function(dist):
    return out, grad1, grad2
```

其中：

```text
out   = sin(freq * dist) / dist
grad1 = d out.sum() / d dist
grad2 = d grad1.sum() / d dist
```

注意：这个接口是 diagnostic / smoke test 用，不是最终 autograd 推荐接口。

---

## 5. 生成混合版 kernel

生成 runner：

```text
run_matris_bessel_rbf_generation.py
```

测试文件：

```text
matris_bessel_rbf_test.py
```

执行：

```bash
python /share/home/liuyao/workspace/KernelAgent/run_matris_bessel_rbf_generation.py
```

生成结果：

```text
matris_bessel_rbf_generated_kernel.py
```

测试结果：

```text
Test test_kernel.py passed
Success! Kernel passed test in round 1
```

混合版 kernel 一次性计算：

```python
out, grad1, grad2 = kernel_function(dist)
```

这种形式适合验证：

```text
FX graph -> prompt -> Triton generation -> numerical check
```

但不适合作为真实 MatRIS autograd 算子接口。

---

## 6. 拆成真实 autograd 三阶段

生产推荐形式是拆开：

```text
forward kernel
backward kernel
double backward kernel
```

原因：

- 推理只需要 forward；
- 普通训练只需要 forward + backward；
- force / stress training 才需要 double backward；
- PyTorch autograd 的真实接口不是 `out, grad1, grad2`，而是 staged VJP / double VJP。

---

## 7. Forward kernel

文件：

```text
matris_bessel_rbf_forward_kernel.py
```

接口：

```python
bessel_rbf_forward(dist) -> out
```

数学定义：

```python
out[i, j] = sin(freq[j] * dist[i]) / dist[i]
```

---

## 8. Backward kernel

文件：

```text
matris_bessel_rbf_backward_kernel.py
```

接口：

```python
bessel_rbf_backward(dist, grad_out) -> grad_dist
```

真实 backward 不是固定的：

```python
d out.sum() / d dist
```

而是：

```python
grad_dist[i] = sum_j grad_out[i, j] * d out[i, j] / d dist[i]
```

公式：

```python
d/dr sin(f*r)/r = f*cos(f*r)/r - sin(f*r)/r^2
```

---

## 9. Double backward kernel

文件：

```text
matris_bessel_rbf_double_backward_kernel.py
```

接口：

```python
bessel_rbf_double_backward(
    dist,
    grad_out,
    grad_grad_dist,
) -> grad_grad_out, grad2_dist
```

其中：

```text
grad_grad_dist 是来自上游的二阶反传梯度
```

对于：

```python
grad_dist = sum_j grad_out[j] * f_j'(dist)
```

double backward 需要：

```python
grad_grad_out[j] = grad_grad_dist * f_j'(dist)
grad2_dist       = grad_grad_dist * sum_j grad_out[j] * f_j''(dist)
```

二阶导公式：

```python
f_j''(r)
= -freq[j]^2 * sin(freq[j] * r) / r
  - 2 * freq[j] * cos(freq[j] * r) / r^2
  + 2 * sin(freq[j] * r) / r^3
```

---

## 10. 分阶段验证

验证脚本：

```text
test_matris_bessel_rbf_split_kernels.py
```

验证内容：

1. `bessel_rbf_forward` 对比 PyTorch reference；
2. `bessel_rbf_backward` 对比 `torch.autograd.grad`；
3. `bessel_rbf_double_backward` 对比 PyTorch double backward。

执行：

```bash
python /share/home/liuyao/workspace/KernelAgent/test_matris_bessel_rbf_split_kernels.py
```

结果：

```text
PASS
```

---

## 11. 推荐模板

后续 MatRIS 算子生成应采用这个模式：

```text
一个数学算子
  -> 提取 Hessian FX graph
  -> 生成 diagnostic mixed kernel 做 smoke test
  -> 拆成 forward / backward / double backward 三阶段
  -> 每阶段独立验证
```

最终交付接口应优先是：

```python
op_forward(...)
op_backward(..., grad_out)
op_double_backward(..., grad_out, grad_grad_input)
```

而不是：

```python
op(...)-> out, grad1, grad2
```

---

## 12. 当前产物列表

```text
matris_operator_generation_plan.md
matris_bessel_rbf_builder.py
matris_bessel_rbf_hessian_artifacts/fx_graph_code.py
matris_bessel_rbf_hessian_artifacts/metadata.json
matris_bessel_rbf_generation_prompt.txt
matris_bessel_rbf_test.py
run_matris_bessel_rbf_generation.py
matris_bessel_rbf_generated_kernel.py
matris_bessel_rbf_forward_kernel.py
matris_bessel_rbf_backward_kernel.py
matris_bessel_rbf_double_backward_kernel.py
test_matris_bessel_rbf_split_kernels.py
```

---

## 13. 结论

`hessian-extract-graph` 可以用于把 PyTorch autograd 的二阶导计算图显式化，并作为 KernelAgent prompt 的结构化上下文。

混合版 kernel 适合做生成 smoke test；正式算子应拆成 autograd 三阶段：

```text
forward / backward / double backward
```

阶段内部可以 fusion，但阶段之间不应默认混合。
