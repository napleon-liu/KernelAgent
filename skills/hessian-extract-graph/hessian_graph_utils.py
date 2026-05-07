import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx
from torch.fx import GraphModule, Node
from torch.library import Library, impl, register_fake
from torch._functorch.aot_autograd import aot_module_simplified
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo import lookup_backend
from functools import partial
import contextlib
from torch._inductor.compile_fx import compile_fx_inner as inductor_compile_inner
from torch._inductor import config as inductor_config_module
from contextlib import contextmanager
 
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
 
@contextlib.contextmanager
def fx_duck_shape(enabled: bool):
    """
    For our use of `make_fx` to unfold the autograd graph, we must set the following `use_duck_shape` parameter to `False` (it's `True` by default).
    It forces dynamic batch dims (num_frames, num_atoms, num_edges) to shape specialize if the batch dim is the same as that of a static dim.
    E.g. in training, shape specialization would occur if a weight tensor has a dimension with shape (16,) and we use a batch size of 16 (so the dynamic batch dim `num_frames` is 16) because of the duck shaping.
    """
    # save previous state
    init_duck_shape = torch.fx.experimental._config.use_duck_shape
    # set mode variables
    torch.fx.experimental._config.use_duck_shape = enabled
    try:
        yield
    finally:
        # restore state
        torch.fx.experimental._config.use_duck_shape = init_duck_shape
 
try:
    from hbm_lib import set_graph_opt_fn, graph_hbm_opt_only_gemm
    custom_backend = "aot_hbm_opt"
except ImportError:
    print("[ERROR]!!! hbm_lib not found, using aot_eager backend")
    custom_backend = "aot_eager"
 
def make_fx_compile(model, example_inputs, compile_args):
    torch._dynamo.reset()
    torch.fx.experimental._config.use_duck_shape = False
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    with fx_duck_shape(False):
        fx_graph = make_fx(
                model,
                tracing_mode="symbolic",
                _allow_non_fake_inputs=True,
                _error_on_data_dependent_ops=True)(*example_inputs)
    #return torch.compile(fx_graph, **compile_args)
    try:
        from hbm_lib import set_graph_opt_fn, set_graph_partition_fn, graph_hbm_opt_only_gemm, set_graph_opt_fn, only_gemm_partition
        import functools
        set_graph_partition_fn(only_gemm_partition)
        gemm_opt_configured = functools.partial(
            graph_hbm_opt_only_gemm, 
            move_inputs_to_hbm=True, 
            move_outputs_to_ddr=False
        )
        set_graph_opt_fn(gemm_opt_configured)
    except ImportError:
        pass
 
    return torch.compile(fx_graph, backend=custom_backend, **compile_args)

def make_fx_compile_with_grad(model, example_inputs, compile_args, order=1):
    """
    Compile model with gradient support up to specified order.
    
    Args:
        model: PyTorch model to compile
        example_inputs: Example inputs for tracing
        compile_args: Compilation arguments
        order: Derivative order (1 for first-order, 2 for second-order)
    
    Returns:
        Compiled model with gradient support
    """
    torch._dynamo.reset()
    torch.fx.experimental._config.use_duck_shape = False
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    
    # Ensure inputs require grad for derivative computation
    grad_inputs = []
    for inp in example_inputs:
        if isinstance(inp, torch.Tensor) and inp.dtype.is_floating_point:
            inp_copy = inp.clone().detach().requires_grad_(True)
            grad_inputs.append(inp_copy)
        else:
            grad_inputs.append(inp)
    
    # For second-order derivatives, wrap model to compute gradients
    if order >= 2:
        model = _wrap_model_for_hessian(model)
    
    with fx_duck_shape(False):
        # Trace forward pass (with gradients if order >= 2)
        fx_graph = make_fx(
            model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True
        )(*grad_inputs)
    
    # Apply HBM optimization if available
    try:
        from hbm_lib import set_graph_opt_fn, set_graph_partition_fn, graph_hbm_opt_only_gemm, only_gemm_partition
        import functools
        set_graph_partition_fn(only_gemm_partition)
        gemm_opt_configured = functools.partial(
            graph_hbm_opt_only_gemm, 
            move_inputs_to_hbm=True, 
            move_outputs_to_ddr=False
        )
        set_graph_opt_fn(gemm_opt_configured)
    except ImportError:
        pass
    
    return torch.compile(fx_graph, backend=custom_backend, **compile_args)


def _wrap_model_for_hessian(model):
    """
    Wrap a model to compute first and second-order gradients.
    
    This is used internally by make_fx_compile_with_grad when order=2.
    """
    import torch.nn as nn
    
    class HessianWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, *args):
            # Filter inputs that need gradients
            grad_inputs = [arg for arg in args 
                          if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point]
            
            # Forward pass
            output = self.base_model(*args)
            
            # Compute first-order gradient
            first_grad = torch.autograd.grad(
                output.sum(), 
                grad_inputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            # Compute second-order gradient
            if first_grad is not None and first_grad.requires_grad:
                second_grad = torch.autograd.grad(
                    first_grad.sum(),
                    grad_inputs,
                    retain_graph=False,
                    allow_unused=True
                )[0]
            else:
                second_grad = torch.zeros_like(grad_inputs[0]) if grad_inputs else None
            
            return output, first_grad, second_grad
    
    return HessianWrapper(model)


def extract_hessian_graph(model, example_inputs, mode='diagonal'):
    """
    Extract computation graph including Hessian (second-order derivatives).
    
    This function traces the model and extracts a graph that includes:
    1. Forward pass: y = f(x)
    2. First-order gradient: dy/dx
    3. Second-order gradient: d²y/dx²
    
    Args:
        model: PyTorch model to trace
        example_inputs: Example inputs for tracing
        mode: Hessian computation mode ('diagonal', 'full', or 'hvp')
    
    Returns:
        FX graph module containing forward + gradient + hessian computation
    
    Example:
        >>> from current_op_1.GraphPooling import GraphPooling
        >>> model = GraphPooling(average=False)
        >>> node_feat = torch.randn(100, 64, requires_grad=True)
        >>> segment = torch.randint(0, 10, (100,))
        >>> fx_graph = extract_hessian_graph(model, [node_feat, segment])
        >>> output, grad1, grad2 = fx_graph(node_feat, segment)
    """
    from hessian_operator import HessianOperator, HessianMode
    
    # Convert mode string to enum
    if isinstance(mode, str):
        mode = HessianMode(mode)
    
    # Wrap model with HessianOperator
    hessian_model = HessianOperator(model, mode=mode)
    
    # Ensure inputs require grad
    grad_inputs = []
    for inp in example_inputs:
        if isinstance(inp, torch.Tensor) and inp.dtype.is_floating_point:
            inp_copy = inp.clone().detach().requires_grad_(True)
            grad_inputs.append(inp_copy)
        else:
            grad_inputs.append(inp)
    
    # Reset dynamo state
    torch._dynamo.reset()
    torch.fx.experimental._config.use_duck_shape = False
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    
    # Trace the Hessian computation
    with fx_duck_shape(False):
        fx_graph = make_fx(
            hessian_model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True
        )(*grad_inputs)
    
    return fx_graph


def visualize_hessian_graph(fx_graph, output_path=None):
    """
    Visualize the Hessian computation graph.
    
    Args:
        fx_graph: FX graph module from extract_hessian_graph
        output_path: Optional path to save visualization (as .svg or .png)
    
    Returns:
        Graph code as string
    """
    graph_code = fx_graph.code
    
    if output_path:
        # Try to generate a visual graph if graphviz is available
        try:
            import graphviz
            from torch.fx.passes.graph_drawer import FxGraphDrawer
            
            drawer = FxGraphDrawer(fx_graph, "hessian_graph")
            dot_graph = drawer.get_dot_graph()
            
            # Determine format from extension
            fmt = output_path.split('.')[-1] if '.' in output_path else 'svg'
            dot_graph.render(output_path.rsplit('.', 1)[0], format=fmt, cleanup=True)
            print(f"Graph visualization saved to: {output_path}")
        except ImportError:
            print("graphviz not available, skipping visualization")
    
    return graph_code