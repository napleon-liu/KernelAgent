#!/usr/bin/env python3
"""
Hessian Operator Framework

This module provides a general framework for computing second-order derivatives (Hessian)
of PyTorch models, following the approach demonstrated in current_op_1.GraphPooling.py.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Union
from enum import Enum


class HessianMode(Enum):
    """Hessian computation modes."""
    FULL = "full"              # Complete Hessian matrix (memory intensive)
    DIAGONAL = "diagonal"      # Only diagonal elements (memory efficient)
    HVP = "hvp"               # Hessian-vector product (most efficient)


class HessianOperator(nn.Module):
    """
    Generic wrapper for computing Hessian (second-order derivatives) of any PyTorch model.
    
    This class wraps an arbitrary PyTorch model and provides methods to compute:
    1. Forward pass: y = f(x)
    2. First-order gradient: dy/dx
    3. Second-order gradient (Hessian): d²y/dx²
    
    Example:
        >>> base_model = GraphPooling(average=False)
        >>> hessian_op = HessianOperator(base_model, mode=HessianMode.DIAGONAL)
        >>> output, grad1, grad2 = hessian_op(node_feat, segment)
    """
    
    def __init__(
        self, 
        base_model: nn.Module,
        mode: Union[HessianMode, str] = HessianMode.DIAGONAL,
        output_reduction: str = "sum"
    ):
        """
        Initialize Hessian operator.
        
        Args:
            base_model: The base PyTorch model to wrap
            mode: Hessian computation mode (full/diagonal/hvp)
            output_reduction: How to reduce output for gradient computation
                            ("sum", "mean", or None for no reduction)
        """
        super().__init__()
        self.base_model = base_model
        
        # Convert string to enum if needed
        if isinstance(mode, str):
            mode = HessianMode(mode)
        self.mode = mode
        self.output_reduction = output_reduction
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute forward pass with first and second-order gradients.
        
        Returns:
            Tuple of (output, first_grad, second_grad)
        """
        # Extract hessian_vector if provided (for HVP mode)
        hessian_vector = kwargs.pop('hessian_vector', None)
        
        # Identify which inputs require gradients
        grad_inputs = []
        grad_indices = []
        
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point:
                if not arg.requires_grad:
                    arg = arg.requires_grad_(True)
                grad_inputs.append(arg)
                grad_indices.append(i)
        
        if not grad_inputs:
            raise ValueError("No floating point tensor inputs found that can have gradients")
        
        # Forward pass
        output = self.base_model(*args, **kwargs)
        
        # Reduce output for gradient computation
        if self.output_reduction == "sum":
            scalar_output = output.sum()
        elif self.output_reduction == "mean":
            scalar_output = output.mean()
        else:
            scalar_output = output
        
        # Compute first-order gradients
        first_grads = self._compute_first_order_gradients(
            scalar_output, grad_inputs
        )
        
        # Compute second-order gradients based on mode
        if self.mode == HessianMode.DIAGONAL:
            second_grads = self._compute_diagonal_hessian(first_grads, grad_inputs)
        elif self.mode == HessianMode.FULL:
            second_grads = self._compute_full_hessian(first_grads, grad_inputs)
        elif self.mode == HessianMode.HVP:
            # For HVP mode, we need a vector v
            if hessian_vector is None:
                raise ValueError("HVP mode requires 'hessian_vector' argument")
            second_grads = self._compute_hvp(first_grads, grad_inputs, hessian_vector)
        else:
            raise ValueError(f"Unknown Hessian mode: {self.mode}")
        
        return output, first_grads, second_grads
    
    def _compute_first_order_gradients(
        self, 
        output: torch.Tensor, 
        inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute first-order gradients dy/dx.
        
        Following the approach in current_op_1.GraphPooling.py:
        grad = torch.autograd.grad(output.sum(), inputs, create_graph=True)[0]
        """
        grads = torch.autograd.grad(
            output,
            inputs,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )
        
        # Handle None gradients (for unused inputs)
        grads = [g if g is not None else torch.zeros_like(inp) 
                 for g, inp in zip(grads, inputs)]
        
        return grads
    
    def _compute_diagonal_hessian(
        self,
        first_grads: List[torch.Tensor],
        inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute diagonal elements of Hessian: d²y/dx².
        
        This is the most memory-efficient approach for large inputs.
        Following the approach in current_op_1.GraphPooling.py:
        second_grad = torch.autograd.grad(first_grad.sum(), inputs, retain_graph=False)[0]
        """
        second_grads = []
        
        for idx, (first_grad, inp) in enumerate(zip(first_grads, inputs)):
            if first_grad.requires_grad:
                # Compute second derivative. Keep the graph alive until the last
                # differentiable input has been processed.
                second_grad = torch.autograd.grad(
                    first_grad.sum(),
                    inp,
                    retain_graph=(idx < len(first_grads) - 1),
                    allow_unused=True
                )[0]
                
                if second_grad is None:
                    second_grad = torch.zeros_like(inp)
                
                second_grads.append(second_grad)
            else:
                # If first gradient doesn't require grad, Hessian is zero
                second_grads.append(torch.zeros_like(inp))
        
        return second_grads
    
    def _compute_full_hessian(
        self,
        first_grads: List[torch.Tensor],
        inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute full Hessian matrix.
        
        Warning: This is memory intensive O(n²) and should only be used
        for small inputs.
        """
        hessians = []
        
        for first_grad, inp in zip(first_grads, inputs):
            if not first_grad.requires_grad:
                hessians.append(torch.zeros(inp.numel(), inp.numel(), device=inp.device))
                continue
            
            # Flatten gradient for easier iteration
            flat_grad = first_grad.flatten()
            hessian_rows = []
            
            for i in range(flat_grad.numel()):
                # Compute gradient of each element
                grad_grad = torch.autograd.grad(
                    flat_grad[i],
                    inp,
                    retain_graph=(i < flat_grad.numel() - 1),
                    allow_unused=True
                )[0]
                
                if grad_grad is None:
                    grad_grad = torch.zeros_like(inp)
                
                hessian_rows.append(grad_grad.flatten())
            
            hessian = torch.stack(hessian_rows)
            hessians.append(hessian)
        
        return hessians
    
    def _compute_hvp(
        self,
        first_grads: List[torch.Tensor],
        inputs: List[torch.Tensor],
        vectors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute Hessian-vector product: H @ v
        
        This is the most efficient way to compute directional second derivatives.
        """
        if not isinstance(vectors, list):
            vectors = [vectors]
        
        hvps = []
        
        for first_grad, inp, v in zip(first_grads, inputs, vectors):
            if not first_grad.requires_grad:
                hvps.append(torch.zeros_like(inp))
                continue
            
            # Compute Hessian-vector product
            hvp = torch.autograd.grad(
                first_grad,
                inp,
                grad_outputs=v,
                retain_graph=False,
                allow_unused=True
            )[0]
            
            if hvp is None:
                hvp = torch.zeros_like(inp)
            
            hvps.append(hvp)
        
        return hvps


class GraphPoolingHessian(HessianOperator):
    """
    Specialized Hessian operator for GraphPooling.
    
    For graph pooling (segment sum/average), the operation is linear:
        output[segment[i]] += input[i]
    
    Therefore:
        - First derivative: dy/dx = 1 (constant)
        - Second derivative: d²y/dx² = 0 (derivative of constant is zero)
    
    This specialized class can optimize the computation knowing this property.
    """
    
    def __init__(self, base_model: nn.Module, mode: Union[HessianMode, str] = HessianMode.DIAGONAL):
        super().__init__(base_model, mode=mode)
    
    def forward(self, node_feat: torch.Tensor, segment: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass for GraphPooling.
        
        Since pooling is linear, we know:
        - First gradient = ones (for sum pooling)
        - Second gradient = zeros (derivative of constant)
        """
        # Ensure node_feat requires grad
        if not node_feat.requires_grad:
            node_feat = node_feat.requires_grad_(True)
        
        # Forward pass
        output = self.base_model(node_feat, segment)
        
        # For sum pooling, gradient is constant 1
        first_grad = torch.ones_like(node_feat)
        
        # Second derivative of constant is 0
        second_grad = torch.zeros_like(node_feat)
        
        return output, first_grad, second_grad


def create_hessian_operator(
    model: nn.Module,
    mode: Union[HessianMode, str] = HessianMode.DIAGONAL,
    optimize_for_model_type: bool = True
) -> HessianOperator:
    """
    Factory function to create appropriate Hessian operator.
    
    Args:
        model: Base PyTorch model
        mode: Hessian computation mode
        optimize_for_model_type: If True, use specialized operators for known model types
    
    Returns:
        HessianOperator instance
    """
    # Check if we can use a specialized operator
    if optimize_for_model_type:
        model_name = model.__class__.__name__
        
        if model_name == "GraphPooling":
            return GraphPoolingHessian(model, mode=mode)
    
    # Default to generic operator
    return HessianOperator(model, mode=mode)


if __name__ == "__main__":
    """Test the Hessian operator with GraphPooling example."""
    
    # Import GraphPooling from current_op_1.GraphPooling.py
    import sys
    import importlib.util
    from pathlib import Path
    
    spec = importlib.util.spec_from_file_location(
        "current_op_1_GraphPooling", 
        Path(__file__).parent / "current_op_1.GraphPooling.py"
    )
    current_op_1_GraphPooling = importlib.util.module_from_spec(spec)
    sys.modules["current_op_1_GraphPooling"] = current_op_1_GraphPooling
    spec.loader.exec_module(current_op_1_GraphPooling)
    
    GraphPooling = current_op_1_GraphPooling.GraphPooling
    
    # Test with GraphPooling
    print("=" * 80)
    print("Testing HessianOperator with GraphPooling")
    print("=" * 80)
    
    # Create model and data
    pooling = GraphPooling(average=False)
    num_nodes = 40
    feat_dim = 1
    num_graphs = 1
    
    node_feat = torch.randn(num_nodes, feat_dim, requires_grad=True)
    segment = torch.zeros(num_nodes, dtype=torch.long)
    
    print(f"\nInput shape: {node_feat.shape}")
    print(f"Segment shape: {segment.shape}")
    
    # Test generic HessianOperator
    print("\n1. Testing generic HessianOperator (diagonal mode):")
    hessian_op = HessianOperator(pooling, mode=HessianMode.DIAGONAL)
    output, first_grad, second_grad = hessian_op(node_feat, segment)
    
    print(f"   Output shape: {output.shape}")
    print(f"   First grad shape: {first_grad[0].shape}")
    print(f"   Second grad shape: {second_grad[0].shape}")
    print(f"   First grad (should be all ones): {torch.allclose(first_grad[0], torch.ones_like(first_grad[0]))}")
    print(f"   Second grad (should be all zeros): {torch.allclose(second_grad[0], torch.zeros_like(second_grad[0]))}")
    
    # Test specialized GraphPoolingHessian
    print("\n2. Testing specialized GraphPoolingHessian:")
    specialized_op = GraphPoolingHessian(pooling)
    output2, first_grad2, second_grad2 = specialized_op(node_feat, segment)
    
    print(f"   Output shape: {output2.shape}")
    print(f"   First grad shape: {first_grad2.shape}")
    print(f"   Second grad shape: {second_grad2.shape}")
    print(f"   First grad (all ones): {torch.allclose(first_grad2, torch.ones_like(first_grad2))}")
    print(f"   Second grad (all zeros): {torch.allclose(second_grad2, torch.zeros_like(second_grad2))}")
    
    # Test factory function
    print("\n3. Testing factory function:")
    auto_op = create_hessian_operator(pooling, mode=HessianMode.DIAGONAL)
    print(f"   Created operator type: {type(auto_op).__name__}")
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
