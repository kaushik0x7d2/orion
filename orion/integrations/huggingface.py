"""
HuggingFace → Orion FHE model converter.

Converts PyTorch models (from HuggingFace or any source) into
FHE-compatible Orion models by replacing standard torch.nn layers
with their orion.nn equivalents and transferring weights.

Supported layers:
    torch.nn.Linear      → orion.nn.Linear
    torch.nn.Conv2d      → orion.nn.Conv2d
    torch.nn.BatchNorm1d → orion.nn.BatchNorm1d
    torch.nn.BatchNorm2d → orion.nn.BatchNorm2d
    torch.nn.AvgPool2d   → orion.nn.AvgPool2d
    torch.nn.AdaptiveAvgPool2d → orion.nn.AdaptiveAvgPool2d
    torch.nn.Flatten     → orion.nn.Flatten
    torch.nn.ReLU        → orion.nn.ReLU (polynomial approx)
    torch.nn.GELU        → orion.nn.GELU (polynomial approx)
    torch.nn.SiLU        → orion.nn.SiLU (polynomial approx)
    torch.nn.Sigmoid     → orion.nn.Sigmoid (polynomial approx)
    torch.nn.ELU         → orion.nn.ELU (polynomial approx)
    torch.nn.SELU        → orion.nn.SELU (polynomial approx)
    torch.nn.Mish        → orion.nn.Mish (polynomial approx)
    torch.nn.Softplus    → orion.nn.Softplus (polynomial approx)
    torch.nn.Identity    → (removed, passthrough)
    torch.nn.Dropout*    → (removed, not needed for inference)

NOT supported (will be reported):
    torch.nn.MultiheadAttention, LayerNorm, MaxPool2d,
    Softmax, Transformer*, RNN/LSTM/GRU, Embedding

Usage:
    from orion.integrations import convert_to_orion, check_compatibility

    # Check if a model is compatible
    report = check_compatibility(hf_model)
    print(report)

    # Convert a compatible model
    orion_model = convert_to_orion(hf_model, activation_degree=7)
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Type

import torch
import torch.nn as nn

import orion.nn as on

logger = logging.getLogger("orion.integrations.huggingface")

# ================================================================
#  Layer Mapping Tables
# ================================================================

# Layers that map directly to Orion equivalents
SUPPORTED_LAYERS = {
    nn.Linear: "Linear",
    nn.Conv2d: "Conv2d",
    nn.BatchNorm1d: "BatchNorm1d",
    nn.BatchNorm2d: "BatchNorm2d",
    nn.AvgPool2d: "AvgPool2d",
    nn.AdaptiveAvgPool2d: "AdaptiveAvgPool2d",
    nn.Flatten: "Flatten",
    nn.LayerNorm: "FHELayerNorm",
}

# Activations that can be polynomial-approximated
SUPPORTED_ACTIVATIONS = {
    nn.ReLU: "ReLU",
    nn.GELU: "GELU",
    nn.SiLU: "SiLU",
    nn.Sigmoid: "Sigmoid",
    nn.ELU: "ELU",
    nn.SELU: "SELU",
    nn.Mish: "Mish",
    nn.Softplus: "Softplus",
}

# Layers that can be safely removed for inference
REMOVABLE_LAYERS = {
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.Identity,
}

# Layers that CANNOT work under FHE
UNSUPPORTED_LAYERS = {
    nn.Softmax: "Use PolySoftmax (x^p/sum(x^p)) instead",
    nn.LogSoftmax: "Requires exponentiation and logarithm",
    nn.GroupNorm: "Requires mean/variance computation and division",
    nn.InstanceNorm1d: "Requires mean/variance computation",
    nn.InstanceNorm2d: "Requires mean/variance computation",
    nn.MaxPool1d: "Requires comparison (argmax)",
    nn.MaxPool2d: "Requires comparison (argmax)",
    nn.MaxPool3d: "Requires comparison (argmax)",
    nn.AdaptiveMaxPool2d: "Requires comparison (argmax)",
    nn.Embedding: "Lookup tables not supported on encrypted indices",
    nn.RNN: "Sequential recurrence not supported",
    nn.LSTM: "Sequential recurrence with gating not supported",
    nn.GRU: "Sequential recurrence with gating not supported",
    nn.MultiheadAttention: "Use FHEMultiHeadAttention with PolySoftmax",
    nn.TransformerEncoderLayer: "Use FHETransformerEncoderLayer instead",
    nn.TransformerDecoderLayer: "Use FHETransformerEncoderLayer instead",
    nn.Tanh: "Use SiLU or GELU instead (better polynomial approximation)",
}


# ================================================================
#  Compatibility Report
# ================================================================

@dataclass
class FHECompatibilityReport:
    """Report on whether a model can be converted for FHE inference."""
    model_name: str
    compatible: bool = True
    total_layers: int = 0
    supported_count: int = 0
    removable_count: int = 0
    unsupported_count: int = 0
    supported_layers: List[str] = field(default_factory=list)
    removable_layers: List[str] = field(default_factory=list)
    unsupported_layers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    estimated_depth: int = 0

    def __str__(self):
        lines = [
            f"FHE Compatibility Report: {self.model_name}",
            f"{'=' * 50}",
            f"Compatible: {'YES' if self.compatible else 'NO'}",
            f"",
            f"Layers: {self.total_layers} total",
            f"  Supported:    {self.supported_count}",
            f"  Removable:    {self.removable_count} (dropout/identity)",
            f"  Unsupported:  {self.unsupported_count}",
        ]

        if self.supported_layers:
            lines.append(f"\nSupported layers:")
            for layer in self.supported_layers:
                lines.append(f"  + {layer}")

        if self.removable_layers:
            lines.append(f"\nRemovable layers (dropped for FHE):")
            for layer in self.removable_layers:
                lines.append(f"  ~ {layer}")

        if self.unsupported_layers:
            lines.append(f"\nUnsupported layers (BLOCKERS):")
            for layer in self.unsupported_layers:
                lines.append(f"  X {layer}")

        if self.warnings:
            lines.append(f"\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        if self.compatible:
            lines.append(f"\nEstimated multiplicative depth: ~{self.estimated_depth}")
            lines.append(f"Recommendation: Use LogN=14 with at least "
                        f"{self.estimated_depth + 4} LogQ levels.")

        return "\n".join(lines)


def check_compatibility(model: nn.Module, model_name: str = None) -> FHECompatibilityReport:
    """
    Check if a PyTorch model can be converted for FHE inference.

    Args:
        model: Any torch.nn.Module (from HuggingFace, torchvision, etc.)
        model_name: Optional display name for the report.

    Returns:
        FHECompatibilityReport with detailed analysis.
    """
    if model_name is None:
        model_name = model.__class__.__name__

    report = FHECompatibilityReport(model_name=model_name)

    depth = 0
    for name, module in model.named_modules():
        if name == "":
            continue  # skip root

        module_type = type(module)
        report.total_layers += 1

        # Check if it's a container (Sequential, ModuleList, etc.)
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            report.total_layers -= 1  # don't count containers
            continue

        # Already an orion.nn layer — FHE-compatible
        if isinstance(module, on.Module):
            has_children = any(True for _ in module.children())
            if has_children:
                # Container-like Orion module — skip, check children
                report.total_layers -= 1
            else:
                # Leaf Orion module — already FHE-compatible
                label = f"{name}: {module_type.__name__} (FHE-compatible)"
                report.supported_layers.append(label)
                report.supported_count += 1
                if hasattr(module, 'depth') and module.depth is not None:
                    depth += module.depth
            continue

        if module_type in SUPPORTED_LAYERS:
            label = f"{name}: {module_type.__name__} → orion.nn.{SUPPORTED_LAYERS[module_type]}"
            report.supported_layers.append(label)
            report.supported_count += 1
            # Linear/Conv consume 1 level, LayerNorm consumes 2
            if module_type in (nn.Linear, nn.Conv2d):
                depth += 1
            elif module_type == nn.LayerNorm:
                depth += 2

        elif module_type in SUPPORTED_ACTIVATIONS:
            label = f"{name}: {module_type.__name__} → orion.nn.{SUPPORTED_ACTIVATIONS[module_type]}"
            report.supported_layers.append(label)
            report.supported_count += 1
            # Polynomial activations consume ~3-4 levels for degree 7
            depth += 4

        elif module_type in REMOVABLE_LAYERS:
            label = f"{name}: {module_type.__name__} (removed for inference)"
            report.removable_layers.append(label)
            report.removable_count += 1

        elif module_type in UNSUPPORTED_LAYERS:
            reason = UNSUPPORTED_LAYERS[module_type]
            label = f"{name}: {module_type.__name__} — {reason}"
            report.unsupported_layers.append(label)
            report.unsupported_count += 1

        else:
            # Unknown layer — warn but don't block
            label = f"{name}: {module_type.__name__} (unknown — may not work)"
            report.warnings.append(label)
            report.unsupported_count += 1
            report.unsupported_layers.append(label)

    report.estimated_depth = depth
    report.compatible = report.unsupported_count == 0

    # Depth warnings
    if depth > 20:
        report.warnings.append(
            f"Estimated depth ({depth}) is high. May need multiple bootstraps "
            f"and large LogQ chain."
        )

    return report


# ================================================================
#  Model Converter
# ================================================================

class HFModelConverter:
    """
    Converts a standard PyTorch model to an FHE-compatible Orion model.

    The converter:
    1. Walks the module tree
    2. Replaces torch.nn layers with orion.nn equivalents
    3. Transfers weights from the original model
    4. Removes dropout/identity layers

    Args:
        activation_degree: Polynomial degree for activation approximations.
            Higher = more accurate but slower. Default: 7.
        relu_degrees: Degrees for composite ReLU (3-stage). Default: [15,15,27].
    """
    def __init__(self, activation_degree: int = 7, relu_degrees: list = None):
        self.activation_degree = activation_degree
        self.relu_degrees = relu_degrees or [15, 15, 27]

    def _convert_linear(self, module: nn.Linear) -> on.Linear:
        """Convert torch.nn.Linear to orion.nn.Linear."""
        orion_layer = on.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
        orion_layer.weight = module.weight
        if module.bias is not None:
            orion_layer.bias = module.bias
        return orion_layer

    def _convert_conv2d(self, module: nn.Conv2d) -> on.Conv2d:
        """Convert torch.nn.Conv2d to orion.nn.Conv2d."""
        orion_layer = on.Conv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
            stride=module.stride[0] if isinstance(module.stride, tuple) else module.stride,
            padding=module.padding[0] if isinstance(module.padding, tuple) else module.padding,
            dilation=module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
        )
        orion_layer.weight = module.weight
        if module.bias is not None:
            orion_layer.bias = module.bias
        return orion_layer

    def _convert_batchnorm1d(self, module: nn.BatchNorm1d) -> on.BatchNorm1d:
        """Convert torch.nn.BatchNorm1d to orion.nn.BatchNorm1d."""
        orion_layer = on.BatchNorm1d(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
        )
        if module.affine:
            orion_layer.weight = module.weight
            orion_layer.bias = module.bias
        orion_layer.running_mean = module.running_mean
        orion_layer.running_var = module.running_var
        orion_layer.num_batches_tracked = module.num_batches_tracked
        return orion_layer

    def _convert_batchnorm2d(self, module: nn.BatchNorm2d) -> on.BatchNorm2d:
        """Convert torch.nn.BatchNorm2d to orion.nn.BatchNorm2d."""
        orion_layer = on.BatchNorm2d(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
        )
        if module.affine:
            orion_layer.weight = module.weight
            orion_layer.bias = module.bias
        orion_layer.running_mean = module.running_mean
        orion_layer.running_var = module.running_var
        orion_layer.num_batches_tracked = module.num_batches_tracked
        return orion_layer

    def _convert_layernorm(self, module: nn.LayerNorm):
        """Convert torch.nn.LayerNorm to orion.nn.FHELayerNorm."""
        from orion.nn.transformer import FHELayerNorm
        orion_layer = FHELayerNorm(
            normalized_shape=module.normalized_shape,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
        )
        if module.elementwise_affine:
            orion_layer.weight = module.weight
            orion_layer.bias = module.bias
        return orion_layer

    def _convert_avgpool2d(self, module: nn.AvgPool2d):
        """Convert torch.nn.AvgPool2d to orion.nn.AvgPool2d."""
        ks = module.kernel_size
        if isinstance(ks, tuple):
            ks = ks[0]
        stride = module.stride
        if isinstance(stride, tuple):
            stride = stride[0]
        padding = module.padding
        if isinstance(padding, tuple):
            padding = padding[0]
        return on.AvgPool2d(kernel_size=ks, stride=stride, padding=padding)

    def _convert_adaptive_avgpool2d(self, module: nn.AdaptiveAvgPool2d):
        """Convert torch.nn.AdaptiveAvgPool2d to orion.nn.AdaptiveAvgPool2d."""
        output_size = module.output_size
        if isinstance(output_size, tuple):
            output_size = output_size[0]
        return on.AdaptiveAvgPool2d(output_size=output_size)

    def _convert_activation(self, module: nn.Module):
        """Convert a torch activation to its polynomial Orion equivalent."""
        module_type = type(module)
        degree = self.activation_degree

        mapping = {
            nn.GELU: lambda: on.GELU(degree=degree),
            nn.SiLU: lambda: on.SiLU(degree=degree),
            nn.Sigmoid: lambda: on.Sigmoid(degree=degree),
            nn.ELU: lambda: on.ELU(
                alpha=getattr(module, 'alpha', 1.0), degree=degree),
            nn.SELU: lambda: on.SELU(degree=degree),
            nn.Mish: lambda: on.Mish(degree=degree),
            nn.Softplus: lambda: on.Softplus(degree=degree),
            nn.ReLU: lambda: on.ReLU(degrees=self.relu_degrees),
        }

        factory = mapping.get(module_type)
        if factory:
            return factory()
        return None

    def _convert_module(self, module: nn.Module):
        """Convert a single torch.nn module to its Orion equivalent."""
        module_type = type(module)

        # Linear transforms
        if module_type == nn.Linear:
            return self._convert_linear(module)
        elif module_type == nn.Conv2d:
            return self._convert_conv2d(module)

        # Normalization
        elif module_type == nn.BatchNorm1d:
            return self._convert_batchnorm1d(module)
        elif module_type == nn.BatchNorm2d:
            return self._convert_batchnorm2d(module)
        elif module_type == nn.LayerNorm:
            return self._convert_layernorm(module)

        # Pooling
        elif module_type == nn.AvgPool2d:
            return self._convert_avgpool2d(module)
        elif module_type == nn.AdaptiveAvgPool2d:
            return self._convert_adaptive_avgpool2d(module)

        # Reshape
        elif module_type == nn.Flatten:
            return on.Flatten()

        # Activations
        elif module_type in SUPPORTED_ACTIVATIONS:
            return self._convert_activation(module)

        # Removable
        elif module_type in REMOVABLE_LAYERS:
            return nn.Identity()  # will be skipped in forward

        # Already an orion.nn layer — no conversion needed
        elif isinstance(module, on.Module):
            return None

        return None

    def convert(self, model: nn.Module, model_class_name: str = None) -> on.Module:
        """
        Convert a PyTorch model to an Orion FHE-compatible model.

        This performs a recursive module replacement: for each leaf module in
        the model, it replaces the torch.nn layer with its orion.nn equivalent.

        Args:
            model: The source PyTorch model (trained, with weights).
            model_class_name: Optional name for the generated class.

        Returns:
            An orion.nn.Module with the same architecture and weights.

        Raises:
            ValueError: If the model contains unsupported layers.
        """
        # First check compatibility
        report = check_compatibility(model, model_class_name)
        if not report.compatible:
            raise ValueError(
                f"Model contains unsupported layers:\n"
                + "\n".join(f"  - {l}" for l in report.unsupported_layers)
                + "\n\nSee check_compatibility() for details."
            )

        model = copy.deepcopy(model)
        model.eval()

        # Recursively replace modules
        self._replace_modules(model)

        # Change the base class to orion.nn.Module
        # We create a wrapper that inherits from on.Module
        wrapper = _OrionWrapper(model)
        return wrapper

    def _replace_modules(self, model: nn.Module):
        """Recursively replace torch.nn modules with Orion equivalents."""
        for name, child in model.named_children():
            converted = self._convert_module(child)
            if converted is not None:
                setattr(model, name, converted)
                logger.debug("Converted %s: %s → %s",
                           name, type(child).__name__, type(converted).__name__)
            else:
                # Recurse into container modules
                self._replace_modules(child)


class _OrionWrapper(on.Module):
    """
    Wraps a converted model as an orion.nn.Module.

    This wrapper delegates forward() to the inner model while providing
    the orion.nn.Module interface (he(), eval(), scheme, etc.).
    """
    def __init__(self, inner_model: nn.Module):
        super().__init__()
        # Register all child modules from inner model
        for name, module in inner_model.named_children():
            self.add_module(name, module)

    def forward(self, x):
        # Walk through child modules in order
        for module in self.children():
            if isinstance(module, nn.Identity):
                continue  # skip removed layers (dropout, etc.)
            x = module(x)
        return x


# ================================================================
#  Convenience Functions
# ================================================================

def convert_to_orion(
    model: nn.Module,
    activation_degree: int = 7,
    relu_degrees: list = None,
    model_name: str = None,
) -> on.Module:
    """
    Convert a PyTorch model to an FHE-compatible Orion model.

    This is the main entry point for HuggingFace integration. Takes any
    PyTorch model and returns an Orion model with:
    - Linear/Conv layers replaced with FHE-compatible versions
    - Activations replaced with polynomial approximations
    - Weights transferred from the original model
    - Dropout layers removed

    Args:
        model: A trained PyTorch model.
        activation_degree: Polynomial degree for activations (3, 7, 15, 31).
            Higher = more accurate but more FHE levels consumed.
        relu_degrees: Composite ReLU degrees [stage1, stage2, stage3].
        model_name: Display name for logging.

    Returns:
        An orion.nn.Module ready for orion.fit() and orion.compile().

    Raises:
        ValueError: If model contains unsupported layers (Softmax, etc.)

    Example:
        from transformers import AutoModel
        from orion.integrations import convert_to_orion, check_compatibility

        # Load a small MLP from HuggingFace
        hf_model = AutoModel.from_pretrained("my-tabular-model")

        # Check compatibility first
        report = check_compatibility(hf_model)
        print(report)

        # Convert to Orion
        if report.compatible:
            orion_model = convert_to_orion(hf_model, activation_degree=7)
            # Now use with Orion FHE pipeline:
            # orion.fit(orion_model, data)
            # orion.compile(orion_model)
            # ...
    """
    converter = HFModelConverter(
        activation_degree=activation_degree,
        relu_degrees=relu_degrees,
    )
    return converter.convert(model, model_class_name=model_name)


def from_huggingface(
    model_id: str,
    activation_degree: int = 7,
    trust_remote_code: bool = False,
) -> on.Module:
    """
    Load a model directly from HuggingFace Hub and convert for FHE.

    Args:
        model_id: HuggingFace model ID (e.g., "username/my-model").
        activation_degree: Polynomial degree for activations.
        trust_remote_code: Whether to trust remote code (for custom architectures).

    Returns:
        An orion.nn.Module ready for FHE.

    Raises:
        ImportError: If transformers library is not installed.
        ValueError: If model contains unsupported layers.
    """
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "HuggingFace transformers not installed. "
            "pip install transformers"
        )

    logger.info("Loading model from HuggingFace: %s", model_id)
    hf_model = AutoModel.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    hf_model.eval()

    return convert_to_orion(hf_model, activation_degree=activation_degree,
                           model_name=model_id)
