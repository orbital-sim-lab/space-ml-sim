"""Compute module: fault injection, TMR, checkpointing, and scheduling."""

from space_ml_sim.compute.fault_injector import FaultInjector, FaultReport
from space_ml_sim.compute.transformer_fault import TransformerFaultInjector
from space_ml_sim.compute.tmr import TMRWrapper
from space_ml_sim.compute.checkpoint import CheckpointManager
from space_ml_sim.compute.scheduler import InferenceScheduler
from space_ml_sim.compute.quantization import (
    quantize_model,
    compare_quantization_resilience,
    plot_quantization_comparison,
)

__all__ = [
    "FaultInjector",
    "FaultReport",
    "TransformerFaultInjector",
    "TMRWrapper",
    "CheckpointManager",
    "InferenceScheduler",
    "quantize_model",
    "compare_quantization_resilience",
    "plot_quantization_comparison",
]

# Optional ONNX support — only available when 'onnx' and 'onnxruntime' are installed.
try:
    from space_ml_sim.compute.onnx_adapter import OnnxModel, load_onnx

    __all__ += ["OnnxModel", "load_onnx"]
except ImportError:
    pass  # onnx extras not installed
