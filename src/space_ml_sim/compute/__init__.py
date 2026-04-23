"""Compute module: fault injection, TMR, checkpointing, scheduling, and distributed inference."""

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
from space_ml_sim.compute.distributed import (
    DistributedInferenceTask,
    DistributedResult,
    DistributedExecutor,
)
from space_ml_sim.compute.model_parallel import (
    partition_model,
    PipelineExecutor,
    PipelineResult,
)
from space_ml_sim.compute.federated import (
    compress_gradients,
    fed_avg,
    FederatedCoordinator,
    FederatedRoundResult,
)
from space_ml_sim.compute.tmr_recommender import (
    TMRRecommendation,
    recommend_tmr,
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
    "DistributedInferenceTask",
    "DistributedResult",
    "DistributedExecutor",
    "partition_model",
    "PipelineExecutor",
    "PipelineResult",
    "compress_gradients",
    "fed_avg",
    "FederatedCoordinator",
    "FederatedRoundResult",
    "TMRRecommendation",
    "recommend_tmr",
]

# Optional ONNX support — only available when 'onnx' and 'onnxruntime' are installed.
try:
    from space_ml_sim.compute.onnx_adapter import OnnxModel, load_onnx

    __all__ += ["OnnxModel", "load_onnx"]
except ImportError:
    pass  # onnx extras not installed
