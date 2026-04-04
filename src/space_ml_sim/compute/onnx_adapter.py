"""ONNX model adapter for fault injection.

Allows loading .onnx models and running fault injection without
requiring users to define PyTorch model code. Makes onnxruntime
an optional dependency — importing this module never raises ImportError;
only calling OnnxModel() or load_onnx() will raise if the packages are absent.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from space_ml_sim.compute.fault_injector import FaultInjector


def _require_onnx():
    """Check that onnx and onnxruntime are installed.

    Returns:
        Tuple of (onnx module, onnxruntime module).

    Raises:
        ImportError: With a pip install hint when either package is missing.
    """
    try:
        import onnx
        import onnxruntime

        # Guard against the test harness placing None into sys.modules
        if onnx is None or onnxruntime is None:
            raise ImportError("onnx or onnxruntime is None")

        return onnx, onnxruntime
    except (ImportError, TypeError) as e:
        raise ImportError(
            "ONNX support requires 'onnx' and 'onnxruntime'. "
            "Install with: pip install space-ml-sim[onnx]"
        ) from e


class OnnxModel:
    """Wrapper around an ONNX model for fault injection.

    Extracts weight initializers from the ONNX graph as float32 torch tensors,
    allows bit-flip fault injection via FaultInjector.flip_random_bits, then
    rebuilds the ONNX InferenceSession with the modified weights before running
    inference.

    The onnx and onnxruntime packages are imported lazily inside __init__ so
    that importing this module itself never fails on a system that lacks those
    packages.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Load an ONNX model.

        Args:
            model_path: Path to a .onnx file.

        Raises:
            ImportError: If onnx/onnxruntime are not installed.
            FileNotFoundError: If model_path does not exist.
        """
        onnx_mod, ort = _require_onnx()

        self._path = Path(model_path)
        if not self._path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._path}")

        self._onnx = onnx_mod
        self._ort = ort

        # Load and validate the protobuf model graph.
        self._model = onnx_mod.load(str(self._path))
        onnx_mod.checker.check_model(self._model)

        # Extract weight initializers as float32 torch tensors.
        # We copy the numpy array so that the tensor owns its memory and
        # is not a read-only view into the protobuf buffer.
        self._weights: dict[str, torch.Tensor] = {}
        for initializer in self._model.graph.initializer:
            np_array = onnx_mod.numpy_helper.to_array(initializer)
            self._weights[initializer.name] = torch.from_numpy(np_array.copy()).float()

        # Build the initial inference session from the file on disk.
        self._session = ort.InferenceSession(str(self._path))
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    # ------------------------------------------------------------------
    # Parameter accessors — compatible with FaultInjector interface
    # ------------------------------------------------------------------

    def named_parameters(self):
        """Yield ``(name, Parameter)`` pairs compatible with FaultInjector.

        Each tensor is wrapped as ``torch.nn.Parameter(requires_grad=True)``
        so that callers using the standard PyTorch parameter API work without
        modification.
        """
        for name, tensor in self._weights.items():
            yield name, torch.nn.Parameter(tensor, requires_grad=True)

    def parameters(self):
        """Yield parameter tensors (without names)."""
        for _, param in self.named_parameters():
            yield param

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a shallow copy of the weight tensors keyed by initializer name.

        The returned dict is independent of the internal state — mutating
        values inside it does not affect the model.
        """
        return {name: tensor.clone() for name, tensor in self._weights.items()}

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------

    def inject_faults(self, num_faults: int) -> int:
        """Inject bit-flip faults into the model weights.

        Faults are distributed across weight tensors proportionally to their
        element count.  After injection the ONNX InferenceSession is rebuilt
        so that subsequent ``__call__`` invocations use the faulted weights.

        Args:
            num_faults: Number of random bit flips to inject.  Values <= 0
                are treated as zero (no-op).

        Returns:
            Number of faults actually injected (int).
        """
        if num_faults <= 0:
            return 0

        all_weights = list(self._weights.items())
        total_elements = sum(t.numel() for _, t in all_weights)
        if total_elements == 0:
            return 0

        injected = 0
        for name, tensor in all_weights:
            remaining = num_faults - injected
            if remaining <= 0:
                break

            layer_faults = max(1, int(num_faults * tensor.numel() / total_elements))
            layer_faults = min(layer_faults, remaining)

            with torch.no_grad():
                FaultInjector.flip_random_bits(tensor, layer_faults)

            injected += layer_faults

        # Rebuild the ONNX session with the faulted weight values.
        self._rebuild_session()
        return injected

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def __call__(self, inputs: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Run inference on the (possibly faulted) model.

        Args:
            inputs: Float32 input tensor or numpy array with shape
                ``(batch_size, input_features)``.

        Returns:
            Model output as a float32 torch tensor.
        """
        if isinstance(inputs, torch.Tensor):
            np_inputs = inputs.detach().numpy().astype(np.float32)
        else:
            np_inputs = np.asarray(inputs, dtype=np.float32)

        feed = {self._input_names[0]: np_inputs}
        outputs = self._session.run(self._output_names, feed)
        return torch.from_numpy(outputs[0])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weight_count(self) -> int:
        """Total number of scalar weight elements across all initializers."""
        return sum(t.numel() for t in self._weights.values())

    @property
    def layer_names(self) -> list[str]:
        """Names of all weight initializers (in graph order)."""
        return list(self._weights.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_session(self) -> None:
        """Rebuild the ONNX InferenceSession from the current (faulted) weights.

        Creates a deep copy of the original protobuf model, overwrites each
        initializer with the current torch tensor values, serialises to bytes,
        and constructs a new InferenceSession from those bytes.  The original
        ``self._model`` protobuf is never mutated so that repeated calls are
        idempotent with respect to the reference graph.
        """
        onnx_mod = self._onnx

        model_copy = copy.deepcopy(self._model)
        for initializer in model_copy.graph.initializer:
            if initializer.name in self._weights:
                new_array = self._weights[initializer.name].detach().numpy()
                new_tensor = onnx_mod.numpy_helper.from_array(new_array, name=initializer.name)
                initializer.CopyFrom(new_tensor)

        model_bytes = model_copy.SerializeToString()
        self._session = self._ort.InferenceSession(model_bytes)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def load_onnx(path: str | Path) -> OnnxModel:
    """Load an ONNX model file and return an ``OnnxModel`` ready for fault injection.

    Args:
        path: Path to a ``.onnx`` file (string or :class:`pathlib.Path`).

    Returns:
        :class:`OnnxModel` instance wrapping the loaded model.

    Raises:
        ImportError: If ``onnx`` or ``onnxruntime`` are not installed.
        FileNotFoundError: If *path* does not point to an existing file.
    """
    return OnnxModel(path)
