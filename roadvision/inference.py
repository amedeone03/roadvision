"""Optional ONNX Runtime Core ML session for one video run.

Ultralytics 8.4.193 builds the ONNX session inside ``ONNXBackend.load_model``.
That method accepts no execution-provider argument. Its MPS branch requests
the NeuralNetwork format, which is not the MLProgram session measured on
the reference video. ``AutoBackend`` also constructs that class with only
``device`` and ``fp16``.

The workaround replaces ``onnxruntime.InferenceSession`` on the imported
module. Ultralytics looks that name up on the calling thread when the
first ``track()`` creates the session, so the replacement is visible.
It is process-wide module state, not an instance setting. It stays in
place from detector construction through that first ``track()`` call,
then the original constructor is written back, including when
construction or the first call raises.

Two overlapping Core ML initializations would save and restore the same
attribute. That race is rejected. Simultaneous video jobs are not
supported. Sequential runs are independent: each one builds a new
Ultralytics model, and ByteTrack state lives on that model.
"""

from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path

_INIT_LOCK = threading.Lock()

COREML_PROVIDER_OPTIONS = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",
    "RequireStaticInputShapes": "1",
    "EnableOnSubgraphs": "1",
    "SpecializationStrategy": "FastPrediction",
    "ProfileComputePlan": "0",
    "AllowLowPrecisionAccumulationOnGPU": "0",
}
_PROVIDER_OPTIONS = {
    "ModelFormat": ("MLProgram", "NeuralNetwork"),
    "MLComputeUnits": ("ALL", "CPUAndGPU", "CPUOnly", "CPUAndNeuralEngine"),
    "RequireStaticInputShapes": ("0", "1"),
    "EnableOnSubgraphs": ("0", "1"),
    "SpecializationStrategy": ("Default", "FastPrediction"),
    "ProfileComputePlan": ("0", "1"),
    "AllowLowPrecisionAccumulationOnGPU": ("0", "1"),
}


def coreml_providers() -> list:
    """Return the verified provider list, CPU provider last."""

    problems = invalid_provider_options(COREML_PROVIDER_OPTIONS)
    if problems:
        from roadvision.pipeline import PipelineError

        raise PipelineError(
            "The Core ML provider options are not valid for this "
            f"runtime: {', '.join(problems)}."
        )
    return [
        ("CoreMLExecutionProvider", dict(COREML_PROVIDER_OPTIONS)),
        "CPUExecutionProvider",
    ]


def invalid_provider_options(options: dict) -> list[str]:
    """Return option names this ONNX Runtime build does not accept."""

    problems = []
    for name, value in options.items():
        allowed = _PROVIDER_OPTIONS.get(name)
        if allowed is None or str(value) not in allowed:
            problems.append(name)
    return problems


def require_coreml_runtime() -> None:
    """Reject a Core ML run this machine cannot execute.

    PyTorch remains available. This function does not switch to it.
    """

    from roadvision.pipeline import PipelineError

    if sys.platform != "darwin":
        raise PipelineError(
            "backend 'onnx-coreml' requires macOS. "
            f"This process is on {sys.platform!r}. "
            "RoadVision does not fall back to PyTorch."
        )

    try:
        onnxruntime = _import_onnxruntime()
    except ImportError as exc:
        raise PipelineError(
            "backend 'onnx-coreml' requires onnxruntime with "
            "CoreMLExecutionProvider. It is not installed. "
            "Use the optional .venv-onnx environment. "
            "RoadVision does not fall back to PyTorch."
        ) from exc

    available = list(onnxruntime.get_available_providers())
    if "CoreMLExecutionProvider" not in available:
        raise PipelineError(
            "CoreMLExecutionProvider is not available. "
            f"Installed providers: {available}. "
            "RoadVision does not fall back to PyTorch."
        )


def validate_onnx_model(path: Path) -> None:
    """Reject an ONNX file that cannot use the verified static export.

    The check loads the graph with the CPU provider. It does not run
    Core ML and does not treat a registered provider as proof that
    every node will execute there.
    """

    from roadvision.pipeline import PipelineError

    onnxruntime = _import_onnxruntime()
    try:
        session = onnxruntime.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise PipelineError(
            f"Cannot load ONNX model {path}: {exc}. "
            "Export yolo11n.pt with imgsz=640, dynamic=False, nms=False, "
            "and FP32. RoadVision does not fall back to PyTorch."
        ) from exc

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_shape = _static_shape(inputs[0].shape) if len(inputs) == 1 else None
    output_shape = _static_shape(outputs[0].shape) if len(outputs) == 1 else None
    embedded_nms = (
        output_shape is not None
        and len(output_shape) == 3
        and output_shape[-1] == 6
    )
    if input_shape != [1, 3, 640, 640] or output_shape is None or embedded_nms:
        raise PipelineError(
            "The ONNX model does not match the verified static YOLO11n "
            f"export. Input shape: {getattr(inputs[0], 'shape', None) if inputs else None}. "
            f"Output shapes: {[getattr(item, 'shape', None) for item in outputs]}. "
            "Required input is [1, 3, 640, 640] and NMS must stay outside "
            "the graph. RoadVision does not fall back to PyTorch."
        )


def _static_shape(shape) -> list[int] | None:
    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            return None
        dims.append(dim)
    return dims


def model_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_session(model: object, model_path: Path) -> dict:
    """Read the providers the loaded session actually selected."""

    session = _session(model)
    if session is None:
        from roadvision.pipeline import PipelineError

        raise PipelineError(
            "backend 'onnx-coreml' did not create an ONNX Runtime session. "
            "RoadVision does not fall back to PyTorch."
        )
    providers = list(session.get_providers())
    if not providers or providers[0] != "CoreMLExecutionProvider":
        from roadvision.pipeline import PipelineError

        raise PipelineError(
            "backend 'onnx-coreml' did not select CoreMLExecutionProvider. "
            f"Active providers: {providers}. "
            "RoadVision does not fall back to PyTorch."
        )
    options = dict(session.get_provider_options().get("CoreMLExecutionProvider", {}))
    if options.get("ModelFormat") != "MLProgram" or options.get("MLComputeUnits") != "ALL":
        from roadvision.pipeline import PipelineError

        raise PipelineError(
            "The Core ML session options do not match the verified "
            f"configuration. Effective options: {options}."
        )
    onnxruntime = _import_onnxruntime()
    return {
        "name": "onnx-coreml",
        "experimental": True,
        "model_path": str(model_path),
        "model_sha256": model_sha256(model_path),
        "onnxruntime": onnxruntime.__version__,
        "execution_providers": providers,
        "provider_options": options,
        "node_assignment": (
            "This run verified the selected providers and options. "
            "It did not profile individual graph nodes. The benchmarked "
            "Apple M4 environment assigned 320 of 320 nodes to one "
            "Core ML partition."
        ),
    }


def _import_onnxruntime():
    import onnxruntime

    return onnxruntime


def _session(model: object):
    candidates = []
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        candidates.append(getattr(predictor, "model", None))
    candidates.append(getattr(model, "model", None))
    for backend in candidates:
        session = getattr(backend, "session", None)
        if session is None:
            session = getattr(getattr(backend, "model", None), "session", None)
        if session is not None and hasattr(session, "get_providers"):
            return session
    return None


class SessionProviders:
    """Replace ``onnxruntime.InferenceSession`` until this block exits.

    The replacement is global module state. A second overlapping block
    fails instead of nesting, because the two blocks would otherwise
    restore the attribute in the wrong order.
    """

    def __init__(self, providers: list) -> None:
        self.providers = list(providers)
        self._original = None
        self._holds_lock = False

    def __enter__(self):
        from roadvision.pipeline import PipelineError

        if not _INIT_LOCK.acquire(blocking=False):
            raise PipelineError(
                "backend 'onnx-coreml' does not support concurrent "
                "initialization. Another video run is still creating "
                "its Core ML session. RoadVision does not fall back "
                "to PyTorch."
            )
        self._holds_lock = True
        try:
            onnxruntime = _import_onnxruntime()
            self._original = onnxruntime.InferenceSession
            forced = self.providers
            original = self._original

            def session(model, sess_options=None, providers=None, *args, **kwargs):
                return original(model, sess_options, providers=forced, *args, **kwargs)

            onnxruntime.InferenceSession = session
            return self
        except Exception:
            self._release()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._original is not None:
                onnxruntime = _import_onnxruntime()
                onnxruntime.InferenceSession = self._original
                self._original = None
        finally:
            self._release()

    def _release(self) -> None:
        if not self._holds_lock:
            return
        self._holds_lock = False
        _INIT_LOCK.release()
