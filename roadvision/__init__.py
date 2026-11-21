"""RoadVision application package.

Algorithms stay in ``src/``. This package is the Phase 8
integration layer.
"""

from roadvision.config import (
    ConfigError,
    RoadVisionConfig,
    load_config,
    validate_runtime_paths,
)
from roadvision.pipeline import (
    PipelineError,
    PipelineResult,
    run_video,
)
from roadvision.tracks import (
    ApproachEpisode,
    MotionEstimate,
    TrackMotionProcessor,
)

__all__ = [
    "ApproachEpisode",
    "ConfigError",
    "MotionEstimate",
    "PipelineError",
    "PipelineResult",
    "RoadVisionConfig",
    "TrackMotionProcessor",
    "load_config",
    "run_video",
    "validate_runtime_paths",
]
