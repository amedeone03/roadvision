
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    """
    Simplified pinhole camera model.

    height_m:
        Camera height above a flat road, in meters.

    fy_px:
        Vertical focal length in pixels.

    cy_px:
        Vertical principal point in pixels.

    pitch_deg:
        Camera downward pitch angle in degrees.
        Positive pitch means the camera looks downward.
    """

    height_m: float
    fy_px: float
    cy_px: float
    pitch_deg: float = 0.0

    def __post_init__(self):
        values = (
            self.height_m,
            self.fy_px,
            self.cy_px,
            self.pitch_deg,
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Camera parameters must be finite")

        if self.height_m <= 0:
            raise ValueError("Camera height must be positive")

        if self.fy_px <= 0:
            raise ValueError("Focal length must be positive")

        if not -90 < self.pitch_deg < 90:
            raise ValueError("Camera pitch must be between -90 and 90 degrees")

    @property
    def horizon_y(self):
        """Image coordinate of the theoretical road horizon."""

        pitch = math.radians(self.pitch_deg)

        return self.cy_px - self.fy_px * math.tan(pitch)


def estimate_ground_distance(contact_y, camera):
    """
    Estimate forward distance from a road-contact point.

    Assumptions:
        - Flat road
        - Calibrated pinhole camera parameters
        - Negligible lens distortion
        - Contact point lies on the road
        - Camera pitch is measured relative to the road

    Returns:
        Forward distance in meters, or None if the
        ray does not intersect the road in front
        of the camera.

    This is a geometric estimate, not a validated
    distance measurement.
    """

    contact_y = float(contact_y)

    if not math.isfinite(contact_y):
        raise ValueError("Contact coordinate must be finite")

    # Angle below the camera's optical axis.
    image_angle = math.atan(
        (contact_y - camera.cy_px) / camera.fy_px
    )

    pitch = math.radians(camera.pitch_deg)

    # Total downward angle of the viewing ray.
    downward_angle = pitch + image_angle

    # No forward intersection with the ground.
    if downward_angle <= 0:
        return None

    # Valid forward rays must point below the
    # horizon but not backward.
    if downward_angle >= math.pi / 2:
        return None

    return (
        camera.height_m
        / math.tan(downward_angle)
    )


def project_ground_distance(distance_m, camera):
    """
    Project a known forward ground distance to
    its image Y coordinate.

    Used to test the inverse geometry.
    """

    distance_m = float(distance_m)

    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("Distance must be finite and positive")

    downward_angle = math.atan2(
        camera.height_m,
        distance_m,
    )

    pitch = math.radians(camera.pitch_deg)

    image_angle = downward_angle - pitch

    if abs(image_angle) >= math.pi / 2:
        return None

    return (
        camera.cy_px
        + camera.fy_px * math.tan(image_angle)
    )
