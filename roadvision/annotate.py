"""Draw lane geometry and approach evidence on one frame.

Box colors follow the Phase 6 review video. The lane polygon
follows the integrated video. Labels describe lane membership
and image-space evidence. They are not collision warnings.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.association.vehicle_lane import (
    build_lane_polygon,
    get_bottom_center,
)


LANE_COLORS = {
    "IN_LANE": (0, 220, 255),
    "OUT_LANE": (120, 210, 120),
    "UNKNOWN": (180, 180, 180),
}

DISCLAIMER_TITLE = "RoadVision | Approach evidence"
DISCLAIMER_BODY = (
    "Experimental image-space indicators - not collision warnings"
)

_EVIDENCE_STATES = {"PENDING", "CONFIRMED", "HELD"}


@dataclass(frozen=True, slots=True)
class VehicleOverlay:
    """Fields required to draw one tracked object."""

    bbox: tuple[float, float, float, float]
    track_id: int
    class_name: str
    lane_status: str
    lane_reason: str
    approach_state: str
    event_state: str
    tau_proxy_s: float | None


def draw_lane(frame, left_fit, right_fit):
    """Draw the accepted ego-lane polygon on a copy of ``frame``."""

    annotated = frame.copy()

    if left_fit is None or right_fit is None:
        return annotated

    polygon = build_lane_polygon(left_fit, right_fit)

    if polygon is None:
        return annotated

    overlay = annotated.copy()
    cv2.fillPoly(
        overlay,
        [np.rint(polygon).astype(np.int32)],
        (0, 255, 0),
    )
    annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

    cv2.line(
        annotated,
        (int(left_fit[0]), int(left_fit[1])),
        (int(left_fit[2]), int(left_fit[3])),
        (0, 255, 0),
        5,
    )
    cv2.line(
        annotated,
        (int(right_fit[0]), int(right_fit[1])),
        (int(right_fit[2]), int(right_fit[3])),
        (255, 0, 0),
        5,
    )
    return annotated


def vehicle_lines(vehicle: VehicleOverlay) -> list[tuple[str, tuple[int, int, int]]]:
    """Return the text drawn for one object, top to bottom."""

    color = LANE_COLORS.get(vehicle.lane_status, LANE_COLORS["UNKNOWN"])
    lines = [
        (
            f"ID {vehicle.track_id} {vehicle.class_name} | {vehicle.lane_status}",
            color,
        )
    ]

    if vehicle.lane_status == "IN_LANE":
        lines.append((f"Scale: {vehicle.approach_state}", (255, 255, 255)))

        if vehicle.event_state in _EVIDENCE_STATES:
            evidence_color = (
                (0, 220, 255)
                if vehicle.event_state == "CONFIRMED"
                else (200, 200, 200)
            )
            lines.append((f"Evidence: {vehicle.event_state}", evidence_color))

        if (
            vehicle.event_state == "CONFIRMED"
            and vehicle.tau_proxy_s is not None
        ):
            lines.append(
                (
                    f"Inverse expansion: {vehicle.tau_proxy_s:.2f} s",
                    (255, 255, 255),
                )
            )

    elif vehicle.event_state == "HELD":
        lines.append(
            (
                "Previous approach evidence; lane uncertain",
                (190, 190, 190),
            )
        )

    elif vehicle.lane_status == "UNKNOWN" and vehicle.lane_reason:
        lines.append(
            (vehicle.lane_reason.replace("_", " "), color)
        )

    return lines


def annotate_frame(
    frame,
    left_fit,
    right_fit,
    vehicles: list[VehicleOverlay],
    frame_number: int,
    lane_available: bool,
):
    """Return a copy of ``frame`` with lane, objects, and the disclaimer."""

    annotated = draw_lane(frame, left_fit, right_fit)

    for vehicle in vehicles:
        _draw_vehicle(annotated, vehicle)

    _draw_banner(annotated, frame_number, lane_available)
    return annotated


def _draw_vehicle(frame, vehicle: VehicleOverlay) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in vehicle.bbox)

    if not np.isfinite([x1, y1, x2, y2]).all():
        return

    x1 = int(np.clip(round(x1), 0, width - 1))
    x2 = int(np.clip(round(x2), 0, width - 1))
    y1 = int(np.clip(round(y1), 0, height - 1))
    y2 = int(np.clip(round(y2), 0, height - 1))

    if x2 <= x1 or y2 <= y1:
        return

    color = LANE_COLORS.get(vehicle.lane_status, LANE_COLORS["UNKNOWN"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    contact_x, contact_y = get_bottom_center(vehicle.bbox)
    cv2.circle(
        frame,
        (
            int(np.clip(round(contact_x), 0, width - 1)),
            int(np.clip(round(contact_y), 0, height - 1)),
        ),
        4,
        color,
        -1,
    )

    text_y = y1 - 12

    for index, (text, text_color) in enumerate(vehicle_lines(vehicle)):
        _draw_text(frame, text, x1, text_y, text_color)
        text_y = y2 + 24 if index == 0 else text_y + 28


def _draw_banner(frame, frame_number: int, lane_available: bool) -> None:
    height, width = frame.shape[:2]
    bar_bottom = min(height, 96)
    cv2.rectangle(frame, (0, 0), (width, bar_bottom), (18, 18, 18), -1)

    _draw_text(frame, DISCLAIMER_TITLE, 16, 32, (255, 255, 255), scale=0.7)
    _draw_text(frame, DISCLAIMER_BODY, 16, 58, (180, 200, 255), scale=0.45)

    lane_text = "LANE: AVAILABLE" if lane_available else "LANE: UNAVAILABLE"
    lane_color = (0, 255, 0) if lane_available else (0, 0, 255)
    _draw_text(
        frame,
        f"{lane_text}  FRAME: {frame_number}",
        16,
        84,
        lane_color,
        scale=0.45,
    )


def _draw_text(
    image,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.5,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )
    height, width = image.shape[:2]
    x = int(np.clip(x, 0, max(0, width - text_width - 14)))
    y = int(np.clip(
        y,
        text_height + 8,
        max(text_height + 8, height - baseline - 4),
    ))

    cv2.rectangle(
        image,
        (x, y - text_height - 6),
        (min(width - 1, x + text_width + 10), min(height - 1, y + baseline + 4)),
        (18, 18, 18),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
