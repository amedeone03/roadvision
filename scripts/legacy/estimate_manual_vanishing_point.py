
from pathlib import Path

import cv2
import numpy as np


VIDEO_PATH = Path("data/videos/test_drive.mp4")
OUTPUT_DIR = Path("runs/motion/vanishing_point")

FRAMES = [
    250,
    450,
    650,
    850,
    1200,
    1450,
]

DISPLAY_WIDTH = 1200

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def line_intersection(p1, p2, p3, p4):
    """Intersection of two infinite image lines."""

    line1 = np.cross(
        [p1[0], p1[1], 1.0],
        [p2[0], p2[1], 1.0],
    )

    line2 = np.cross(
        [p3[0], p3[1], 1.0],
        [p4[0], p4[1], 1.0],
    )

    intersection = np.cross(line1, line2)

    if abs(intersection[2]) < 1e-8:
        return None

    return (
        float(intersection[0] / intersection[2]),
        float(intersection[1] / intersection[2]),
    )


def load_frame(frame_number):

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    try:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1,
        )

        success, frame = cap.read()
    finally:
        cap.release()

    if not success:
        raise RuntimeError(
            f"Cannot read frame {frame_number}"
        )

    return frame


def annotate_frame(frame, points, scale):

    image = cv2.resize(
        frame,
        None,
        fx=scale,
        fy=scale,
    )

    # Draw clicked points and selected markings.
    display_points = []

    for i, (x, y) in enumerate(points):

        px = int(round(x * scale))
        py = int(round(y * scale))

        display_points.append((px, py))

        cv2.circle(
            image,
            (px, py),
            6,
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            image,
            str(i + 1),
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    if len(display_points) >= 2:
        cv2.line(
            image,
            display_points[0],
            display_points[1],
            (0, 255, 0),
            2,
        )

    if len(display_points) >= 4:
        cv2.line(
            image,
            display_points[2],
            display_points[3],
            (255, 0, 0),
            2,
        )

    # Display instructions.
    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (image.shape[1], 82),
        (0, 0, 0),
        -1,
    )

    image = cv2.addWeighted(
        overlay,
        0.7,
        image,
        0.3,
        0,
    )

    cv2.putText(
        image,
        "Click 2 points on LEFT marking, then 2 on RIGHT marking",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"Points: {len(points)}/4  |  S = Save  R = Reset  Q/ESC = Quit",
        (18, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return image


def inspect_frame(frame_number):

    frame = load_frame(frame_number)

    original_height, original_width = frame.shape[:2]

    scale = DISPLAY_WIDTH / original_width

    points = []

    window_name = f"RoadVision VP - Frame {frame_number}"

    def on_mouse(event, x, y, flags, userdata):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(points) >= 4:
            return

        # Convert displayed coordinates back to
        # original full-resolution coordinates.
        original_x = x / scale
        original_y = y / scale

        points.append((original_x, original_y))

        print(
            f"Point {len(points)}: "
            f"({original_x:.1f}, {original_y:.1f})"
        )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE,
    )

    cv2.setMouseCallback(
        window_name,
        on_mouse,
    )

    print(f"\n--- FRAME {frame_number} ---")
    print("Click two points on the left road marking.")
    print("Then click two on the right road marking.")
    print("Press S in the image window to calculate.")
    print("Press R to reset, Q or ESC to quit.")

    try:
        while True:

            display = annotate_frame(
                frame,
                points,
                scale,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("r"):
                points.clear()
                print("Selection reset.")

            elif key in (ord("q"), 27):
                print("Frame skipped.")
                return

            elif key == ord("s"):

                if len(points) != 4:
                    print("Select four points first.")
                    continue

                vp = line_intersection(*points)

                if vp is None:
                    print("Lines are parallel or nearly parallel.")
                    continue

                vx, vy = vp

                print(
                    f"Vanishing point: "
                    f"x={vx:.2f}, y={vy:.2f}"
                )

                result = frame.copy()

                colors = [
                    (0, 255, 0),
                    (255, 0, 0),
                ]

                for i in (0, 1):

                    p1 = tuple(
                        int(round(v))
                        for v in points[2 * i]
                    )

                    p2 = tuple(
                        int(round(v))
                        for v in points[2 * i + 1]
                    )

                    cv2.line(
                        result,
                        p1,
                        p2,
                        colors[i],
                        3,
                    )

                # Draw VP row if it falls inside the image.
                if 0 <= vy < original_height:

                    cv2.line(
                        result,
                        (0, int(round(vy))),
                        (
                            original_width - 1,
                            int(round(vy)),
                        ),
                        (0, 165, 255),
                        2,
                    )

                if (
                    0 <= vx < original_width
                    and 0 <= vy < original_height
                ):

                    cv2.circle(
                        result,
                        (
                            int(round(vx)),
                            int(round(vy)),
                        ),
                        9,
                        (0, 165, 255),
                        -1,
                    )

                output_path = (
                    OUTPUT_DIR
                    / f"frame_{frame_number}_vp.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    result,
                )

                print(f"Saved: {output_path}")

                return

    finally:
        cv2.destroyWindow(window_name)


def main():

    for frame_number in FRAMES:
        inspect_frame(frame_number)

    print("\nVanishing-point inspection complete!")


if __name__ == "__main__":
    main()
