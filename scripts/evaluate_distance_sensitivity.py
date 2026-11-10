
import pandas as pd

from src.motion.distance_estimator import (
    CameraModel,
    estimate_ground_distance,
    project_ground_distance,
)


# Hypothetical reference camera.
# These are NOT calibrated RoadVision parameters.
REFERENCE = CameraModel(
    height_m=1.5,
    fy_px=1000.0,
    cy_px=540.0,
    pitch_deg=3.0,
)

TRUE_DISTANCES = [10, 20, 30, 50, 80]

# Explore parameter uncertainty independently.
SCENARIOS = {
    "reference": REFERENCE,

    "height_1.3m": CameraModel(
        height_m=1.3,
        fy_px=1000,
        cy_px=540,
        pitch_deg=3,
    ),

    "height_1.7m": CameraModel(
        height_m=1.7,
        fy_px=1000,
        cy_px=540,
        pitch_deg=3,
    ),

    "pitch_2deg": CameraModel(
        height_m=1.5,
        fy_px=1000,
        cy_px=540,
        pitch_deg=2,
    ),

    "pitch_4deg": CameraModel(
        height_m=1.5,
        fy_px=1000,
        cy_px=540,
        pitch_deg=4,
    ),

    "focal_900px": CameraModel(
        height_m=1.5,
        fy_px=900,
        cy_px=540,
        pitch_deg=3,
    ),

    "focal_1100px": CameraModel(
        height_m=1.5,
        fy_px=1100,
        cy_px=540,
        pitch_deg=3,
    ),
}


rows = []

for true_distance in TRUE_DISTANCES:

    # Generate an image coordinate using the
    # hypothetical reference camera.
    contact_y = project_ground_distance(
        true_distance,
        REFERENCE,
    )

    for scenario_name, camera in SCENARIOS.items():

        estimated = estimate_ground_distance(
            contact_y,
            camera,
        )

        rows.append({
            "true_distance_m": true_distance,
            "scenario": scenario_name,
            "contact_y": contact_y,
            "estimated_distance_m": estimated,
            "error_m": (
                estimated - true_distance
                if estimated is not None
                else float("nan")
            ),
        })


results = pd.DataFrame(rows)

print("\n--- MONOCULAR DISTANCE SENSITIVITY ---")

comparison = results.pivot(
    index="true_distance_m",
    columns="scenario",
    values="estimated_distance_m",
)

print(comparison.round(2).to_string())

print("\n--- ABSOLUTE DISTANCE ERROR ---")

errors = results.groupby("scenario")["error_m"].apply(
    lambda x: x.abs().mean()
)

print(errors.round(2).to_string())

print("\nSensitivity analysis complete!")
