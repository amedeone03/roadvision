
import numpy as np

from src.motion.robust_growth import (
    theil_sen_slope,
    rolling_robust_growth,
)


def main():

    print("\n--- ROBUST GROWTH UNIT TESTS ---")

    # ------------------------------------------
    # Test 1: Exact linear trend
    # ------------------------------------------

    times = np.arange(5) * 0.1

    true_growth = 0.4

    log_heights = 4.5 + true_growth * times

    result = theil_sen_slope(
        times,
        log_heights,
    )

    assert np.isclose(result, true_growth)

    print("Test 1 passed: exact linear trend")

    # ------------------------------------------
    # Test 2: Single large outlier
    # ------------------------------------------

    noisy = log_heights.copy()

    noisy[2] += 1.0

    robust = theil_sen_slope(
        times,
        noisy,
    )

    ordinary = np.polyfit(
        times,
        noisy,
        1,
    )[0]

    assert np.isclose(robust, true_growth)

    # A centered outlier may not change an
    # ordinary least-squares slope on symmetric
    # timestamps, so test robustness directly.
    assert np.isfinite(ordinary)

    print("Test 2 passed: single outlier")

    # ------------------------------------------
    # Test 3: Rolling growth
    # ------------------------------------------

    heights = np.exp(log_heights)

    estimates = rolling_robust_growth(
        times,
        heights,
        window=5,
    )

    assert len(estimates) == 5

    assert np.isnan(estimates[:4]).all()

    assert np.isclose(
        estimates[4],
        true_growth,
    )

    print("Test 3 passed: causal rolling window")

    # ------------------------------------------
    # Test 4: Invalid heights
    # ------------------------------------------

    try:
        rolling_robust_growth(
            times,
            [100, 110, 0, 130, 140],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Non-positive height was accepted"
        )

    print("Test 4 passed: invalid height rejection")

    # ------------------------------------------
    # Test 5: Non-increasing timestamps
    # ------------------------------------------

    try:
        theil_sen_slope(
            [0.0, 0.1, 0.1],
            [1.0, 2.0, 3.0],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Duplicate timestamps were accepted"
        )

    print("Test 5 passed: timestamp validation")

    print("\nAll robust growth tests passed!")


if __name__ == "__main__":
    main()

def test_robust_growth():
    main()