"""The median_and_percentiles_over_ensemble time-axis contract.

The function slices `arrays[-delta_t:]`: time runs along `time_axis`
(default 0). The manuscript notebooks pass [T, ensemble] (default is
correct for them). The state-vs-defect CLI historically passed
[ensemble, T] without transposing, silently computing statistics over the
last delta_t ensemble members including all transients — the committed
results_*.h5 files were generated that way (regeneration gated in Phase 7).
The CLI now passes time_axis=1.
"""

import numpy as np
import pytest

from llna.analysis import median_and_percentiles_over_ensemble


@pytest.fixture()
def converged_ensemble():
    """40 wildly varying transients that all settle to exactly 0.7."""
    rng = np.random.default_rng(0)
    arrays = rng.random((40, 30))  # [ensemble, T]
    arrays[:, -5:] = 0.7
    return arrays


def test_time_first_input_finds_convergence_value(converged_ensemble):
    median, q1, q3 = median_and_percentiles_over_ensemble(converged_ensemble.T, delta_t=5)
    assert median == q1 == q3 == 0.7


def test_ensemble_first_input_with_explicit_time_axis(converged_ensemble):
    median, q1, q3 = median_and_percentiles_over_ensemble(converged_ensemble, delta_t=5, time_axis=1)
    assert median == q1 == q3 == 0.7


def test_time_axis_must_be_0_or_1(converged_ensemble):
    with pytest.raises(ValueError):
        median_and_percentiles_over_ensemble(converged_ensemble, delta_t=5, time_axis=2)


def test_rejects_wrong_dimensionality():
    with pytest.raises(ValueError):
        median_and_percentiles_over_ensemble(np.zeros(10), delta_t=2)
    with pytest.raises(ValueError):
        median_and_percentiles_over_ensemble(np.zeros((2, 3, 4)), delta_t=2)


def test_rejects_percentiles_not_bracketing_median():
    with pytest.raises(ValueError):
        median_and_percentiles_over_ensemble(np.zeros((5, 5)), delta_t=2, lower_percentile=0.6)
