import numpy as np

from gravityhunter.dsp.fft_tools import dft_manual
from gravityhunter.dsp.correlation import (
    direct_correlation,
    fft_linear_correlation,
)
from gravityhunter.analysis.evaluation import confusion_counts


def test_manual_dft_matches_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.allclose(dft_manual(x), np.fft.fft(x), atol=1e-12)


def test_fft_correlation_matches_direct():
    x = np.array([0, 0, 1, 2, 1, 0, 0], dtype=float)
    s = np.array([1, 2, 1], dtype=float)

    l1, r1 = direct_correlation(x, s)
    l2, r2 = fft_linear_correlation(x, s)

    assert np.array_equal(l1, l2)
    assert np.allclose(r1, r2, atol=1e-12)


def test_confusion_metrics():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    pred = np.array([1, 0, 1, 0], dtype=bool)

    c = confusion_counts(truth, pred)
    assert (c.tp, c.fp, c.tn, c.fn) == (1, 1, 1, 1)
    assert np.isclose(c.sensitivity, 0.5)
    assert np.isclose(c.precision, 0.5)
