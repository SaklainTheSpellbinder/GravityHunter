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


def test_manual_welch_matches_scipy_density_scaling():
    from gravityhunter.dsp.psd import manual_welch_psd, welch_psd

    rng = np.random.default_rng(42)
    x = rng.normal(size=4096) + 0.3 * np.sin(2 * np.pi * 73 * np.arange(4096) / 1024)
    f1, p1 = manual_welch_psd(x, 1024.0, nperseg=512, noverlap=256, window="hann")
    f2, p2 = welch_psd(x, 1024.0, nperseg=512, noverlap=256, window="hann", average="mean")
    assert np.allclose(f1, f2)
    assert np.allclose(p1, p2, rtol=2e-12, atol=1e-15)


def test_whitening_returns_finite_standardized_signal():
    from gravityhunter.dsp.psd import welch_psd
    from gravityhunter.dsp.whitening import whiten

    rng = np.random.default_rng(123)
    x = rng.normal(size=8192)
    f, p = welch_psd(x, 2048.0, nperseg=1024, noverlap=512)
    xw, fw, Xw, pi = whiten(x, 2048.0, f, p, fmin=30, fmax=700, standardize=True)
    assert xw.shape == x.shape
    assert fw.shape == Xw.shape == pi.shape
    assert np.all(np.isfinite(xw))
    assert abs(float(np.mean(xw))) < 1e-12
    assert abs(float(np.std(xw)) - 1.0) < 1e-12


def test_two_basis_snr_accounts_for_nonorthogonal_templates():
    from gravityhunter.dsp.matched_filter import combine_two_basis_snr

    a = np.array([3.0])
    b = np.array([3.0])
    # If two basis vectors are strongly correlated, blindly taking hypot would
    # overstate the independent evidence. The Gram-corrected result is smaller.
    corrected = combine_two_basis_snr(a, b, 0.8)[0]
    naive = np.hypot(a[0], b[0])
    assert corrected < naive
    assert np.isclose(corrected, np.sqrt((9 - 2 * 0.8 * 9 + 9) / (1 - 0.8**2)))


def test_matched_filter_peak_matches_template_norm_for_exact_injection():
    from gravityhunter.dsp.matched_filter import matched_filter_snr, expected_template_snr

    fs = 1024.0
    n = 4096
    m = 400
    start = 1234
    tt = np.arange(m) / fs
    template = np.hanning(m) * np.sin(2 * np.pi * (50.0 * tt + 80.0 * tt**2))
    data = np.zeros(n)
    data[start:start + m] = template

    psd_freq = np.linspace(0, fs / 2, 513)
    psd = np.full_like(psd_freq, 1e-3)
    lags, rho, _ = matched_filter_snr(
        data, template, fs, psd_freq, psd, fmin=20.0, fmax=400.0
    )
    i = int(np.argmax(rho))
    sigma = expected_template_snr(
        template,
        fs,
        psd_freq,
        psd,
        nfft=n + m - 1,
        fmin=20.0,
        fmax=400.0,
    )
    assert lags[i] == start
    assert np.isclose(rho[i], sigma, rtol=1e-12, atol=1e-12)
