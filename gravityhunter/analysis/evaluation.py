from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def sensitivity(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom else float("nan")

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else float("nan")


def confusion_counts(
    truth_event: np.ndarray,
    predicted_event: np.ndarray,
) -> ConfusionCounts:
    truth = np.asarray(truth_event, dtype=bool)
    pred = np.asarray(predicted_event, dtype=bool)

    if truth.shape != pred.shape:
        raise ValueError("truth and predicted arrays must match.")

    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    tn = int(np.sum(~truth & ~pred))
    fn = int(np.sum(truth & ~pred))

    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


def roc_points(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (FPR, TPR) at each threshold.
    """
    positive_scores = np.asarray(positive_scores, dtype=float)
    negative_scores = np.asarray(negative_scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    tpr = np.empty(thresholds.size, dtype=float)
    fpr = np.empty(thresholds.size, dtype=float)

    for i, th in enumerate(thresholds):
        tpr[i] = np.mean(positive_scores >= th) if positive_scores.size else np.nan
        fpr[i] = np.mean(negative_scores >= th) if negative_scores.size else np.nan

    return fpr, tpr


def detection_rate_by_strength(
    strengths: np.ndarray,
    detected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate repeated injection trials into P(detect | strength).
    """
    strengths = np.asarray(strengths, dtype=float)
    detected = np.asarray(detected, dtype=bool)

    if strengths.shape != detected.shape:
        raise ValueError("strengths and detected must match.")

    unique = np.unique(strengths)
    rates = np.array([np.mean(detected[strengths == a]) for a in unique])
    return unique, rates
