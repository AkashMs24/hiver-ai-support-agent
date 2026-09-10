"""
Calculates Cohen's Kappa to measure inter-annotator agreement between:
- Human golden annotations
- LLM Judge ratings
Validates whether the automated evaluation judge can actually be trusted.
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import cohen_kappa_score


def compute_calibration_agreement(
    human_scores: List[int],
    judge_scores: List[int],
) -> Dict[str, Any]:
    """
    Compute unweighted and quadratic weighted Cohen's Kappa for ordinal ratings.
    """
    if len(human_scores) != len(judge_scores):
        raise ValueError("Human and judge score lists must be equal length.")

    # Calculate exact match agreement percentage
    exact_match = np.mean(np.array(human_scores) == np.array(judge_scores))

    # Calculate within-1-point agreement (standard tolerance for 5-point Likert scales)
    within_one = np.mean(np.abs(np.array(human_scores) - np.array(judge_scores)) <= 1)

    # Standard Cohen's Kappa
    try:
        kappa_unweighted = cohen_kappa_score(human_scores, judge_scores)
    except Exception:
        kappa_unweighted = 0.0

    # Quadratic weighted Cohen's Kappa (penalizes large distance discrepancies more)
    try:
        kappa_weighted = cohen_kappa_score(human_scores, judge_scores, weights="quadratic")
    except Exception:
        kappa_weighted = 0.0

    # Qualitative interpretation
    interpretation = "Slight"
    if kappa_weighted >= 0.81:
        interpretation = "Almost Perfect Agreement"
    elif kappa_weighted >= 0.61:
        interpretation = "Substantial Agreement"
    elif kappa_weighted >= 0.41:
        interpretation = "Moderate Agreement"
    elif kappa_weighted >= 0.21:
        interpretation = "Fair Agreement"

    return {
        "num_pairs": len(human_scores),
        "exact_match_rate": round(float(exact_match), 3),
        "within_1_point_rate": round(float(within_one), 3),
        "cohen_kappa_unweighted": round(float(kappa_unweighted), 3),
        "cohen_kappa_quadratic": round(float(kappa_weighted), 3),
        "interpretation": interpretation,
    }
