"""Statistical primitives for the paired perception-vs-reasoning analysis.

Stdlib-only (no scipy): the exact McNemar test uses `math.comb`, Wilson intervals
and the power formula use a self-contained inverse-normal (probit), and the recovery
CI uses a paired bootstrap. Everything here is a pure function of its inputs so the
analyzer stays deterministic and testable.

The oracle design is PAIRED - the same scene is answered under `full` (image) and
`oracle` (ground-truth text) conditions - so the comparison of interest is a paired
one. Reporting two independent per-condition CIs and eyeballing overlap is the wrong
test (it ignores the correlation between conditions and is under-powered); McNemar's
test on the discordant pairs is the correct one.
"""
from __future__ import annotations

import math
import random as _random

# Inverse normal CDF (probit) - Acklam's rational approximation.
# Max relative error ~1.15e-9 over p in (0, 1); ample for CIs and power.

_A = (
    -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
    1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
)
_B = (
    -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
    6.680131188771972e01, -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
    -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
)
_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF. `p` must be in the open interval (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    )


# Wilson score interval for a binomial proportion.

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """95%-by-default Wilson score interval for k successes in n trials.

    Preferred over the normal (Wald) interval, which misbehaves near 0/1 and at
    small n - exactly the regime free-tier runs live in. Returns None for n == 0.
    """
    if n == 0:
        return None
    z = norm_ppf(1.0 - alpha / 2.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return [round(lo, 4), round(hi, 4)]


# McNemar's exact test on discordant pairs.

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    `b` = # pairs correct under full but wrong under oracle; `c` = the reverse.
    Under H0 (no systematic difference) each discordant pair is an independent
    fair coin, so the p-value is the two-sided exact binomial tail with p=0.5.
    Concordant pairs (both right / both wrong) carry no information and are excluded.
    Returns 1.0 when there are no discordant pairs (no evidence either way).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


# Paired bootstrap CI on recovery = acc_oracle - acc_full.

def paired_bootstrap_recovery(
    pairs: list[tuple[bool, bool]],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[float] | None:
    """Percentile bootstrap CI for recovery, resampling SCENES (paired rows).

    Each element of `pairs` is (full_correct, oracle_correct) for one scene.
    Resampling whole pairs (not the 2n individual observations) preserves the
    within-scene correlation between conditions, which is the entire point of a
    paired design. Returns None for an empty input.
    """
    n = len(pairs)
    if n == 0:
        return None
    fulls = [1 if f else 0 for f, _ in pairs]
    oracles = [1 if o else 0 for _, o in pairs]
    rng = _random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        rf = sum(fulls[i] for i in idx) / n
        ro = sum(oracles[i] for i in idx) / n
        boots.append(ro - rf)
    boots.sort()
    lo = boots[int((alpha / 2.0) * n_boot)]
    hi = boots[int((1.0 - alpha / 2.0) * n_boot)]
    return [round(lo, 4), round(hi, 4)]


# Sample-size / power for the paired McNemar test.

def mcnemar_power_n(
    effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    discordance: float = 0.30,
) -> int:
    """Paired-n needed to detect a recovery of `effect` at `power` and `alpha`.

    Uses the standard normal-approximation McNemar sample-size formula:

        n = ( z_{1-α/2}·√p_d + z_{power}·√(p_d − δ²) )² / δ²

    where δ = |effect| is the target difference in marginal accuracy and p_d is
    the assumed proportion of DISCORDANT pairs (the nuisance parameter you cannot
    know a priori - smaller p_d needs larger n). δ ≤ p_d is required, since the
    marginal difference cannot exceed the discordant rate.
    """
    delta = abs(effect)
    if delta <= 0.0:
        raise ValueError("effect must be non-zero")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if not 0.0 < discordance <= 1.0:
        raise ValueError("discordance must be in (0, 1]")
    if discordance < delta:
        raise ValueError(
            f"discordance ({discordance}) must be >= |effect| ({delta}): the "
            "discordant-pair rate bounds the achievable marginal difference."
        )
    z_a = norm_ppf(1.0 - alpha / 2.0)
    z_b = norm_ppf(power)
    num = z_a * math.sqrt(discordance) + z_b * math.sqrt(discordance - delta * delta)
    return math.ceil((num * num) / (delta * delta))


# Stability of a categorical verdict under resampling.

def _replicate_accuracies(
    per_scene: list[dict[str, tuple[int, int]]],
    idx,
) -> dict[str, float | None]:
    """Pooled per-condition accuracy over the scenes named by `idx`."""
    totals: dict[str, list[int]] = {}
    for i in idx:
        for cond, (k, n) in per_scene[i].items():
            cell = totals.setdefault(cond, [0, 0])
            cell[0] += k
            cell[1] += n
    return {c: (k / n if n else None) for c, (k, n) in totals.items()}


def bootstrap_label_stability(
    per_scene: list[dict[str, tuple[int, int]]],
    label_of,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict | None:
    """How often a categorical verdict survives resampling the scenes.

    `per_scene` holds one dict per scene mapping condition -> (n_correct, n_rows)
    for that scene; a condition with no valid rows for a scene is simply absent.
    `label_of` receives condition -> accuracy (None when a condition had no rows
    in that replicate) and returns the verdict for it.

    Resampling whole scenes preserves the within-scene correlation across
    conditions, for the same reason `paired_bootstrap_recovery` does. A verdict
    read off thresholds is a step function of the accuracies, so a point estimate
    near a threshold can flip on one item; this reports how often it does.

    Returns {"label", "stability", "distribution"}, or None for empty input.
    """
    n = len(per_scene)
    if n == 0:
        return None
    point = label_of(_replicate_accuracies(per_scene, range(n)))
    rng = _random.Random(seed)
    counts: dict[str, int] = {}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        lbl = label_of(_replicate_accuracies(per_scene, idx))
        counts[lbl] = counts.get(lbl, 0) + 1
    dist = {
        k: round(v / n_boot, 4)
        for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    }
    return {"label": point, "stability": dist.get(point, 0.0), "distribution": dist}
