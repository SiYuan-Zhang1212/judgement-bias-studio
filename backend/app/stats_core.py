"""Minimal inferential statistics kernel for the Judgement Bias SOP.

Why this module exists
----------------------
In this paradigm every anchor tone maps to a *fixed* correct key, so "high vs low
tone" and "right vs left hand" are near-collinear on anchor trials (measured
agreement 124/126 for 001, 126/126 for 002).  Any condition contrast therefore
has to be reported under three design flavours side by side, never one alone:

    A  anchors only          y ~ pitch_lin                   (legacy-comparable, contains hand)
    B  all main trials       y ~ pitch_lin + hand + block    (hand in model; collinear -> lower bound)
    C  ambiguous tones only  y ~ pitch_lin + hand            (hand has variance; cleanest, low power)

Reporting only A or only B is selective reporting.

Conventions fixed SOP-wide (do NOT vary them per analysis):
* Permutation p-values use ``(k + 1) / (n_perm + 1)`` -- never ``k / n_perm``:
  with 1000 permutations that floor is 1/1001 and no p can be reported as 0.
* Labels are shuffled **within block**.  Blocks carry real nuisance structure
  (fatigue, adaptation, reward accumulation); a global shuffle inflates the null
  variance and silently makes real effects look significant.
* Family correction is ``max|t|`` over (channel x window) *inside one arm*, so
  the p-value belongs to the family and never to a hand-picked peak channel.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


__all__ = [
    "design",
    "ols_t",
    "t_from_corr",
    "vif",
    "condition_number",
    "perm_p_abs",
    "permute_within_blocks",
    "corr_r",
    "thresholds",
]


def design(*cols: Sequence[float] | np.ndarray, intercept: bool = True) -> np.ndarray:
    """Stack regressor columns into a design matrix, intercept first.

    Each argument may be 1-D (n,) or already 2-D (n, k).  A 1-D argument is one
    regressor; this keeps call sites readable when most arms have a mix.

    >>> design([1.0, 2.0], [0.0, 1.0]).shape
    (2, 3)
    """
    if not cols:
        raise ValueError("design() needs at least one column")

    blocks: list[np.ndarray] = []
    n: int | None = None
    for c in cols:
        a = np.asarray(c, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        elif a.ndim != 2:
            raise ValueError("design columns must be 1-D or 2-D")
        if n is None:
            n = a.shape[0]
        elif a.shape[0] != n:
            raise ValueError("all design columns must have the same number of rows")
        blocks.append(a)

    mats = ([np.ones((n, 1), dtype=np.float64)] if intercept else []) + blocks
    return np.column_stack(mats)


def ols_t(Y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS for many dependent columns at once.

    Y : (n, ...)   trailing shape is flattened and restored
    X : (n, p)     design matrix, intercept included by the caller

    Returns ``beta`` and ``t`` with shape ``(p, *Y.shape[1:])``.  Uses the normal
    equations with a pseudo-inverse so a rank-deficient (e.g. all-same-hand)
    design degrades to the minimum-norm solution instead of raising; callers are
    expected to check ``matrix_rank`` before trusting the t values.
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if Y.ndim == 0 or X.ndim != 2:
        raise ValueError("Y must have at least 1 dim and X must be 2-D")
    n, p = X.shape
    if Y.shape[0] != n:
        raise ValueError(f"Y has {Y.shape[0]} rows but X has {n}")

    y2 = Y.reshape(n, -1)
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y2)

    resid = y2 - X @ beta
    dof = max(n - int(np.linalg.matrix_rank(X)), 1)
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(np.outer(np.diag(xtx_inv), s2)) + 1e-30
    t = beta / se

    return beta.reshape((p,) + Y.shape[1:]), t.reshape((p,) + Y.shape[1:])


def t_from_corr(Y: np.ndarray, x: Sequence[float] | np.ndarray) -> np.ndarray:
    """t statistic of a single-regressor regression, computed from Pearson r.

    Equivalent to the second coefficient of ``OLS(Y ~ 1 + x)`` but cheaper; kept
    for direct comparability with the legacy analysis scripts.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    Y = np.asarray(Y, dtype=np.float64)
    n = x.size
    if Y.shape[0] != n:
        raise ValueError(f"Y has {Y.shape[0]} rows but x has {n}")

    xc = x - x.mean()
    y2 = Y.reshape(n, -1)
    yc = y2 - y2.mean(axis=0)
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum(axis=0)) + 1e-30
    r = np.clip((xc @ yc) / den, -0.999999, 0.999999)
    t = r * np.sqrt(max(n - 2, 1)) / np.sqrt(1.0 - r ** 2)
    return t.reshape(Y.shape[1:])


def vif(x: Sequence[float] | np.ndarray, others: np.ndarray) -> float:
    """Variance inflation factor of ``x`` given ``others``: 1 / (1 - R^2).

    ``others`` is (n, k) and must NOT contain an intercept column; one is added
    here.  A value above 5 (rule R3) means the collinearity is severe enough that
    the two regressors cannot be separated in this design.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    others = np.asarray(others, dtype=np.float64)
    if others.ndim == 1:
        others = others[:, None]
    if others.shape[0] != x.size:
        raise ValueError("x and others must have the same number of rows")

    X = np.column_stack([np.ones(x.size), others])
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    resid = x - X @ beta
    ss_tot = float(((x - x.mean()) ** 2).sum())
    if ss_tot <= 1e-30:
        return float("inf")
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
    return 1.0 / max(1e-12, 1.0 - r2)


def condition_number(X: np.ndarray) -> float:
    """Condition number of the design matrix on unit-norm scaled columns.

    Columns are scaled to unit L2 norm first (the intercept is left alone) so the
    number does not depend on whether pitch is in semitones or Hz.  Above ~20
    with a handful of regressors is a red flag for near-separation.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    Z = X.copy()
    for j in range(Z.shape[1]):
        nrm = float(np.linalg.norm(Z[:, j]))
        if nrm > 1e-12:
            Z[:, j] /= nrm
    return float(np.linalg.cond(Z))


def perm_p_abs(observed: float, null: Sequence[float] | np.ndarray) -> float:
    """Two-sided permutation p with the ``(k + 1) / (n + 1)`` correction."""
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    k = int((np.abs(null) >= abs(observed)).sum())
    return (k + 1) / (null.size + 1)


def permute_within_blocks(
    values: Sequence[float] | np.ndarray,
    blocks: Sequence[float] | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle ``values`` independently inside each block.

    Blocks of size 1 stay put; that is correct (they carry no permutation
    information) and keeps the function total rather than raising.
    """
    values = np.asarray(values).copy()
    blocks = np.asarray(blocks)
    if values.shape[0] != blocks.shape[0]:
        raise ValueError("values and blocks must have the same length")
    for b in np.unique(blocks):
        m = np.flatnonzero(blocks == b)
        if m.size > 1:
            values[m] = values[rng.permutation(m)]
    return values


def corr_r(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Pearson r with a NaN- and zero-variance-safe fallback."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    av, bv = a[m], b[m]
    if av.std() <= 1e-12 or bv.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def thresholds() -> dict[str, Any]:
    """Numeric thresholds quoted in the SOP, in one place so reports stay consistent."""
    return {
        "hand_agree_95": 0.95,      # R1: >=95% -> must be labelled "co-varies with hand"
        "vif_max": 5.0,             # R3
        "abs_r_max": 0.70,          # R3
        "hand_map_r_confound": 0.50,  # R4 -> motion-confound candidate
        "hand_map_r_clean": 0.30,   # R4 -> "cannot be explained by keypress"
        "topk_overlap_confound": 0.50,  # R4
        "rt_r_weak": 0.30,          # R6
        "min_minority_hand": 20,    # R4: below this, mid-tone hand variance is unusable
    }
