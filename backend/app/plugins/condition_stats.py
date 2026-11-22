"""Hand-confound diagnosis and three-flavour condition contrasts (SOP rules R1-R8).

The problem this plugin exists to prevent
-----------------------------------------
In this paradigm each anchor tone maps to a *fixed* correct key.  Measured on the
two existing subjects, hand side agrees with the tone->mapping prediction on
124/126 and 126/126 anchor trials, and ``r(pitch, hand)`` is +/-0.84 across all
main trials.  "High vs low tone" and "right vs left hand" are therefore the same
variable on anchor trials, and no reweighting inside the anchors can undo it.
Adding ``hand`` as a covariate does not remove the confound either: it splits the
shared variance (CIL8 t 5.38 -> 2.07 for 001) without telling you which half was
the keypress.

So the only honest output is a *set* of flavours reported side by side:

    A  anchors only          y ~ 1 + pitch_lin                 legacy-comparable, contains hand
    B  all main trials       y ~ 1 + pitch_lin + hand + block  hand in model; collinear -> lower bound
    C  ambiguous tones only  y ~ 1 + pitch_lin + hand          hand has real variance; cleanest, low power

Reporting only A or only B is selective reporting and the plugin refuses to
present a single number as "the" effect: every arm gets its own max|t| family
correction over (channel x window) and the report prints all three.

Rules implemented (see AGENTS.md):
  R1 hand agreement >= 95% -> contrast must be labelled "condition (co-varies
     with hand)", never "valence effect" / "decision effect".
  R2 all three arms reported, with family-corrected max|t| AND per-channel t.
  R3 report VIF(pitch), design-matrix condition number, r(pitch, hand);
     VIF > 5 or |r| > 0.7 -> "not separable".
  R4 mid-tone hand map: r(condition map, hand map), top-10 overlap, whether the
     peak channel sits on the hand map.  |r| > 0.5 or overlap >= 50% ->
     "motion-confound candidate", not admissible for the conclusions section.
  R5 primary window 0-200 ms; 150-350 ms tagged as possibly containing premotor.
  R6 per-channel r(amplitude, RT); report the |r| > 0.3 fraction and the peak.
  R7 nothing is dropped or reweighted to manufacture hand balance.
  R8 without a counterbalanced two-hand design, never claim hand is controlled.

Statistics conventions are fixed in ``backend/app/stats_core.py``.
"""

from __future__ import annotations

from typing import Any

import os
import numpy as np

from .. import stats_core as sc
from ..plugin_base import AnalysisPlugin, PluginContext

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_jbs")

# Overlapping windows are intentional: the family is a set of *questions*
# ("is anything different in the first 50 ms"), not a partition of time.
WINDOWS: dict[str, tuple[float, float]] = {
    "0-50ms": (0.0, 0.05),
    "50-100ms": (0.05, 0.10),
    "100-200ms": (0.10, 0.20),
    "200-300ms": (0.20, 0.30),
    "150-350ms": (0.15, 0.35),
    "300-500ms": (0.30, 0.50),
}
PRIMARY_WINDOW = "0-200ms"
PREMOTOR_WINDOWS = {"150-350ms", "200-300ms", "300-500ms"}

ARM_KEYS = ("A", "B", "C")
ARM_LABELS = {
    "A": "A · 锚点 only（含手，与旧结果可比）",
    "B": "B · 全部 main + hand + block（共线，只当下界）",
    "C": "C · 中音 only + hand（较干净，试次少）",
}


# ---------------------------------------------------------------------------
# Metadata coercion: MNE round-trips booleans/strings through several dtypes
# depending on version, and a single unexpected None must not kill the run.
# ---------------------------------------------------------------------------

def _num_col(meta, name: str, n: int) -> np.ndarray:
    """Numeric column with NaN for anything unparsable."""
    if name not in meta.columns:
        return np.full(n, np.nan)
    out = np.full(n, np.nan)
    for i, v in enumerate(meta[name].values):
        try:
            if v is None:
                continue
            out[i] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _str_col(meta, name: str, n: int) -> np.ndarray:
    """String column with '' for missing values."""
    if name not in meta.columns:
        return np.full(n, "", dtype=object)
    out = np.empty(n, dtype=object)
    for i, v in enumerate(meta[name].values):
        out[i] = "" if v is None else str(v).strip()
    return out


def _bool_col(meta, name: str, n: int) -> np.ndarray:
    """Boolean column tolerant of 0/1, 'True'/'False', '1.0'."""
    if name not in meta.columns:
        return np.zeros(n, dtype=bool)
    out = np.zeros(n, dtype=bool)
    for i, v in enumerate(meta[name].values):
        if isinstance(v, (bool, np.bool_)):
            out[i] = bool(v)
        else:
            s = str(v).strip().lower()
            if s in ("1", "1.0", "true", "yes", "y"):
                out[i] = True
    return out


# ---------------------------------------------------------------------------
# Trial table -> design columns
# ---------------------------------------------------------------------------

def build_design_columns(meta, n_ch_expected: int | None = None) -> dict[str, Any]:
    """Derive pitch / hand / block / anchor / mid from an epochs metadata table.

    ``pitch_semitone`` is the semitone offset of the physical tone relative to
    261.63 Hz, shifted so the five-tone ladder is -6, -3, 0, +3, +6.  This keeps
    the *physical* axis, which is what "condition" means when valence and pitch
    are collinear inside a session.

    The correct key per anchor class is inferred from trials the logger marked
    ``correct == 1`` rather than hardcoded, because the pos->hand mapping is
    counterbalanced across subjects and sessions.
    """
    n = len(meta)
    hz = _num_col(meta, "hz", n)
    choice_phys = _str_col(meta, "choice_phys", n)
    choice_sem = _str_col(meta, "choice_sem", n)
    cls = _str_col(meta, "cls", n)
    eff = _str_col(meta, "eff", n)
    responded = _bool_col(meta, "responded", n)
    correct = _num_col(meta, "correct", n)
    block = _num_col(meta, "block", n)
    rt = _num_col(meta, "rt_ms", n)
    phase = _str_col(meta, "phase", n)

    pitch = np.round(12.0 * np.log2(np.where(hz > 0, hz, np.nan) / 261.63), 2) - 6.0
    hand = np.where(choice_phys == "right", 1.0, np.where(choice_phys == "left", 0.0, np.nan))

    is_anchor = np.isin(cls, ["high", "low"])
    is_mid = np.array([c.startswith("m") for c in cls])
    is_main = (phase == "main") if np.any(phase == "main") else np.ones(n, dtype=bool)

    # Infer the correct physical key for each anchor class from scored trials.
    # If a class has no correct trials we leave it as NaN and the agreement
    # statistic is simply unavailable rather than silently assuming "right".
    pred_hand: dict[str, float] = {}
    for c in ("high", "low"):
        m = is_anchor & (cls == c) & (correct == 1) & np.isfinite(hand)
        if m.sum() >= 1:
            pred_hand[c] = 1.0 if hand[m].mean() >= 0.5 else 0.0

    # Valence side of each anchor, kept only for reporting; never used as a
    # regressor because inside one session it is collinear with pitch.
    pos_cls = ""
    for c in ("high", "low"):
        if np.any(is_anchor & (cls == c) & (eff == "pos")):
            pos_cls = c
            break

    return {
        "n": n,
        "pitch": pitch,
        "hand": hand,
        "block": block,
        "rt": rt,
        "cls": cls,
        "choice_sem": choice_sem,
        "is_anchor": is_anchor,
        "is_mid": is_mid,
        "is_main": is_main,
        "responded": responded,
        "pred_hand": pred_hand,
        "pos_anchor_cls": pos_cls,
        "neg_anchor_cls": ("low" if pos_cls == "high" else ("high" if pos_cls == "low" else "")),
    }


def hand_agreement(cols: dict[str, Any]) -> dict[str, Any]:
    """R1: fraction of anchor trials whose hand matches the tone->key mapping."""
    is_anchor = cols["is_anchor"]
    cls = cols["cls"]
    hand = cols["hand"]
    pred_hand = cols["pred_hand"]

    m = is_anchor & np.isfinite(hand)
    if m.sum() == 0 or not pred_hand:
        return {
            "n_anchor": int(m.sum()),
            "n_agree": 0,
            "agree_rate": float("nan"),
            "minority_hand_trials": 0,
            "available": False,
        }

    pred = np.array([pred_hand.get(c, np.nan) for c in cls[m]], dtype=float)
    ok = np.isfinite(pred)
    n_agree = int((pred[ok] == hand[m][ok]).sum())
    n_valid = int(ok.sum())
    agree_rate = n_agree / n_valid if n_valid else float("nan")

    return {
        "n_anchor": n_valid,
        "n_agree": n_agree,
        "agree_rate": agree_rate,
        "minority_hand_trials": n_valid - n_agree,
        "available": n_valid > 0,
    }


def collinearity(cols: dict[str, Any], use: np.ndarray, with_block: bool = True) -> dict[str, Any]:
    """R3: r(pitch, hand), VIF(pitch), design-matrix condition number."""
    p = cols["pitch"][use]
    h = cols["hand"][use]
    b = cols["block"][use]

    ok = np.isfinite(p) & np.isfinite(h)
    p, h = p[ok], h[ok]
    b = np.nan_to_num(b[ok], nan=0.0)

    out: dict[str, Any] = {"n": int(ok.sum())}
    if ok.sum() < 4:
        out.update({"r_pitch_hand": float("nan"), "vif_pitch": float("nan"),
                    "condition_number": float("nan"), "n_hand_levels": 0})
        return out

    out["r_pitch_hand"] = sc.corr_r(p, h)
    others = np.column_stack([h, b]) if with_block else h[:, None]
    out["vif_pitch"] = sc.vif(p, others)
    out["condition_number"] = sc.condition_number(sc.design(p, h, b) if with_block else sc.design(p, h))
    out["n_hand_levels"] = int(len(np.unique(h[np.isfinite(h)])))
    out["n_minority_hand"] = int(min((h == 0).sum(), (h == 1).sum()))
    return out


def separability_verdict(r: float, vif: float) -> str:
    """R3 threshold -> one of three fixed phrases."""
    if not (np.isfinite(r) or np.isfinite(vif)):
        return "不可判定（设计矩阵奇异或试次不足）"
    if (np.isfinite(vif) and vif > sc.thresholds()["vif_max"]) or (
        np.isfinite(r) and abs(r) > sc.thresholds()["abs_r_max"]
    ):
        return "不可分离"
    return "可分离（但见 A/B 差异）"


# ---------------------------------------------------------------------------
# Core estimator: works on plain arrays so it can be tested without MNE
# ---------------------------------------------------------------------------

def window_matrix(data: np.ndarray, times: np.ndarray, windows: dict[str, tuple[float, float]]) -> tuple[np.ndarray, list[str]]:
    """Average each window to give (n_trials, n_ch * n_windows) plus the flat labels."""
    blocks = []
    labels: list[str] = []
    for name, (t0, t1) in windows.items():
        m = (times >= t0) & (times < t1)
        if not np.any(m):
            continue
        blocks.append(data[:, :, m].mean(axis=2))
        labels.extend([f"{name}|{i}" for i in range(data.shape[1])])
    if not blocks:
        raise ValueError("所有分析窗口都不在 Epoch 时间范围内。")
    return np.concatenate(blocks, axis=1), labels


def _nuisance_basis(nuisance: np.ndarray | None, n: int) -> tuple[np.ndarray, int]:
    """Orthonormal basis of [intercept | nuisance] plus its column rank.

    SVD rather than QR: when a nuisance column is constant (e.g. every trial used
    the same hand), the design is rank deficient and an unpivoted QR would hand
    back a basis that misses the true column space.  SVD with a relative cutoff
    gives a genuinely orthonormal basis of whatever is identifiable.
    """
    cols = [np.ones((n, 1), dtype=np.float64)]
    if nuisance is not None and np.size(nuisance):
        arr = np.asarray(nuisance, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        cols.append(arr)
    R = np.column_stack(cols)
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    if s.size == 0:
        return np.zeros((n, 0)), 0
    k = int((s > s[0] * 1e-10).sum())
    return U[:, :k], k


def _residualize(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Remove the nuisance subspace from v (last axis is time/trials)."""
    if Q.shape[1] == 0:
        return v
    return v - Q @ (Q.T @ v)


def _t_from_residuals(Yr: np.ndarray, p_res: np.ndarray, dof: int) -> np.ndarray:
    """t of the single residualized regressor p_res against residualized Yr.

    Frisch-Waugh-Lovell: fitting Y ~ [nuisance, pitch] gives exactly this t for
    the pitch coefficient, because Yr and p_res live in the orthogonal complement
    of the nuisance space.
    """
    pp = float(p_res @ p_res)
    if pp <= 1e-30:
        return np.full(Yr.shape[-1], np.nan)
    beta = (p_res @ Yr) / pp
    resid = Yr - np.outer(p_res, beta)
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(s2 / pp) + 1e-30
    return beta / se


def run_arm(
    flat: np.ndarray,
    labels: list[str],
    cols: dict[str, Any],
    use: np.ndarray,
    regressors: list[np.ndarray],
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    """Fit one arm, then build the max|t| null by permuting the pitch column.

    Only the pitch column is permuted: hand and block stay attached to their real
    trials, so under the null we ask "how large does the pitch coefficient get
    when tone labels are shuffled inside block, given this hand structure?".
    """
    keep = use & np.isfinite(cols["pitch"])
    for r in regressors:
        keep = keep & np.isfinite(r)
    if int(keep.sum()) < 8:
        return {
            "n_trials": int(keep.sum()),
            "max_abs_t": float("nan"),
            "family_p": float("nan"),
            "peak_label": "",
            "peak_window": "",
            "peak_channel_index": -1,
            "peak_t_signed": float("nan"),
            "t_primary": np.full(flat.shape[1], np.nan),
            "unavailable": f"有效试次 {int(keep.sum())} < 8，无法拟合。",
        }

    Y = np.asarray(flat[keep], dtype=np.float64)
    pitch = np.asarray(cols["pitch"][keep], dtype=np.float64)
    block = np.nan_to_num(np.asarray(cols["block"][keep], dtype=np.float64), nan=0.0)
    regs = [np.asarray(r, dtype=np.float64)[keep] for r in regressors]

    nuisance = np.column_stack(regs) if regs else None
    Q, k = _nuisance_basis(nuisance, Y.shape[0])
    dof = max(Y.shape[0] - k - 1, 1)

    Yr = _residualize(Y, Q)
    p_res = _residualize(pitch, Q)

    t_pitch = _t_from_residuals(Yr, p_res, dof)
    observed = float(np.nanmax(np.abs(t_pitch)))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        p_perm = sc.permute_within_blocks(pitch, block, rng)
        ti = _t_from_residuals(Yr, _residualize(p_perm, Q), dof)
        null[i] = float(np.nanmax(np.abs(ti)))

    p_family = sc.perm_p_abs(observed, null)

    j = int(np.nanargmax(np.abs(t_pitch)))
    peak_label = labels[j]
    peak_window, peak_ch = peak_label.split("|", 1)

    # Restrict the per-channel t to the primary window for the hand-map test.
    primary_cols = [i for i, lab in enumerate(labels) if lab.startswith(PRIMARY_WINDOW + "|")]
    t_primary = np.full(flat.shape[1], np.nan)
    if primary_cols:
        t_primary[primary_cols] = t_pitch[primary_cols]

    return {
        "n_trials": int(keep.sum()),
        "max_abs_t": observed,
        "family_p": p_family,
        "null_q95": float(np.nanpercentile(null, 95)),
        "peak_label": peak_label,
        "peak_window": peak_window,
        "peak_channel_index": int(peak_ch),
        "peak_t_signed": float(t_pitch[j]),
        "t_primary": t_primary,
        "unavailable": "",
    }


def run_condition_stats(
    data: np.ndarray,
    times: np.ndarray,
    cols: dict[str, Any],
    ch_names: list[str],
    n_perm: int = 1000,
    seed: int = 20261112,
    windows: dict[str, tuple[float, float]] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run R1-R6 and return everything a report needs. No MNE, no file IO."""
    windows = windows or {**WINDOWS, PRIMARY_WINDOW: (0.0, 0.2)}
    flat, labels = window_matrix(data, times, windows)
    base = cols["is_main"] & cols["responded"]
    anchor = base & cols["is_anchor"]
    mid = base & cols["is_mid"]

    # ---- R1: how badly is hand tied to condition? -------------------------
    agree = hand_agreement(cols)
    rate = agree["agree_rate"]
    co_varies = bool(np.isfinite(rate) and rate >= sc.thresholds()["hand_agree_95"])
    label = "条件（与手侧共变）" if co_varies else "条件"

    # ---- R3: collinearity diagnostics per arm ----------------------------
    diag = {
        "A": collinearity(cols, anchor, with_block=False),
        "B": collinearity(cols, base, with_block=True),
        "C": collinearity(cols, mid, with_block=False),
    }

    # ---- R2: the three arms ----------------------------------------------
    arms: dict[str, dict[str, Any]] = {}
    arms["A"] = run_arm(flat, labels, cols, anchor, [], n_perm, seed + 1)
    arms["B"] = run_arm(flat, labels, cols, base, [cols["hand"], cols["block"]], n_perm, seed + 2)
    arms["C"] = run_arm(flat, labels, cols, mid, [cols["hand"]], n_perm, seed + 3)

    # ---- R4: mid-tone hand map -------------------------------------------
    hand_map = hand_effect_map(flat, labels, cols, mid, seed + 4)
    confound = hand_map_vs_condition(hand_map, arms, top_k)

    # ---- R6: amplitude vs RT ---------------------------------------------
    rt_check = amplitude_rt_correlation(flat, labels, cols, base)

    return {
        "label": label,
        "co_varies_with_hand": co_varies,
        "hand_agreement": agree,
        "collinearity": diag,
        "separability": {k: separability_verdict(v.get("r_pitch_hand", float("nan")),
                                                 v.get("vif_pitch", float("nan")))
                         for k, v in diag.items()},
        "arms": arms,
        "hand_map": hand_map,
        "confound": confound,
        "rt_check": rt_check,
        "n_anchor": int(anchor.sum()),
        "n_mid": int(mid.sum()),
        "n_base": int(base.sum()),
        "n_windows": len(windows),
        "n_channels": len(ch_names),
        "n_perm": n_perm,
        "seed": seed,
        "windows": list(windows.keys()),
        "labels": labels,
        "ch_names": ch_names,
        "can_claim_hand_controlled": False,
    }


def hand_effect_map(
    flat: np.ndarray,
    labels: list[str],
    cols: dict[str, Any],
    mid: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """R4: per-channel t of hand, estimated on ambiguous tones only.

    Mid tones are the only place hand varies independently of pitch, so this map
    is the purest available picture of "what a keypress looks like here".  If it
    is unavailable (one hand only, or too few minority trials) the report says so
    instead of silently substituting the anchor contrast.
    """
    keep = mid & np.isfinite(cols["pitch"]) & np.isfinite(cols["hand"])
    n_minority = int(min((cols["hand"][keep] == 0).sum(), (cols["hand"][keep] == 1).sum()))
    if int(keep.sum()) < 8 or n_minority < sc.thresholds()["min_minority_hand"]:
        return {
            "available": False,
            "n_trials": int(keep.sum()),
            "n_minority": n_minority,
            "reason": (
                f"中音手侧变异不足，无法诊断（少数手 {n_minority} 个试次 "
                f"< {sc.thresholds()['min_minority_hand']}）。"
            ),
            "t_primary": np.full(flat.shape[1], np.nan),
        }

    Y = flat[keep]
    X = sc.design(cols["pitch"][keep], cols["hand"][keep])
    _, t = sc.ols_t(Y, X)
    t_hand = t[2]

    primary = [i for i, lab in enumerate(labels) if lab.startswith(PRIMARY_WINDOW + "|")]
    t_primary = np.full(flat.shape[1], np.nan)
    t_primary[primary] = t_hand[primary]

    return {
        "available": True,
        "n_trials": int(keep.sum()),
        "n_minority": n_minority,
        "reason": "",
        "t_primary": t_primary,
        "max_abs_t": float(np.nanmax(np.abs(t_primary))),
    }


def confound_verdict(r: float, overlap: float, chance: float) -> str:
    """R4 decision rule as a pure function so it can be unit-tested.

    ``overlap`` is the top-k overlap fraction and ``chance`` is what two unrelated
    maps would give on this montage (k / n_channels).  Requiring overlap to be
    both above the absolute SOP threshold and >= 3x chance keeps small montages
    from being spuriously labelled: on 24 channels a top-10 list already overlaps
    ~42% by chance.
    """
    th = sc.thresholds()
    r_high = bool(np.isfinite(r) and abs(r) > th["hand_map_r_confound"])
    ratio = overlap / chance if np.isfinite(chance) and chance > 0 else float("nan")
    ov_high = bool(
        overlap >= th["topk_overlap_confound"]
        and np.isfinite(ratio)
        and ratio >= 3.0
    )
    if r_high or ov_high:
        return "运动混淆候选"
    return ""


def hand_map_vs_condition(hand_map: dict[str, Any], arms: dict[str, Any], top_k: int) -> dict[str, Any]:
    """R4 verdicts: spatial correlation, top-k overlap, does the peak sit on the hand map.

    The SOP quotes "top-10 overlap >= 50%" from a ~170-channel dataset where
    chance overlap is ~6%.  On a small montage (say 24 channels) two unrelated
    maps already agree on 42% of a top-10 list, so the raw threshold would call
    everything a motion confound.  We therefore report the overlap *and* its
    ratio to the chance level, and only fire on overlap when it is also well
    above chance.
    """
    if not hand_map.get("available"):
        return {k: {"verdict": "无法诊断", "reason": hand_map.get("reason", "")} for k in ARM_KEYS}

    t_hand = hand_map["t_primary"]
    n_ch = int(t_hand.size)
    finite = np.isfinite(t_hand)
    k_eff = max(3, min(int(top_k), n_ch))
    chance = k_eff / n_ch if n_ch else float("nan")
    top_hand = set(np.argsort(-np.abs(np.nan_to_num(t_hand, nan=-np.inf)))[:k_eff].tolist())

    out: dict[str, Any] = {}
    for key in ARM_KEYS:
        arm = arms.get(key, {})
        t_cond = arm.get("t_primary")
        if t_cond is None or not np.any(np.isfinite(t_cond)):
            out[key] = {"verdict": "无法诊断", "reason": "该口径无可用 t 图。"}
            continue

        m = finite & np.isfinite(t_cond)
        r = sc.corr_r(t_cond[m], t_hand[m])
        top_cond = set(np.argsort(-np.abs(np.nan_to_num(t_cond, nan=-np.inf)))[:k_eff].tolist())
        overlap = len(top_cond & top_hand) / max(k_eff, 1)
        overlap_over_chance = overlap / chance if np.isfinite(chance) and chance > 0 else float("nan")

        peak = arm.get("peak_channel_index", -1)
        peak_on_hand = bool(
            0 <= peak < len(t_hand) and np.isfinite(t_hand[peak]) and abs(t_hand[peak]) > 2.0
        )

        confounded = confound_verdict(r, overlap, chance)

        if confounded:
            verdict = confounded
        elif np.isfinite(r) and abs(r) < sc.thresholds()["hand_map_r_clean"] and not peak_on_hand:
            verdict = "不能用按键解释（仍须注明共线）"
        else:
            verdict = "灰区：不足以判定归属"

        out[key] = {
            "r_condition_vs_hand": r,
            "topk_overlap": overlap,
            "topk_k": k_eff,
            "topk_chance": chance,
            "topk_over_chance": overlap_over_chance,
            "peak_channel_index": int(peak),
            "peak_on_hand_map": peak_on_hand,
            "peak_hand_t": float(t_hand[peak]) if 0 <= peak < len(t_hand) else float("nan"),
            "verdict": verdict,
            "reason": "",
            "top_hand_channels": sorted(top_hand),
        }
    return out


def amplitude_rt_correlation(
    flat: np.ndarray,
    labels: list[str],
    cols: dict[str, Any],
    base: np.ndarray,
) -> dict[str, Any]:
    """R6: per-channel r(amplitude in the primary window, RT).

    This is the table that lets a reader rule out "the effect is just how fast
    they responded".  It is mandatory; the plugin always emits it.
    """
    primary = [i for i, lab in enumerate(labels) if lab.startswith(PRIMARY_WINDOW + "|")]
    if not primary:
        return {"available": False, "reason": f"主窗口 {PRIMARY_WINDOW} 不在分析范围内。"}

    keep = base & np.isfinite(cols["rt"])
    if int(keep.sum()) < 8:
        return {"available": False, "reason": f"有 RT 的试次只有 {int(keep.sum())} 个，无法计算。"}

    Y = flat[np.ix_(keep, primary)]
    rt = cols["rt"][keep]

    yz = Y - Y.mean(axis=0)
    rz = rt - rt.mean()
    den = np.sqrt((yz ** 2).sum(axis=0) * (rz ** 2).sum()) + 1e-30
    r = (rz @ yz) / den

    return {
        "available": True,
        "n_trials": int(keep.sum()),
        "r": r,
        "frac_abs_gt_0.3": float(np.mean(np.abs(r) > sc.thresholds()["rt_r_weak"])),
        "max_abs_r": float(np.max(np.abs(r))),
        "peak_index": int(np.argmax(np.abs(r))),
        "peak_r_signed": float(r[np.argmax(np.abs(r))]),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v: float, digits: int, suffix: str = "") -> str:
    """Format a possibly-NaN number for a Markdown table, em dash when unavailable."""
    return f"{v:.{digits}f}{suffix}" if np.isfinite(v) else "—"


def render_report(res: dict[str, Any], ch_names: list[str], top_k: int = 10) -> str:
    """Markdown report. Every number the SOP demands appears here or in the CSVs."""
    th = sc.thresholds()
    ag = res["hand_agreement"]
    L: list[str] = []

    L.append("# 条件效应 × 手侧混淆诊断（R1–R8）\n")
    L.append("> 本报告由 `condition_stats` 插件自动生成。规则见仓库 `README.md` 的 "
             "「手侧混淆」一节；统计口径见 `backend/app/stats_core.py`。\n")

    # ---- headline verdict ----
    L.append("## 0 结论口径（先看这一节）\n")
    if res["co_varies_with_hand"]:
        L.append(
            f"**本对比必须写成「{res['label']}」。** 锚点试次中手侧与音高→按键映射的符合率为 "
            f"{ag['n_agree']}/{ag['n_anchor']}"
            f"（{100 * ag['agree_rate']:.1f}% ≥ {100 * th['hand_agree_95']:.0f}%），"
            "因此「高 vs 低」与「左手 vs 右手」在本会话内是同一个变量。"
            "**不得**写成「效价效应」「决策效应」「判断偏倚的神经相关」。\n"
        )
    else:
        rate_txt = "不可计算" if not np.isfinite(ag["agree_rate"]) else f"{100 * ag['agree_rate']:.1f}%"
        L.append(
            f"锚点手侧符合率 {ag['n_agree']}/{ag['n_anchor']}（{rate_txt}），"
            f"低于 {100 * th['hand_agree_95']:.0f}% 门限，对比可写作「条件」，"
            "但仍须报告手侧诊断与共线性。\n"
        )

    L.append(
        "**不能声称「已控制手」。** 本范式每个锚点只对应一个正确按键，"
        "既没有双手 counterbalance，也没有随试次随机的按键映射；"
        "所有 `hand` 协变量做法只是把共享方差分流，不给出归属。\n"
    )

    # ---- R1 design facts ----
    L.append("## 1 设计事实（R1）\n")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 正性锚点音 | `{res.get('pos_anchor_cls') or '未知'}` |")
    L.append(f"| 锚点试次 | {res['n_anchor']} |")
    L.append(f"| 中音试次 | {res['n_mid']} |")
    L.append(f"| 全部 main（有手侧）| {res['n_base']} |")
    L.append(f"| 手侧符合映射 | {ag['n_agree']} / {ag['n_anchor']}"
             + (f" = {100 * ag['agree_rate']:.1f}%" if np.isfinite(ag['agree_rate']) else "")
             + " |")
    L.append(f"| 不一致（少数手）试次 | {ag['minority_hand_trials']} |")
    L.append("")

    # ---- R3 collinearity ----
    L.append("## 2 共线性门槛（R3）\n")
    L.append("| 口径 | n | r(pitch, hand) | VIF(pitch) | 条件数 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for key in ARM_KEYS:
        d = res["collinearity"][key]
        r = d.get("r_pitch_hand", float("nan"))
        v = d.get("vif_pitch", float("nan"))
        c = d.get("condition_number", float("nan"))
        L.append(
            f"| {key} | {d.get('n', 0)} | "
            f"{_fmt(r, 3)} | "
            f"{_fmt(v, 2)} | "
            f"{_fmt(c, 1)} | "
            f"{res['separability'][key]} |"
        )
    L.append("")
    L.append(f"判据：`VIF > {th['vif_max']:g}` 或 `|r| > {th['abs_r_max']:g}` → **不可分离**，"
             "只能写「与手侧不可分」，不得给出方向性解释。\n")

    # ---- R2 three arms ----
    L.append("## 3 三口径并列（R2）\n")
    L.append(f"家族 = 通道 × {res['n_windows']} 窗口；置换 {res.get('n_perm', 1000)} 次（block 内），"
             f"p = (k+1)/(n+1)，seed = {res.get('seed', '—')}。\n")
    L.append("| 口径 | 设计 | n | max&#124;t&#124; | 家族 p | 峰通道 | 峰窗口 | 峰 t（带符号）|")
    L.append("|---|---|---|---|---|---|---|---|")
    for key in ARM_KEYS:
        a = res["arms"][key]
        if a.get("unavailable"):
            L.append(f"| {key} | {ARM_LABELS[key]} | {a['n_trials']} | — | — | — | — | — |")
            continue
        ch = ch_names[a["peak_channel_index"]] if a["peak_channel_index"] < len(ch_names) else "?"
        L.append(
            f"| {key} | {ARM_LABELS[key]} | {a['n_trials']} | "
            f"{a['max_abs_t']:.2f} | {a['family_p']:.3f} | `{ch}` | {a['peak_window']} | "
            f"{a['peak_t_signed']:+.2f} |"
        )
    L.append("")
    L.append("**读法**：A 与 B 的差异量就是「无法归属的共享方差」大小。"
             "C 只有在少数手试次充足时才可信（见下节）。"
             "**只报 A 或只报 B 视为选择性报告。**\n")

    # ---- R4 hand map ----
    L.append("## 4 手图诊断（R4）\n")
    hm = res["hand_map"]
    if not hm.get("available"):
        L.append(f"⚠️ {hm.get('reason', '')}\n")
    else:
        L.append(f"中音手效应图：n = {hm['n_trials']}（少数手 {hm['n_minority']} 试次），"
                 f"max|t_hand| = {hm['max_abs_t']:.2f}（主窗口 {PRIMARY_WINDOW}）。\n")
        c_k = res["confound"]["A"].get("topk_k", top_k)
        L.append(f"| 口径 | r(条件图, 手图) | top-{c_k} 重合 | 相对机会 | 峰通道 | 峰在手图 | 判定 |")
        L.append("|---|---|---|---|---|---|---|")
        for key in ARM_KEYS:
            c = res["confound"].get(key, {})
            if c.get("verdict") == "无法诊断":
                L.append(f"| {key} | — | — | — | — | — | 无法诊断 |")
                continue
            r = c["r_condition_vs_hand"]
            pk = c.get("peak_channel_index", -1)
            pk_name = ch_names[pk] if 0 <= pk < len(ch_names) else "?"
            oc = c.get("topk_over_chance", float("nan"))
            L.append(
                f"| {key} | {_fmt(r, 3)} | "
                f"{100 * c['topk_overlap']:.0f}% (k={c.get('topk_k', top_k)}) | "
                f"{_fmt(oc, 1, '×')} | `{pk_name}` | "
                f"{'是' if c.get('peak_on_hand_map') else '否'} | {c['verdict']} |"
            )
        L.append("")
        L.append(
            f"判据：`|r| > {th['hand_map_r_confound']:g}` 或（top-k 重合 ≥ "
            f"{100 * th['topk_overlap_confound']:.0f}% **且** ≥ 3× 机会水平）→ **运动混淆候选**，"
            f"不进结论段；`|r| < {th['hand_map_r_clean']:g}` 且峰通道不在手图上 → "
            "可写「不能用按键解释」。"
            "（重合率必须相对机会水平 `k/通道数` 看：24 通道时 top-10 的随机重合就有 42%。）\n"
        )

    # ---- R6 RT ----
    L.append("## 5 RT 校验（R6）\n")
    rc = res["rt_check"]
    if not rc.get("available"):
        L.append(f"⚠️ {rc.get('reason', '')}\n")
    else:
        peak_ch = ch_names[rc["peak_index"]] if rc["peak_index"] < len(ch_names) else "?"
        L.append("| 项 | 值 |")
        L.append("|---|---|")
        L.append(f"| 试次 | {rc['n_trials']} |")
        L.append(f"| &#124;r&#124; > {th['rt_r_weak']:g} 的通道占比 | {100 * rc['frac_abs_gt_0.3']:.1f}% |")
        L.append(f"| 峰 &#124;r&#124; | {rc['max_abs_r']:.3f}（`{peak_ch}`，r = {rc['peak_r_signed']:+.3f}）|")
        L.append("")
        L.append("注：EMG 通道的相关性需要原始连续数据，插件内无法计算；"
                 "请在预处理阶段单独报告。\n")

    # ---- what may be written ----
    L.append("## 6 可以写 / 不可以写\n")
    L.append("**可以写：**")
    L.append(f"- 条件对比（n = {res['n_anchor'] if res['arms']['A'].get('n_trials') else 0} 锚点试次）"
             "的 t 图与家族校正 p 值；")
    L.append("- A/B/C 三口径并列的结果与它们的差异；")
    L.append("- 手侧诊断的全部数字（符合率、VIF、条件数、r(条件图, 手图)、top-10 重合）。")
    L.append("")
    L.append("**不可以写：**")
    L.append("- 「效价效应」「情绪效应」「决策效应」「判断偏倚的神经相关」——条件与手共变；")
    L.append("- 「已控制手」「排除了运动解释」——没有 counterbalance 就不成立；")
    L.append("- 剔除错误按键锚点、按 RT 配平手侧、事后挑通道/窗口来规避手图重叠；")
    L.append("- 把 `hand` 与 `pitch` 同时放进只含锚点的模型（完全共线，结果不可解释）。")
    L.append("")

    return "\n".join(L)


def csv_rows_models(res: dict[str, Any], ch_names: list[str]) -> list[dict[str, Any]]:
    """One row per (arm, window): the table a reader reproduces figures from."""
    rows: list[dict[str, Any]] = []
    for key in ARM_KEYS:
        a = res["arms"][key]
        d = res["collinearity"][key]
        c = res["confound"].get(key, {})
        rows.append(
            {
                "arm": key,
                "arm_label": ARM_LABELS[key],
                "n_trials": a.get("n_trials", 0),
                "n_windows": res["n_windows"],
                "max_abs_t": a.get("max_abs_t"),
                "family_p": a.get("family_p"),
                "null_q95": a.get("null_q95"),
                "peak_channel": (ch_names[a["peak_channel_index"]]
                                 if 0 <= a.get("peak_channel_index", -1) < len(ch_names) else ""),
                "peak_window": a.get("peak_window", ""),
                "peak_t_signed": a.get("peak_t_signed"),
                "r_pitch_hand": d.get("r_pitch_hand"),
                "vif_pitch": d.get("vif_pitch"),
                "condition_number": d.get("condition_number"),
                "separability": res["separability"][key],
                "r_condition_vs_hand": c.get("r_condition_vs_hand"),
                "topk_overlap": c.get("topk_overlap"),
                "confound_verdict": c.get("verdict", ""),
                "contrast_label": res["label"],
            }
        )
    return rows


def csv_rows_channels(res: dict[str, Any], ch_names: list[str]) -> list[dict[str, Any]]:
    """Per-channel t in the primary window, all arms plus the hand map."""
    rows: list[dict[str, Any]] = []
    hm = res["hand_map"].get("t_primary")
    rc = res["rt_check"]
    r_by_ch = rc.get("r") if rc.get("available") else None

    for i, ch in enumerate(ch_names):
        row: dict[str, Any] = {"channel": ch}
        for key in ARM_KEYS:
            t = res["arms"][key].get("t_primary")
            row[f"t_arm_{key}"] = float(t[i]) if t is not None and i < len(t) else None
        row["t_hand_map"] = float(hm[i]) if hm is not None and i < len(hm) else None
        row["r_rt"] = float(r_by_ch[i]) if r_by_ch is not None and i < len(r_by_ch) else None
        rows.append(row)
    return rows


def csv_rows_design(res: dict[str, Any]) -> list[dict[str, Any]]:
    ag = res["hand_agreement"]
    return [
        {
            "pos_anchor_cls": res.get("pos_anchor_cls", ""),
            "n_anchor": res["n_anchor"],
            "n_mid": res["n_mid"],
            "n_main_responded": res["n_base"],
            "hand_agrees_with_mapping": f"{ag['n_agree']}/{ag['n_anchor']}",
            "hand_agree_rate": ag["agree_rate"],
            "minority_hand_trials": ag["minority_hand_trials"],
            "contrast_label": res["label"],
            "co_varies_with_hand": res["co_varies_with_hand"],
            "can_claim_hand_controlled": res["can_claim_hand_controlled"],
            "primary_window": PRIMARY_WINDOW,
        }
    ]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def render_figure(res: dict[str, Any], ch_names: list[str], out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plot_style import BG, FG, MUTED, SPINE, apply_dark_theme, style_axes

    apply_dark_theme(plt)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=150, facecolor=BG)
    for ax in axes:
        style_axes(ax)

    # Panel A: condition map vs hand map, arm B (the arm where hand is in the model)
    ax = axes[0]
    t_cond = res["arms"]["B"].get("t_primary")
    t_hand = res["hand_map"].get("t_primary")
    if t_cond is not None and t_hand is not None:
        m = np.isfinite(t_cond) & np.isfinite(t_hand)
        strong = np.abs(t_hand[m]) > 2.0
        ax.scatter(t_cond[m][~strong], t_hand[m][~strong], s=18, color="#64748b",
                   alpha=0.8, label="|t_hand| ≤ 2")
        ax.scatter(t_cond[m][strong], t_hand[m][strong], s=26, color="#f43f5e",
                   alpha=0.9, label="|t_hand| > 2")
        r = res["confound"].get("B", {}).get("r_condition_vs_hand")
        ax.set_title(
            f"A. 条件图 vs 手图（{PRIMARY_WINDOW}，口径 B）\n"
            f"r = {r:+.3f}" if r is not None and np.isfinite(r) else f"A. 条件图 vs 手图（{PRIMARY_WINDOW}）",
            color="#f8fafc", fontsize=10,
        )
        ax.axhline(0, color=MUTED, lw=0.8, alpha=0.5)
        ax.axvline(0, color=MUTED, lw=0.8, alpha=0.5)
        ax.set_xlabel("t_condition (pitch)", color=FG)
        ax.set_ylabel("t_hand（中音估计）", color=FG)
        ax.legend(facecolor=BG, edgecolor=SPINE, labelcolor=FG, fontsize=8)
    else:
        ax.text(0.5, 0.5, "手图不可用\n（中音手侧变异不足）", color=FG,
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel B: max|t| per arm, with family p annotated
    ax = axes[1]
    keys = list(ARM_KEYS)
    vals = [res["arms"][k].get("max_abs_t", np.nan) for k in keys]
    ps = [res["arms"][k].get("family_p", np.nan) for k in keys]
    colors = ["#64748b", "#38bdf8", "#34d399"]
    bars = ax.bar(keys, [v if np.isfinite(v) else 0 for v in vals], color=colors, alpha=0.9, width=0.6)
    for b, v, p in zip(bars, vals, ps):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}\np={p:.3f}",
                    ha="center", va="bottom", color=FG, fontsize=8)
    ax.set_ylim(0, max([v for v in vals if np.isfinite(v)] + [3.0]) * 1.35)
    ax.set_ylabel("max t（跨通道 × 窗口）", color=FG)
    ax.set_title("B. 三口径家族校正结果（必须并列报告）", color="#f8fafc", fontsize=10)

    fig.suptitle(
        f"条件效应 × 手侧混淆诊断 · {res['label']} · "
        f"手侧符合 {res['hand_agreement']['n_agree']}/{res['hand_agreement']['n_anchor']}",
        color="#f8fafc", fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class ConditionStatsPlugin(AnalysisPlugin):
    plugin_id = "condition_stats"
    name = "条件效应 × 手侧混淆诊断（A/B/C 三口径 + 家族校正）"
    version = "1.0.0"
    description = (
        "按 SOP 规则 R1–R8 报告条件效应：先算锚点手侧符合率与共线性（VIF / 条件数 / r），"
        "再并列给出 A 锚点-only、B 全试次+hand、C 中音-only 三口径的 max|t| 家族校正结果，"
        "并用中音手图做运动混淆诊断。只报其中一个口径视为选择性报告。"
    )
    category = "statistics"
    requires = ["preprocess"]
    params_schema = [
        {
            "key": "n_perm",
            "label": "置换次数（建议 1000，最低 200）",
            "type": "number",
            "default": 1000,
        },
        {
            "key": "baseline",
            "label": "分析基线",
            "type": "select",
            "default": "pre_stim",
            "options": [
                {"value": "pre_stim", "label": "刺激前 -200…0 ms（推荐，分析阶段统一重基线）"},
                {"value": "none", "label": "不重基线（沿用预处理时的基线）"},
            ],
        },
        {
            "key": "top_k",
            "label": "手图 top-k 重合的 k",
            "type": "number",
            "default": 10,
        },
    ]

    def run(self, ctx: PluginContext, params: dict[str, Any]) -> dict[str, Any]:
        import mne

        if not ctx.epochs_fif_path or not ctx.epochs_fif_path.exists():
            raise FileNotFoundError("请先完成 Step 5（SEEG 预处理）以生成 Epochs .fif 文件。")

        n_perm = int(params.get("n_perm", 1000) or 1000)
        if n_perm < 1:
            raise ValueError("置换次数必须 ≥ 1。")
        top_k = int(params.get("top_k", 10) or 10)

        epochs = mne.read_epochs(str(ctx.epochs_fif_path), preload=True, verbose="ERROR")
        if epochs.metadata is None:
            raise ValueError(
                "Epochs 缺少 metadata（cls / hz / choice_phys / block）。"
                "请用 Step 5 重新生成，本插件无法从纯波形恢复条件标签。"
            )

        data = epochs.get_data() * 1e6  # uV, (n_trials, n_ch, n_times)
        times = np.asarray(epochs.times, dtype=float)
        ch_names = list(epochs.ch_names)

        # Analytical re-baselining. The preprocessing default is whole-epoch mean,
        # which smears post-stimulus evoked activity backwards into the pre-stim
        # interval and manufactures t values; the analysis stage must re-baseline.
        baseline_mode = str(params.get("baseline", "pre_stim"))
        if baseline_mode == "pre_stim":
            m = (times >= -0.2) & (times < 0.0)
            if np.any(m):
                data = data - data[:, :, m].mean(axis=2, keepdims=True)
            else:
                raise ValueError(
                    f"Epoch 起点为 {times[0]:.2f} s，不足以做 -200…0 ms 重基线。"
                    "请在 Step 5 把 Epoch 起点设为 ≤ -0.2 s，或改选「不重基线」。"
                )

        cols = build_design_columns(epochs.metadata)
        cols["ch_names"] = ch_names

        res = run_condition_stats(
            data=data,
            times=times,
            cols=cols,
            ch_names=ch_names,
            n_perm=n_perm,
            top_k=top_k,
        )
        res["pos_anchor_cls"] = cols["pos_anchor_cls"]

        out = ctx.output_dir
        out.mkdir(parents=True, exist_ok=True)

        report_path = out / "CONDITION_STATS.md"
        report_path.write_text(render_report(res, ch_names, top_k=top_k), encoding="utf-8")

        import csv

        def _write(path, rows):
            if not rows:
                return None
            keys = list({k for r in rows for k in r})
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            return str(path)

        diag_csv = _write(out / "s5_hand_diagnostics.csv", csv_rows_design(res))
        model_csv = _write(out / "s5_hand_models.csv", csv_rows_models(res, ch_names))
        chan_csv = _write(out / "s5_hand_channels.csv", csv_rows_channels(res, ch_names))

        fig_path = out / "condition_stats.png"
        render_figure(res, ch_names, fig_path)

        a_b = res["arms"]["A"]
        b_b = res["arms"]["B"]
        c_b = res["arms"]["C"]
        summary = (
            f"{res['label']}｜手侧符合 {res['hand_agreement']['n_agree']}/"
            f"{res['hand_agreement']['n_anchor']}｜"
            f"A max|t|={a_b.get('max_abs_t', float('nan')):.2f} (p={a_b.get('family_p', float('nan')):.3f})，"
            f"B={b_b.get('max_abs_t', float('nan')):.2f} (p={b_b.get('family_p', float('nan')):.3f})，"
            f"C={c_b.get('max_abs_t', float('nan')):.2f} (p={c_b.get('family_p', float('nan')):.3f})"
            + ("" if res["hand_map"].get("available") else "｜手图不可用")
        )

        return {
            "summary": summary,
            "metrics": {
                "contrast_label": res["label"],
                "co_varies_with_hand": res["co_varies_with_hand"],
                "can_claim_hand_controlled": res["can_claim_hand_controlled"],
                "hand_agree_rate": res["hand_agreement"]["agree_rate"],
                "n_anchor": res["n_anchor"],
                "n_mid": res["n_mid"],
                "arms": {
                    k: {
                        "n_trials": res["arms"][k].get("n_trials"),
                        "max_abs_t": res["arms"][k].get("max_abs_t"),
                        "family_p": res["arms"][k].get("family_p"),
                        "peak_channel": ch_names[res["arms"][k]["peak_channel_index"]]
                        if 0 <= res["arms"][k].get("peak_channel_index", -1) < len(ch_names) else "",
                        "peak_window": res["arms"][k].get("peak_window"),
                    }
                    for k in ARM_KEYS
                },
                "separability": res["separability"],
                "confound": {
                    k: {"verdict": v.get("verdict"), "r": v.get("r_condition_vs_hand")}
                    for k, v in res["confound"].items()
                },
                "hand_map_available": res["hand_map"].get("available"),
                "rt_max_abs_r": res["rt_check"].get("max_abs_r"),
            },
            "image_path": str(fig_path),
            "artifacts": {
                "report": str(report_path),
                "diagnostics_csv": diag_csv,
                "models_csv": model_csv,
                "channels_csv": chan_csv,
                "figure": str(fig_path),
            },
        }
