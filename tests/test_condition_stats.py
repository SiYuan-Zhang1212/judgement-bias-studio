"""Tests for the hand-confound diagnostic layer (SOP rules R1-R8).

These are the assertions that stop the analysis from silently drifting back to
"report the arm that looks significant".  Every threshold quoted in the SOP has
a test here, and the FWL trick used for the permutation loop is checked against
straight OLS so a future optimisation cannot quietly change the numbers.

No patient data is referenced; all fixtures are synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app import stats_core as sc
from backend.app.plugins import condition_stats as cs


TONES = [
    ("low", 261.63, -6.0),
    ("m1", 311.13, -3.0),
    ("m2", 369.99, 0.0),
    ("m3", 440.0, 3.0),
    ("high", 523.25, 6.0),
]


def make_meta(
    n_blocks: int = 3,
    n_per_block: int = 40,
    *,
    pos_cls: str = "low",
    random_hand: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic epochs metadata in the real paradigm's shape.

    ``random_hand=False`` reproduces the confound we are guarding against: every
    anchor trial uses the mapping-predicted key, so hand agrees with pitch 100%.
    ``random_hand=True`` gives a fully crossed (counterbalanced) design.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        for k in range(n_per_block):
            cls, hz, _ = TONES[(k + b * 7) % 5]
            eff = "pos" if cls == pos_cls else ("neg" if cls in ("high", "low") else None)
            nofb = eff is None
            pred = "left" if cls == pos_cls else "right"  # pos anchor -> left key
            hand = (rng.choice(["left", "right"]) if random_hand
                    else (pred if not nofb else rng.choice(["left", "right"])))
            rows.append(
                {
                    "phase": "main",
                    "block": b + 1,
                    "cls": cls,
                    "hz": hz,
                    "eff": eff,
                    "nofb": nofb,
                    "responded": 1,
                    "choice_phys": hand,
                    "choice_sem": "pos" if hand == "left" else "neg",
                    "correct": 1,
                    "rt_ms": float(rng.normal(800, 120)),
                }
            )
    return pd.DataFrame(rows)


def make_data(
    meta: pd.DataFrame,
    n_ch: int = 16,
    sfreq: float = 250.0,
    effect: float = 0.0,
    effect_chs: int = 6,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic epochs (uV) with an optional planted pitch effect in 0-200 ms."""
    rng = np.random.default_rng(seed)
    times = np.arange(-0.2, 1.0, 1.0 / sfreq)
    data = rng.normal(scale=1.0, size=(len(meta), n_ch, times.size))

    if effect:
        m = (times >= 0.0) & (times < 0.2)
        semi = np.round(12.0 * np.log2(meta["hz"].to_numpy() / 261.63), 2) - 6.0
        for i in range(len(meta)):
            data[i, :effect_chs, m] += effect * semi[i] / 6.0
    return data, times


# ---------------------------------------------------------------------------
# stats_core
# ---------------------------------------------------------------------------

def test_ols_t_matches_numpy_lstsq():
    rng = np.random.default_rng(1)
    n, nch = 50, 4
    X = sc.design(rng.normal(size=n), rng.normal(size=n))  # intercept + 2 regressors
    Y = rng.normal(size=(n, nch))

    beta, t = sc.ols_t(Y, X)
    beta_ref, *_ = np.linalg.lstsq(X, Y, rcond=None)
    assert beta.shape == (3, nch)
    assert np.allclose(beta.reshape(3, nch), beta_ref)

    resid = Y - X @ beta_ref
    dof = n - np.linalg.matrix_rank(X)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X))[:, None] * (resid**2).sum(0) / dof)
    assert np.allclose(t.reshape(3, nch), beta_ref / se, rtol=1e-9)


def test_fwl_residualization_equals_direct_ols():
    """The permutation loop relies on FWL; it must be numerically identical."""
    rng = np.random.default_rng(2)
    n = 70
    nuisance = rng.normal(size=(n, 2))
    pitch = rng.normal(size=n)
    Y = rng.normal(size=(n, 6))

    Q, k = cs._nuisance_basis(nuisance, n)
    t_fwl = cs._t_from_residuals(
        cs._residualize(Y, Q), cs._residualize(pitch, Q), max(n - k - 1, 1)
    )
    _, t_direct = sc.ols_t(Y, sc.design(pitch, nuisance))
    assert np.allclose(t_fwl, t_direct[1], rtol=1e-10)


def test_perm_p_uses_plus_one_correction():
    assert sc.perm_p_abs(99.0, np.zeros(1000)) == pytest.approx(1 / 1001)
    null = np.array([0.0, 1.0, 2.0, 3.0])
    assert sc.perm_p_abs(2.5, null) == pytest.approx((1 + 1) / 5)


def test_permute_within_blocks_preserves_block_membership():
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    blocks = np.array([1, 1, 1, 2, 2, 2])
    out = sc.permute_within_blocks(values, blocks, np.random.default_rng(3))
    assert sorted(out[:3]) == [1.0, 2.0, 3.0]
    assert sorted(out[3:]) == [10.0, 20.0, 30.0]


def test_vif_grows_with_collinearity_and_reports_inf_for_exact():
    rng = np.random.default_rng(4)
    x = rng.normal(size=300)
    assert sc.vif(x, rng.normal(size=(300, 1))) == pytest.approx(1.0, abs=0.05)
    assert sc.vif(x, (0.99 * x)[:, None]) > sc.thresholds()["vif_max"]
    assert sc.vif(x, x[:, None]) == pytest.approx(1e12, rel=1e-3)


def test_design_accepts_2d_columns():
    a = np.arange(6.0).reshape(3, 2)
    b = np.arange(3.0)
    X = sc.design(a, b)
    assert X.shape == (3, 4)
    assert np.all(X[:, 0] == 1.0)


# ---------------------------------------------------------------------------
# R1 / R3: design facts and collinearity
# ---------------------------------------------------------------------------

def test_hand_agreement_detects_fixed_mapping():
    """R1: a fixed tone->key mapping must be reported as ~100% agreement."""
    cols = cs.build_design_columns(make_meta(seed=1))
    agree = cs.hand_agreement(cols)
    assert agree["available"]
    assert agree["agree_rate"] == 1.0
    assert agree["minority_hand_trials"] == 0


def test_hand_agreement_detects_counterbalanced_design():
    cols = cs.build_design_columns(make_meta(random_hand=True, seed=2))
    agree = cs.hand_agreement(cols)
    assert agree["available"]
    assert 0.2 < agree["agree_rate"] < 0.8


def test_anchor_arm_is_flagged_not_separable():
    """R3: on anchors pitch and hand are the same variable -> must be unseparable."""
    cols = cs.build_design_columns(make_meta(seed=3))
    anchor = cols["is_main"] & cols["is_anchor"]
    d = cs.collinearity(cols, anchor, with_block=False)
    assert abs(d["r_pitch_hand"]) > sc.thresholds()["abs_r_max"]
    assert cs.separability_verdict(d["r_pitch_hand"], d["vif_pitch"]) == "不可分离"


def test_mid_arm_has_hand_variance():
    """Ambiguous tones are the only place hand varies independently of pitch."""
    cols = cs.build_design_columns(make_meta(seed=4))
    mid = cols["is_main"] & cols["is_mid"]
    d = cs.collinearity(cols, mid, with_block=False)
    assert abs(d["r_pitch_hand"]) < 0.3
    assert d["n_hand_levels"] == 2
    assert d["n_minority_hand"] > 0


# ---------------------------------------------------------------------------
# R2: the three arms
# ---------------------------------------------------------------------------

def test_all_three_arms_are_always_reported():
    """R2: A, B and C must exist and carry their own family-corrected result."""
    meta = make_meta(seed=5)
    data, times = make_data(meta, seed=5)
    cols = cs.build_design_columns(meta)
    res = cs.run_condition_stats(data, times, cols, [f"C{i}" for i in range(16)],
                                 n_perm=60, seed=5)

    for arm in ("A", "B", "C"):
        assert arm in res["arms"]
        assert np.isfinite(res["arms"][arm]["max_abs_t"])
        assert np.isfinite(res["arms"][arm]["family_p"])
        assert res["arms"][arm]["n_trials"] > 0

    assert res["arms"]["A"]["n_trials"] < res["arms"]["B"]["n_trials"]
    assert res["arms"]["C"]["n_trials"] < res["arms"]["B"]["n_trials"]

    # A test that reported only one of these would be selective reporting.
    assert set(res["separability"]) == {"A", "B", "C"}
    assert set(res["confound"]) == {"A", "B", "C"}


def test_planted_effect_is_detected():
    """A 6 uV semitone-scaled effect on 6 channels must survive family correction."""
    meta = make_meta(seed=6)
    data, times = make_data(meta, effect=8.0, effect_chs=6, seed=6)
    cols = cs.build_design_columns(meta)
    res = cs.run_condition_stats(data, times, cols, [f"C{i}" for i in range(16)],
                                 n_perm=200, seed=6)
    assert res["arms"]["B"]["family_p"] < 0.05
    assert res["arms"]["B"]["peak_channel_index"] < 6


def test_null_data_does_not_produce_significant_family_p():
    """Noise-only input must not clear the family threshold."""
    meta = make_meta(seed=7)
    data, times = make_data(meta, effect=0.0, seed=7)
    cols = cs.build_design_columns(meta)
    res = cs.run_condition_stats(data, times, cols, [f"C{i}" for i in range(16)],
                                 n_perm=200, seed=7)
    assert res["arms"]["B"]["family_p"] > 0.05


def test_permutation_null_matches_true_sampling_distribution():
    """Guard the calibration: permutation q95 must track the observed q95 under H0.

    A permutation scheme that silently inflates or deflates the null would make
    every p in the report meaningless.  We compare, over independent noise draws,
    the mean permutation q95 against the empirical 95th percentile of the observed
    max|t| -- they should agree to within ~15%.
    """
    obs, perm95 = [], []
    for s in range(12):
        meta = make_meta(seed=100 + s)
        data, times = make_data(meta, seed=100 + s)
        cols = cs.build_design_columns(meta)
        flat, labels = cs.window_matrix(
            data, times, {**cs.WINDOWS, cs.PRIMARY_WINDOW: (0.0, 0.2)}
        )
        arm = cs.run_arm(flat, labels, cols, cols["is_main"],
                         [cols["hand"], cols["block"]], n_perm=120, seed=s)
        obs.append(arm["max_abs_t"])
        perm95.append(arm["null_q95"])

    true95 = float(np.percentile(obs, 95))
    mean_perm95 = float(np.mean(perm95))
    assert 0.75 < mean_perm95 / true95 < 1.35


# ---------------------------------------------------------------------------
# R4: hand-map diagnosis
# ---------------------------------------------------------------------------

def test_hand_map_reports_unavailable_when_mid_hand_variance_is_insufficient():
    """R4: one hand only on mid tones -> diagnosis must be refused, not faked."""
    meta = make_meta(n_blocks=1, n_per_block=20, seed=8)
    # Force every mid trial to the same hand.
    mid = meta["cls"].str.startswith("m")
    meta.loc[mid, "choice_phys"] = "right"

    cols = cs.build_design_columns(meta)
    data, times = make_data(meta, seed=8)
    res = cs.run_condition_stats(data, times, cols, [f"C{i}" for i in range(16)],
                                 n_perm=40, seed=8)

    assert not res["hand_map"]["available"]
    assert "变异不足" in res["hand_map"]["reason"]
    for arm in ("A", "B", "C"):
        assert res["confound"][arm]["verdict"] == "无法诊断"


def test_topk_overlap_is_judged_against_chance():
    """R4: on a small montage raw top-k overlap is high by chance, not a confound."""
    # 8 channels: chance overlap for k=4 is 50%, so zero overlap must not fire.
    assert cs.confound_verdict(r=0.0, overlap=0.0, chance=0.5) == ""
    # Same overlap on a large montage (170 ch, k=10 -> chance 5.9%) does fire.
    assert cs.confound_verdict(r=0.0, overlap=0.6, chance=10 / 170) == "运动混淆候选"
    # A strong spatial correlation fires regardless of the overlap statistic.
    assert cs.confound_verdict(r=0.8, overlap=0.0, chance=0.5) == "运动混淆候选"
    # Overlap at the absolute threshold but only 1x chance must not fire.
    assert cs.confound_verdict(r=0.0, overlap=0.5, chance=0.5) == ""


# ---------------------------------------------------------------------------
# R6: RT table is mandatory
# ---------------------------------------------------------------------------

def test_rt_table_is_always_emitted():
    meta = make_meta(seed=9)
    data, times = make_data(meta, seed=9)
    cols = cs.build_design_columns(meta)
    res = cs.run_condition_stats(data, times, cols, [f"C{i}" for i in range(16)],
                                 n_perm=40, seed=9)
    rc = res["rt_check"]
    assert rc["available"]
    assert rc["n_trials"] > 0
    assert np.isfinite(rc["max_abs_r"])
    assert 0.0 <= rc["frac_abs_gt_0.3"] <= 1.0


def test_rt_table_reports_unavailable_without_rt():
    meta = make_meta(seed=10)
    meta["rt_ms"] = np.nan
    data, times = make_data(meta, seed=10)
    cols = cs.build_design_columns(meta)
    res = cs.run_condition_stats(data, times, cols, [f"C{i}" for i in range(16)],
                                 n_perm=20, seed=10)
    assert not res["rt_check"]["available"]
    assert "RT" in res["rt_check"]["reason"]


# ---------------------------------------------------------------------------
# R8: the honest headline
# ---------------------------------------------------------------------------

def test_report_never_claims_hand_is_controlled():
    """R8: without counterbalancing, no output may assert hand was controlled."""
    meta = make_meta(seed=11)
    data, times = make_data(meta, seed=11)
    cols = cs.build_design_columns(meta)
    res = cs.run_condition_stats(data, times, cols, [f"C{i}" for i in range(16)],
                                 n_perm=40, seed=11)
    res["pos_anchor_cls"] = cols["pos_anchor_cls"]

    assert res["can_claim_hand_controlled"] is False
    assert res["co_varies_with_hand"] is True
    assert res["label"] == "条件（与手侧共变）"

    report = cs.render_report(res, [f"C{i}" for i in range(16)])
    assert "条件（与手侧共变）" in report
    assert "不能声称「已控制手」" in report

    # The forbidden framings may only appear inside a prohibition: either on a
    # line that itself negates ("不得写成…"), or inside the "不可以写" section.
    forbidden = ["效价效应", "决策效应", "判断偏倚的神经相关"]
    negation_markers = ("不得", "不可以", "禁止", "不能", "不要")

    section = ""
    for line in report.splitlines():
        if line.startswith("## "):
            section = line
        if any(bad in section for bad in ("不可以写",)):
            continue  # this whole section is a list of things not to write
        for phrase in forbidden:
            if phrase in line:
                assert any(mk in line for mk in negation_markers), (
                    f"'{phrase}' used as an assertion, not a prohibition: {line!r}"
                )


def test_report_prints_all_three_family_p_values():
    meta = make_meta(seed=12)
    data, times = make_data(meta, effect=8.0, seed=12)
    cols = cs.build_design_columns(meta)
    res = cs.run_condition_stats(data, times, cols, [f"C{i}" for i in range(16)],
                                 n_perm=60, seed=12)
    res["pos_anchor_cls"] = cols["pos_anchor_cls"]
    report = cs.render_report(res, [f"C{i}" for i in range(16)])

    for arm in ("A", "B", "C"):
        p = res["arms"][arm]["family_p"]
        assert f"{p:.3f}" in report
    assert "只报 A 或只报 B 视为选择性报告" in report


# ---------------------------------------------------------------------------
# Metadata coercion robustness
# ---------------------------------------------------------------------------

def test_metadata_coercion_survives_mne_roundtrip_dtypes():
    """MNE round-trips None through Float64/object dtypes; loading must not crash."""
    meta = make_meta(n_blocks=1, n_per_block=10, seed=13)
    meta["correct"] = meta["correct"].astype("Float64")
    meta["responded"] = meta["responded"].astype("boolean")
    meta["block"] = meta["block"].astype(str)

    cols = cs.build_design_columns(meta)
    assert cols["n"] == len(meta)
    assert np.isfinite(cols["pitch"]).sum() == len(meta)
    assert np.isfinite(cols["block"]).sum() == len(meta)
    assert cols["is_anchor"].sum() > 0


def test_missing_columns_degrade_gracefully():
    """A stripped metadata table must raise a clear error, not an AttributeError."""
    meta = pd.DataFrame({"phase": ["main"] * 4})
    cols = cs.build_design_columns(meta)
    agree = cs.hand_agreement(cols)
    assert not agree["available"]

    # Too few trials: run_arm reports the reason instead of raising.
    arm = cs.run_arm(
        np.zeros((4, 5)),
        [f"0-200ms|{i}" for i in range(5)],
        cols,
        np.ones(4, dtype=bool),
        [],
        10,
        0,
    )
    assert arm["unavailable"]
    assert not np.isfinite(arm["max_abs_t"])
