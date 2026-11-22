"""Valence-agnostic Judgement Bias behavioral analysis & standardized reporting.

Never hardcodes whether high/low pitch or left/right key corresponds to positive/negative!
Instead, dynamically infers anchor valence from each trial's `eff` ("pos" vs "neg") and
maps the 5-tone continuum monotonically onto:
    rank 0: Neg Anchor (eff == "neg")
    rank 1: Near-Neg   (ambiguous tone adjacent to Neg Anchor)
    rank 2: Mid        (m2, center ambiguous tone)
    rank 3: Near-Pos   (ambiguous tone adjacent to Pos Anchor)
    rank 4: Pos Anchor (eff == "pos")
so every subject is directly comparable regardless of counterbalancing (`cond`/`combo`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math
import os
import numpy as np

# Ensure matplotlib has a writable config dir inside sandboxed/non-standard environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_jbs")


CANONICAL_LABELS = ["Neg Anchor", "Near-Neg", "Mid", "Near-Pos", "Pos Anchor"]
CANONICAL_KEYS = ["neg_anchor", "near_neg", "mid", "near_pos", "pos_anchor"]


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam / scipy fallback)."""
    from scipy.stats import norm

    return float(norm.ppf(p))


def infer_valence_mapping(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Dynamically determine whether 'low' or 'high' is the positive anchor."""
    pos_cls = None
    neg_cls = None
    pos_hz = None
    neg_hz = None
    pos_key = None
    neg_key = None
    for tr in trials:
        eff = tr.get("eff")
        if eff == "pos" and pos_cls is None:
            pos_cls = tr.get("cls")
            pos_hz = tr.get("hz")
        elif eff == "neg" and neg_cls is None:
            neg_cls = tr.get("cls")
            neg_hz = tr.get("hz")
        if tr.get("choice_sem") == "pos" and tr.get("choice_phys") and pos_key is None:
            pos_key = tr.get("choice_phys")
        if tr.get("choice_sem") == "neg" and tr.get("choice_phys") and neg_key is None:
            neg_key = tr.get("choice_phys")

    # Standard 5-tone physical order: low < m1 < m2 < m3 < high
    if pos_cls == "high" or (pos_cls is None and neg_cls == "low"):
        pos_cls = "high"
        neg_cls = neg_cls or "low"
        cls_to_rank = {"low": 0, "m1": 1, "m2": 2, "m3": 3, "high": 4}
    elif pos_cls == "low" or (pos_cls is None and neg_cls == "high"):
        pos_cls = "low"
        neg_cls = neg_cls or "high"
        cls_to_rank = {"high": 0, "m3": 1, "m2": 2, "m1": 3, "low": 4}
    else:
        raise ValueError(
            "无法从行为学 JSON 的 `eff` 字段判断正/负性锚点（高音或低音），拒绝猜测默认映射，请检查数据完整性。"
        )

    rank_to_cls = {v: k for k, v in cls_to_rank.items()}
    return {
        "pos_anchor_cls": pos_cls,
        "neg_anchor_cls": neg_cls,
        "pos_anchor_hz": pos_hz,
        "neg_anchor_hz": neg_hz,
        "pos_key": pos_key,
        "neg_key": neg_key,
        "cls_to_rank": cls_to_rank,
        "rank_to_cls": rank_to_cls,
    }


def analyze_behavior(
    behavior_json_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run full standardized Judgement Bias behavioral analysis and optionally render plots + CSV."""
    with open(behavior_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    participant = raw.get("participant", {})
    config = raw.get("config", {})
    trials = raw.get("trials", [])
    mood_probes = raw.get("moodProbes", [])

    vmap = infer_valence_mapping(trials)
    cls_to_rank = vmap["cls_to_rank"]

    practice_trials = [t for t in trials if t.get("phase") == "practice"]
    main_trials = [t for t in trials if t.get("phase") == "main"]

    # Enrich each main trial with canonical valence rank (0=Neg Anchor .. 4=Pos Anchor)
    enriched_main: list[dict[str, Any]] = []
    for t in main_trials:
        rank = cls_to_rank.get(t.get("cls"))
        enriched_main.append(
            {
                **t,
                "valence_rank": rank,
                "canonical_tone": CANONICAL_KEYS[rank] if rank is not None else "unknown",
                "is_ambiguous": bool(t.get("nofb")) or t.get("eff") is None,
            }
        )

    n_main = len(enriched_main)
    responded_main = [t for t in enriched_main if t.get("responded")]
    omission_rate = (n_main - len(responded_main)) / n_main if n_main else 0.0

    # Anchors & SDT (log-linear Hautus 0.5 correction when hit/fa hits 0 or 1)
    pos_anc = [t for t in responded_main if t.get("eff") == "pos"]
    neg_anc = [t for t in responded_main if t.get("eff") == "neg"]
    n_pos = len(pos_anc)
    n_neg = len(neg_anc)
    hits = sum(1 for t in pos_anc if t.get("choice_sem") == "pos")
    fas = sum(1 for t in neg_anc if t.get("choice_sem") == "pos")
    acc_pos = hits / n_pos if n_pos else None
    acc_neg = (n_neg - fas) / n_neg if n_neg else None

    hit_adj = (hits + 0.5) / (n_pos + 1.0) if n_pos else 0.5
    fa_adj = (fas + 0.5) / (n_neg + 1.0) if n_neg else 0.5
    dprime = round(_norm_ppf(hit_adj) - _norm_ppf(fa_adj), 4)
    criterion = round(-0.5 * (_norm_ppf(hit_adj) + _norm_ppf(fa_adj)), 4)

    # Ambiguous mid tones
    mids = [t for t in responded_main if t["is_ambiguous"]]
    n_mids = len(mids)
    p_pos_mids = sum(1 for t in mids if t.get("choice_sem") == "pos") / n_mids if n_mids else None
    bias_score = round(2.0 * p_pos_mids - 1.0, 4) if p_pos_mids is not None else None

    # Psychometric curve (canonical order: Neg Anchor -> Near-Neg -> Mid -> Near-Pos -> Pos Anchor)
    psychometric: list[dict[str, Any]] = []
    for rank in range(5):
        sub = [t for t in responded_main if t["valence_rank"] == rank]
        rts = [float(t["rt_ms"]) for t in sub if t.get("rt_ms") is not None]
        p_pos = sum(1 for t in sub if t.get("choice_sem") == "pos") / len(sub) if sub else None
        hz = sub[0].get("hz") if sub else None
        psychometric.append(
            {
                "rank": rank,
                "canonical_key": CANONICAL_KEYS[rank],
                "label": CANONICAL_LABELS[rank],
                "orig_cls": vmap["rank_to_cls"].get(rank),
                "hz": hz,
                "n": len(sub),
                "p_pos": round(p_pos, 4) if p_pos is not None else None,
                "rt_mean_ms": round(float(np.mean(rts)), 1) if rts else None,
                "rt_sd_ms": round(float(np.std(rts)), 1) if rts else None,
            }
        )

    # RT comparison: Anchors vs Ambiguous
    rt_anchor = [float(t["rt_ms"]) for t in responded_main if not t["is_ambiguous"] and t.get("rt_ms") is not None]
    rt_mids = [float(t["rt_ms"]) for t in mids if t.get("rt_ms") is not None]
    rt_anchor_mean = round(float(np.mean(rt_anchor)), 1) if rt_anchor else None
    rt_mids_mean = round(float(np.mean(rt_mids)), 1) if rt_mids else None
    rt_conflict_ms = (
        round(rt_mids_mean - rt_anchor_mean, 1)
        if (rt_anchor_mean is not None and rt_mids_mean is not None)
        else None
    )

    # Block-by-block metrics
    blocks = sorted({t["block"] for t in enriched_main if t.get("block") is not None})
    by_block: list[dict[str, Any]] = []
    for blk in blocks:
        bt = [t for t in responded_main if t.get("block") == blk]
        bm = [t for t in bt if t["is_ambiguous"]]
        ba = [t for t in bt if not t["is_ambiguous"]]
        b_ppos = sum(1 for t in bm if t.get("choice_sem") == "pos") / len(bm) if bm else None
        b_acc = sum(1 for t in ba if t.get("correct") == 1) / len(ba) if ba else None
        b_rts = [float(t["rt_ms"]) for t in bt if t.get("rt_ms") is not None]
        by_block.append(
            {
                "block": blk,
                "n_trials": len(bt),
                "anchor_acc": round(b_acc, 4) if b_acc is not None else None,
                "mids_p_pos": round(b_ppos, 4) if b_ppos is not None else None,
                "mids_bias_score": round(2.0 * b_ppos - 1.0, 4) if b_ppos is not None else None,
                "rt_mean_ms": round(float(np.mean(b_rts)), 1) if b_rts else None,
            }
        )

    # Mood trajectory
    mood_series = [
        {
            "probe_index": m.get("probe_index"),
            "checkpoint": m.get("events_json", {}).get("checkpoint", f"probe_{m.get('probe_index')}"),
            "after_trial": m.get("after_trial"),
            "valence": m.get("valence"),
            "rt_ms": m.get("rt_ms"),
        }
        for m in mood_probes
    ]
    mood_vals = [m["valence"] for m in mood_series if m["valence"] is not None]

    # Practice QC
    prac_acc = (
        sum(1 for t in practice_trials if t.get("correct") == 1) / len(practice_trials)
        if practice_trials
        else None
    )

    # Automatic QC flags
    qc_flags: list[str] = []
    if omission_rate > 0.10:
        qc_flags.append(f"遗漏率偏高 ({omission_rate*100:.1f}%)")
    if dprime < 1.5:
        qc_flags.append(f"锚点辨别力偏低 (d'={dprime:.2f})")
    if (acc_pos is not None and acc_pos < 0.75) or (acc_neg is not None and acc_neg < 0.75):
        qc_flags.append("锚点正确率 < 75%")

    result: dict[str, Any] = {
        "participant": participant,
        "valence_mapping": vmap,
        "metrics": {
            "n_practice": len(practice_trials),
            "practice_acc": round(prac_acc, 4) if prac_acc is not None else None,
            "n_main": n_main,
            "n_responded": len(responded_main),
            "omission_rate": round(omission_rate, 4),
            "acc_pos_anchor": round(acc_pos, 4) if acc_pos is not None else None,
            "acc_neg_anchor": round(acc_neg, 4) if acc_neg is not None else None,
            "dprime": dprime,
            "criterion": criterion,
            "p_pos_on_mids": round(p_pos_mids, 4) if p_pos_mids is not None else None,
            "bias_score": bias_score,
            "rt_anchor_mean_ms": rt_anchor_mean,
            "rt_mids_mean_ms": rt_mids_mean,
            "rt_conflict_ms": rt_conflict_ms,
            "mood_mean": round(float(np.mean(mood_vals)), 2) if mood_vals else None,
            "mood_delta": round(float(mood_vals[-1] - mood_vals[0]), 2) if len(mood_vals) >= 2 else None,
            "balance": raw.get("summary", {}).get("balance"),
            "duration_min": raw.get("summary", {}).get("durationMin"),
        },
        "psychometric": psychometric,
        "by_block": by_block,
        "mood_series": mood_series,
        "qc_pass": len(qc_flags) == 0,
        "qc_flags": qc_flags,
        "artifacts": {},
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Save summary JSON
        (out / "behavior_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Save enriched trial-level CSV
        trial_csv = out / "behavior_trials.csv"
        fieldnames = [
            "idx", "block", "tib", "cls", "canonical_tone", "valence_rank", "hz",
            "eff", "is_ambiguous", "responded", "choice_sem", "choice_phys",
            "correct", "rt_ms", "money", "fb_type", "tone_onset_wall",
        ]
        with open(trial_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for t in enriched_main:
                w.writerow(t)
        # Render 2x2 diagnostic figure
        fig_path = out / "behavior_overview.png"
        _render_behavior_figure(result, fig_path)
        result["artifacts"] = {
            "summary_json": str(out / "behavior_summary.json"),
            "trials_csv": str(trial_csv),
            "overview_png": str(fig_path),
        }

    return result


def _render_behavior_figure(res: dict[str, Any], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), dpi=150, facecolor="#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")
        ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

    # Panel A: Psychometric Curve (Neg Anchor -> Pos Anchor)
    ax = axes[0, 0]
    ps = res["psychometric"]
    xs = [p["rank"] for p in ps]
    ys = [p["p_pos"] if p["p_pos"] is not None else np.nan for p in ps]
    xlabels = [f"{p['label']}\n({p['orig_cls']})" for p in ps]
    ax.plot(xs, ys, "-o", color="#38bdf8", linewidth=2.4, markersize=7)
    ax.axhline(0.5, color="#f59e0b", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvspan(0.7, 3.3, color="#38bdf8", alpha=0.08)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("P('Positive' Response)")
    bs = res["metrics"].get("bias_score")
    ax.set_title(f"A. Psychometric Curve (Bias = {bs:+.3f})" if bs is not None else "A. Psychometric Curve")

    # Panel B: Reaction Time across 5 tones
    ax = axes[0, 1]
    rts = [p["rt_mean_ms"] if p["rt_mean_ms"] is not None else 0 for p in ps]
    sds = [p["rt_sd_ms"] if p["rt_sd_ms"] is not None else 0 for p in ps]
    colors = ["#64748b", "#38bdf8", "#0ea5e9", "#38bdf8", "#64748b"]
    ax.bar(xs, rts, yerr=sds, color=colors, capsize=4, alpha=0.88, error_kw={"ecolor": "#cbd5e1", "lw": 1.1})
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Reaction Time (ms)")
    ax.set_title("B. Reaction Time by Tone (Mean ± SD)")

    # Panel C: Block-by-Block Bias & Anchor Accuracy
    ax = axes[1, 0]
    blks = res["by_block"]
    bx = [b["block"] for b in blks]
    b_bias = [b["mids_bias_score"] for b in blks]
    b_acc = [b["anchor_acc"] for b in blks]
    ax.plot(bx, b_bias, "-s", color="#34d399", lw=2.2, label="Mids Bias Score (-1..+1)")
    ax.plot(bx, b_acc, "--^", color="#fbbf24", lw=2.0, label="Anchor Accuracy (0..1)")
    ax.axhline(0, color="#94a3b8", lw=0.8, alpha=0.5)
    ax.set_xticks(bx)
    ax.set_xticklabels([f"Block {b}" for b in bx])
    ax.set_ylim(-1.05, 1.1)
    ax.set_title("C. Stability Across Blocks")
    ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)

    # Panel D: Mood Probes (VAS-11)
    ax = axes[1, 1]
    ms = res["mood_series"]
    mx = list(range(len(ms)))
    my = [m["valence"] for m in ms]
    ml = [m["checkpoint"].replace("_", "\n") for m in ms]
    ax.plot(mx, my, "-o", color="#f472b6", lw=2.2, markersize=6)
    ax.set_xticks(mx)
    ax.set_xticklabels(ml, fontsize=7.5)
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Mood Valence (VAS 0-10)")
    ax.set_title("D. Mood Probe Trajectory")

    pid = res["participant"].get("name", "Subject")
    cond = res["participant"].get("cond", "?")
    dp = res["metrics"].get("dprime", 0)
    fig.suptitle(
        f"Judgement Bias Behavioral Report — {pid} (Cond {cond} | d' = {dp:.2f} | Pos Anchor = {res['valence_mapping']['pos_anchor_cls']})",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
