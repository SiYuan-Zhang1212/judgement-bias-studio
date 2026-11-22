"""Hardware parallel-port trigger decoding, EDF+D continuity audit, and wall-clock fallback sync.

Hardware protocol (Nihon Kohden NKC-EEG-1200A EDF+D):
- 6-bit parallel port bus on POL DC02..DC07 (bit 0 = DC02 .. bit 5 = DC07, >1.5 V high).
- Tone onset codes:
    Practice: 31 (low), 33 (high)
    Main:     21 (low), 23 (m1), 25 (m2), 27 (m3), 29 (high)
- Response (keydown) codes:
    35 (left physical key), 37 (right physical key)

Safety gates:
1. EDF+D discontinuity check: scans the TAL onset timestamp of every data record directly
   from the EDF binary stream. If any recording pause/gap is detected, flags `has_discontinuity=True`
   so continuous wall-clock interpolation cannot silently drift across pauses.
2. Minimum hardware match gate: requires >= 2 matched tone pulses and >= `min_hw_match_rate`
   (default 80%) hardware tone match before allowing wall-clock recovery of dropped triggers.
   Never fabricates timestamps with intercept = 0 when hardware triggers are missing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re
import numpy as np


TONE_CODES = {21, 23, 25, 27, 29, 31, 33}
RESP_CODES = {35: "left", 37: "right"}
_TAL_ONSET_RE = re.compile(rb"([+-]\d+(?:\.\d+)?)\x14")


def parse_iso_ts(wall: str | None) -> float | None:
    if not wall:
        return None
    try:
        return datetime.fromisoformat(wall.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def load_edf_raw(edf_path: str | Path, preload: bool = False):
    """Load Nihon Kohden EDF with automatic GB18030 -> UTF-8 -> Latin1 fallback."""
    import mne

    last_err: Exception | None = None
    for enc in ("gb18030", "utf-8", "latin1"):
        try:
            return mne.io.read_raw_edf(
                str(edf_path), preload=preload, encoding=enc, verbose="ERROR"
            )
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to read EDF {edf_path}: {last_err}")


def check_edf_discontinuities(
    edf_path: str | Path,
    tol_s: float = 0.005,
) -> dict[str, Any]:
    """Scan all EDF+D record onset timestamps directly from the TAL channel (fast, <0.3s)."""
    p = Path(edf_path)
    with open(p, "rb") as f:
        hdr = f.read(256)
        if len(hdr) < 256:
            return {"checked": False, "warning": "Invalid EDF header"}
        reserved = hdr[192:236].decode("latin1").strip()
        hdr_bytes = int(hdr[184:192].decode("latin1").strip())
        n_records = int(hdr[236:244].decode("latin1").strip())
        dur_record = float(hdr[244:252].decode("latin1").strip())
        ns = int(hdr[252:256].decode("latin1").strip())
        sig_hdr = f.read(ns * 256)
        labels = [sig_hdr[i * 16 : (i + 1) * 16].decode("latin1").strip() for i in range(ns)]
        off = ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
        n_samps = [int(sig_hdr[off + i * 8 : off + (i + 1) * 8].decode("latin1").strip()) for i in range(ns)]

        ann_idx = next((i for i, l in enumerate(labels) if l == "EDF Annotations"), None)
        if ann_idx is None or n_records <= 1 or dur_record <= 0:
            return {
                "checked": True,
                "edf_subtype": reserved,
                "n_records": n_records,
                "dur_record_s": dur_record,
                "has_discontinuity": False,
                "n_gaps": 0,
                "gaps": [],
            }

        rec_bytes = sum(n_samps) * 2
        ann_offset_in_rec = sum(n_samps[:ann_idx]) * 2
        ann_len = n_samps[ann_idx] * 2

        prev_onset: float | None = None
        gaps: list[dict[str, Any]] = []
        for r in range(n_records):
            f.seek(hdr_bytes + r * rec_bytes + ann_offset_in_rec)
            chunk = f.read(min(ann_len, 48))
            m = _TAL_ONSET_RE.match(chunk)
            if not m:
                continue
            onset = float(m.group(1))
            if prev_onset is not None:
                expected = prev_onset + dur_record
                diff = onset - expected
                if abs(diff) > tol_s:
                    gaps.append(
                        {
                            "record_index": r,
                            "prev_onset_s": round(prev_onset, 4),
                            "actual_onset_s": round(onset, 4),
                            "gap_s": round(diff, 4),
                        }
                    )
            prev_onset = onset

    return {
        "checked": True,
        "edf_subtype": reserved,
        "n_records": n_records,
        "dur_record_s": dur_record,
        "has_discontinuity": len(gaps) > 0,
        "n_gaps": len(gaps),
        "gaps": gaps[:20],
    }


def decode_dc_pulses(
    edf_path: str | Path,
    dc_channels: list[str] | None = None,
    voltage_threshold: float = 1.5,
    settle_ms: float = 4.0,
    min_gap_ms: float = 30.0,
    ignore_initial_s: float = 5.0,
) -> dict[str, Any]:
    """Decode 6-bit parallel-port trigger pulses from POL DC02..DC07."""
    raw = load_edf_raw(edf_path, preload=False)
    sfreq = float(raw.info["sfreq"])
    duration_s = float(raw.times[-1]) if len(raw.times) else 0.0
    continuity = check_edf_discontinuities(edf_path)

    if dc_channels is None:
        dc_channels = [f"POL DC{i:02d}" for i in range(2, 8)]
    available = [c for c in dc_channels if c in raw.ch_names]
    if len(available) < 6:
        alt = [f"DC{i:02d}" for i in range(2, 8)]
        available = [c for c in alt if c in raw.ch_names]

    if len(available) < 6:
        return {
            "sfreq": sfreq,
            "duration_s": duration_s,
            "meas_date": str(raw.info.get("meas_date")),
            "dc_channels": available,
            "continuity": continuity,
            "pulses": [],
            "warning": f"未找到完整的 6 路并口触发通道 (POL DC02..DC07)，仅找到 {len(available)} 路。",
        }

    dc = raw.get_data(picks=available)
    medians = np.median(dc, axis=1, keepdims=True)
    bits = ((dc - medians) > voltage_threshold).astype(np.int32)

    code = np.zeros(bits.shape[1], dtype=np.int32)
    for b in range(6):
        code += bits[b] << b

    nz = code > 0
    starts = np.where((~nz[:-1]) & nz[1:])[0] + 1
    settle_samp = max(1, int(round(settle_ms * 1e-3 * sfreq)))
    min_gap_samp = max(1, int(round(min_gap_ms * 1e-3 * sfreq)))

    pulses: list[dict[str, Any]] = []
    last_s = -min_gap_samp * 2
    n_tot = len(code)
    for s in starts:
        if s - last_s < min_gap_samp:
            continue
        t_s = float(s / sfreq)
        if t_s < ignore_initial_s:
            continue
        sample_idx = min(int(s) + settle_samp, n_tot - 1)
        val = int(code[sample_idx])
        if val == 0:
            val = int(code[int(s)])
        if val == 0:
            continue
        e = int(s)
        while e < n_tot and code[e] > 0:
            e += 1
        dur_ms = round((e - int(s)) / sfreq * 1000.0, 2)
        kind = "tone" if val in TONE_CODES else ("response" if val in RESP_CODES else "other")
        pulses.append(
            {
                "sample": int(s),
                "time_s": round(t_s, 6),
                "code": val,
                "dur_ms": dur_ms,
                "kind": kind,
            }
        )
        last_s = int(s)

    return {
        "sfreq": sfreq,
        "duration_s": round(duration_s, 3),
        "meas_date": str(raw.info.get("meas_date")),
        "dc_channels": available,
        "continuity": continuity,
        "pulses": pulses,
    }


def _fit_robust_clock(
    wall_t: np.ndarray, edf_t: np.ndarray
) -> tuple[float, float]:
    """Fit edf_t = a * wall_t + b robustly via Huber-like IRLS."""
    if len(wall_t) == 0:
        return 1.0, 0.0
    if len(wall_t) == 1:
        return 1.0, float(edf_t[0] - wall_t[0])
    a, b = 1.0, float(np.median(edf_t - wall_t))
    for _ in range(5):
        resid = edf_t - (a * wall_t + b)
        mad = float(np.median(np.abs(resid))) + 1e-6
        inliers = np.abs(resid) <= max(0.08, 4.0 * 1.4826 * mad)
        if int(inliers.sum()) >= 2:
            a, b = np.polyfit(wall_t[inliers], edf_t[inliers], 1)
            a, b = float(a), float(b)
    return a, b


def align_triggers_with_behavior(
    trials: list[dict[str, Any]],
    pulses: list[dict[str, Any]],
    sfreq: float,
    max_match_err_s: float = 0.25,
    min_hw_match_rate: float = 0.80,
    continuity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Match decoded hardware pulses to behavioral trials and safely recover dropped triggers.

    Safety rules:
    - If fewer than 2 hardware tone pulses match (e.g. disconnected trigger cable or wrong DC pins),
      DO NOT fabricate wall-clock timestamps with intercept=0; mark `sync_ok=False` and leave trials
      `unmatched`.
    - If `tone_hw_match_rate < min_hw_match_rate` (default 80%) or `continuity.has_discontinuity`
      is True, mark `sync_ok=False` with a clear `sync_error` so downstream preprocessing is blocked.
    """
    n_trials = len(trials)
    if n_trials == 0:
        return {
            "aligned_trials": [],
            "clock_model": None,
            "summary": {
                "n_trials": 0,
                "sync_ok": False,
                "sync_error": "行为学 JSON 中没有任何试次记录。",
            },
        }

    tone_pulses = [p for p in pulses if p["kind"] == "tone"]
    resp_pulses = [p for p in pulses if p["kind"] == "response"]

    raw_walls = [parse_iso_ts(tr.get("tone_onset_wall")) for tr in trials]
    valid_walls = [w for w in raw_walls if w is not None]
    t0_wall = valid_walls[0] if valid_walls else 0.0
    wall_rel = np.array(
        [(w - t0_wall) if w is not None else np.nan for w in raw_walls], dtype=float
    )
    expected_codes = [
        tr.get("extra_json", {}).get("trigger", {}).get("tone_code") for tr in trials
    ]

    # Step 1: Consensus offset voting across same-code hardware pulses
    offset_votes: list[float] = []
    if tone_pulses and valid_walls:
        sample_idxs = np.linspace(0, n_trials - 1, min(n_trials, 60), dtype=int)
        for ti in sample_idxs:
            if np.isnan(wall_rel[ti]):
                continue
            ec = expected_codes[ti]
            for p in tone_pulses:
                if ec is None or p["code"] == ec:
                    offset_votes.append(p["time_s"] - wall_rel[ti])

    has_clock_anchor = False
    a, b = 1.0, 0.0
    matched_tone: dict[int, dict[str, Any]] = {}

    if offset_votes:
        arr = np.sort(np.array(offset_votes, dtype=float))
        best_count = 0
        best_offset = float(arr[0])
        j = 0
        for i in range(len(arr)):
            while j < len(arr) and arr[j] - arr[i] <= 0.4:
                j += 1
            if j - i > best_count:
                best_count = j - i
                best_offset = float(np.median(arr[i:j]))
        a, b = 1.0, best_offset

        # Step 2: Two-pass greedy monotone matching
        for _pass in range(2):
            matched_tone = {}
            used_p: set[int] = set()
            for ti in range(n_trials):
                if np.isnan(wall_rel[ti]):
                    continue
                pred_t = a * wall_rel[ti] + b
                ec = expected_codes[ti]
                best_pi = None
                best_err = max_match_err_s
                for pi, p in enumerate(tone_pulses):
                    if pi in used_p:
                        continue
                    if ec is not None and p["code"] != ec:
                        continue
                    err = abs(p["time_s"] - pred_t)
                    if err < best_err:
                        best_err = err
                        best_pi = pi
                if best_pi is not None:
                    used_p.add(best_pi)
                    matched_tone[ti] = tone_pulses[best_pi]
            if len(matched_tone) >= 2:
                mw = np.array([wall_rel[ti] for ti in matched_tone], dtype=float)
                me = np.array([matched_tone[ti]["time_s"] for ti in matched_tone], dtype=float)
                a, b = _fit_robust_clock(mw, me)
                has_clock_anchor = True

    # Step 3: Build per-trial aligned records (ONLY interpolate missing triggers when anchored!)
    aligned: list[dict[str, Any]] = []
    used_resp: set[int] = set()
    tone_hw_count = 0
    tone_rec_count = 0
    resp_hw_count = 0
    resp_rec_count = 0
    tone_resids_ms: list[float] = []
    rt_diffs_ms: list[float] = []

    for ti, tr in enumerate(trials):
        w_tone = wall_rel[ti]
        pred_tone = float(a * w_tone + b) if (has_clock_anchor and not np.isnan(w_tone)) else None
        if ti in matched_tone:
            p = matched_tone[ti]
            tone_edf_s = float(p["time_s"])
            tone_source = "hardware"
            tone_code = int(p["code"])
            tone_err_ms = round((tone_edf_s - pred_tone) * 1000.0, 2) if pred_tone is not None else 0.0
            tone_hw_count += 1
            tone_resids_ms.append(abs(tone_err_ms))
        elif pred_tone is not None:
            tone_edf_s = round(pred_tone, 6)
            tone_source = "wall_clock_recovered"
            tone_code = expected_codes[ti]
            tone_err_ms = None
            tone_rec_count += 1
        else:
            tone_edf_s = None
            tone_source = "unmatched"
            tone_code = expected_codes[ti]
            tone_err_ms = None

        responded = bool(tr.get("responded"))
        rt_ms = tr.get("rt_ms")
        timing = tr.get("extra_json", {}).get("timing", {})
        kw_ts = parse_iso_ts(timing.get("keydown_wall"))
        resp_edf_s = None
        resp_source = "omission"
        resp_code = None
        if responded and tone_edf_s is not None:
            if kw_ts is not None and valid_walls and has_clock_anchor:
                pred_resp = float(a * (kw_ts - t0_wall) + b)
            elif rt_ms is not None:
                pred_resp = tone_edf_s + float(rt_ms) / 1000.0
            else:
                pred_resp = None

            expected_phys = tr.get("choice_phys")
            best_ri = None
            best_rerr = 0.20
            if pred_resp is not None:
                for ri, rp in enumerate(resp_pulses):
                    if ri in used_resp:
                        continue
                    if expected_phys and RESP_CODES.get(rp["code"]) != expected_phys:
                        continue
                    err = abs(rp["time_s"] - pred_resp)
                    if err < best_rerr:
                        best_rerr = err
                        best_ri = ri
            if best_ri is not None:
                used_resp.add(best_ri)
                resp_edf_s = float(resp_pulses[best_ri]["time_s"])
                resp_source = "hardware"
                resp_code = int(resp_pulses[best_ri]["code"])
                resp_hw_count += 1
                if rt_ms is not None:
                    hw_rt_ms = (resp_edf_s - tone_edf_s) * 1000.0
                    rt_diffs_ms.append(abs(hw_rt_ms - float(rt_ms)))
            elif pred_resp is not None:
                resp_edf_s = round(pred_resp, 6)
                resp_source = "wall_clock_recovered"
                resp_rec_count += 1

        fb_ts = parse_iso_ts(timing.get("feedback_dom_update_wall"))
        fb_edf_s = None
        if tone_edf_s is not None:
            if fb_ts is not None and valid_walls and has_clock_anchor:
                fb_edf_s = round(float(a * (fb_ts - t0_wall) + b), 6)
            else:
                fb_edf_s = round(tone_edf_s + 2.0, 6)

        aligned.append(
            {
                "trial_global_idx": ti,
                "phase": tr.get("phase"),
                "block": tr.get("block"),
                "idx": tr.get("idx"),
                "cls": tr.get("cls"),
                "hz": tr.get("hz"),
                "eff": tr.get("eff"),
                "nofb": tr.get("nofb"),
                "responded": responded,
                "choice_sem": tr.get("choice_sem"),
                "choice_phys": tr.get("choice_phys"),
                "correct": tr.get("correct"),
                "money": tr.get("money"),
                "fb_type": tr.get("fb_type"),
                "rt_ms": rt_ms,
                "tone_edf_s": round(tone_edf_s, 6) if tone_edf_s is not None else None,
                "tone_sample": int(round(tone_edf_s * sfreq)) if tone_edf_s is not None else None,
                "tone_code": tone_code,
                "tone_sync_source": tone_source,
                "tone_clock_resid_ms": tone_err_ms,
                "resp_edf_s": round(resp_edf_s, 6) if resp_edf_s is not None else None,
                "resp_sample": int(round(resp_edf_s * sfreq)) if resp_edf_s is not None else None,
                "resp_code": resp_code,
                "resp_sync_source": resp_source,
                "fb_edf_s": fb_edf_s,
            }
        )

    match_rate = tone_hw_count / n_trials if n_trials else 0.0
    errors: list[str] = []
    if not has_clock_anchor or tone_hw_count < 2:
        errors.append(
            f"未检测到有效硬件并口脉冲（仅匹配 {tone_hw_count}/{n_trials}），疑似触发线未接或 DC 通道号不匹配，已禁止使用墙钟伪造对齐。"
        )
    elif match_rate < min_hw_match_rate:
        errors.append(
            f"硬件触发匹配率过低（{tone_hw_count}/{n_trials} = {match_rate*100:.1f}% < {min_hw_match_rate*100:.0f}%），请检查并口接线或记录文件。"
        )
    if continuity and continuity.get("has_discontinuity"):
        errors.append(
            f"检测到 EDF+D 记录存在 {continuity.get('n_gaps')} 处时间间断（录制中途暂停），禁止按连续时间轴直接切分 Epoch！"
        )

    sync_ok = len(errors) == 0
    clock_drift_ppm = round((a - 1.0) * 1e6, 2) if has_clock_anchor else None
    return {
        "aligned_trials": aligned,
        "clock_model": (
            {
                "slope": round(a, 9),
                "intercept_s": round(b, 6),
                "clock_drift_ppm": clock_drift_ppm,
                "t0_wall_unix": t0_wall,
            }
            if has_clock_anchor
            else None
        ),
        "summary": {
            "sync_ok": sync_ok,
            "sync_error": " ".join(errors) if errors else "",
            "n_trials": n_trials,
            "tone_hw_match_rate": round(match_rate, 4),
            "n_hardware_pulses_total": len(pulses),
            "n_hardware_tone_pulses": len(tone_pulses),
            "n_hardware_resp_pulses": len(resp_pulses),
            "tone_hardware_matched": tone_hw_count,
            "tone_wall_clock_recovered": tone_rec_count,
            "tone_unmatched": n_trials - tone_hw_count - tone_rec_count,
            "resp_hardware_matched": resp_hw_count,
            "resp_wall_clock_recovered": resp_rec_count,
            "median_clock_resid_ms": round(float(np.median(tone_resids_ms)), 2) if tone_resids_ms else None,
            "max_clock_resid_ms": round(float(np.max(tone_resids_ms)), 2) if tone_resids_ms else None,
            "median_hw_vs_json_rt_diff_ms": round(float(np.median(rt_diffs_ms)), 2) if rt_diffs_ms else None,
            "edf_has_discontinuity": bool(continuity and continuity.get("has_discontinuity")),
            "edf_n_gaps": int(continuity.get("n_gaps", 0)) if continuity else 0,
        },
    }


def sync_subject_triggers(edf_path: str | Path, behavior_json_path: str | Path) -> dict[str, Any]:
    """End-to-end helper: check EDF+D continuity, decode DC triggers, and align to behavior JSON."""
    with open(behavior_json_path, "r", encoding="utf-8") as f:
        beh = json.load(f)
    dc_info = decode_dc_pulses(edf_path)
    sync_res = align_triggers_with_behavior(
        beh.get("trials", []),
        dc_info["pulses"],
        dc_info["sfreq"],
        continuity=dc_info.get("continuity"),
    )
    if dc_info.get("warning") and not sync_res["summary"]["sync_error"]:
        sync_res["summary"]["sync_ok"] = False
        sync_res["summary"]["sync_error"] = dc_info["warning"]
    return {
        "edf_info": {
            "sfreq": dc_info["sfreq"],
            "duration_s": dc_info["duration_s"],
            "meas_date": dc_info.get("meas_date"),
            "dc_channels": dc_info["dc_channels"],
            "continuity": dc_info.get("continuity"),
            "warning": dc_info.get("warning"),
        },
        "clock_model": sync_res["clock_model"],
        "summary": sync_res["summary"],
        "aligned_trials": sync_res["aligned_trials"],
    }
