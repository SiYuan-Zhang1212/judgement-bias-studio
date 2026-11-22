"""Isolated, self-contained unit & regression tests for Judgement Bias Studio.

Never mutates `workspace_data/`, never references patient real names or external files,
and directly tests every critical scientific & engineering invariant.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import openpyxl
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.behavior_analysis import analyze_behavior, infer_valence_mapping
from backend.app.trigger_sync import align_triggers_with_behavior
from backend.app.seeg_preprocess import apply_pnas2025_filters
from backend.app.doctor_table import parse_doctor_electrode_table
from backend.app.workspace_store import WorkspaceStore
from backend.app.clinical_data import build_multimodal_clinical_payload, parse_ai_json_response
from backend.app.plugin_base import discover_plugins
from backend.app import main as app_main


def _make_synthetic_behavior(cond: str = "B", pos_cls: str = "low") -> dict:
    """Create a self-contained synthetic Judgement Bias behavioral JSON."""
    neg_cls = "high" if pos_cls == "low" else "low"
    tones = [("low", 261.63, 21), ("m1", 311.13, 23), ("m2", 369.99, 25), ("m3", 440.0, 27), ("high", 523.25, 29)]
    trials = []
    t_unix = 1793323226.0
    from datetime import datetime, timezone

    def iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    for i in range(60):
        cls, hz, code = tones[i % 5]
        eff = "pos" if cls == pos_cls else ("neg" if cls == neg_cls else None)
        nofb = 1 if eff is None else 0
        choice_sem = "pos" if (eff == "pos" or (eff is None and i % 4 != 0)) else "neg"
        choice_phys = "right" if choice_sem == "pos" else "left"
        rt_ms = 900.0 + (i % 5) * 40.0
        trials.append(
            {
                "idx": i,
                "phase": "main",
                "block": (i // 20) + 1,
                "tib": i % 20,
                "cls": cls,
                "hz": hz,
                "eff": eff,
                "nofb": nofb,
                "responded": 1,
                "rt_ms": rt_ms,
                "choice_sem": choice_sem,
                "choice_phys": choice_phys,
                "correct": 1 if eff is not None else None,
                "money": 1 if eff == "pos" else 0,
                "fb_type": "win" if eff == "pos" else ("avoid" if eff == "neg" else "ambiguous_no_feedback"),
                "tone_onset_wall": iso(t_unix),
                "extra_json": {
                    "timing": {
                        "keydown_wall": iso(t_unix + rt_ms / 1000.0),
                        "feedback_dom_update_wall": iso(t_unix + 2.0),
                    },
                    "trigger": {"backend": "parallel", "tone_code": code},
                },
            }
        )
        t_unix += 4.5
    return {
        "participant": {"name": "synth_p01", "pid": 1, "cond": cond, "combo": cond},
        "config": {"toneMs": 500},
        "summary": {"balance": 10, "durationMin": 5.0},
        "trials": trials,
        "moodProbes": [
            {"probe_index": 1, "after_trial": -1, "valence": 6, "rt_ms": 2000, "events_json": {"checkpoint": "pre_block_1"}},
            {"probe_index": 2, "after_trial": 60, "valence": 7, "rt_ms": 1800, "events_json": {"checkpoint": "post_block_3"}},
        ],
    }


class TestJudgementBiasStudio(unittest.TestCase):
    def test_valence_agnostic_and_missing_eff_raises(self):
        b_low = _make_synthetic_behavior(cond="B", pos_cls="low")
        vm_b = infer_valence_mapping(b_low["trials"])
        self.assertEqual(vm_b["pos_anchor_cls"], "low")
        self.assertEqual(vm_b["cls_to_rank"]["high"], 0)
        self.assertEqual(vm_b["cls_to_rank"]["low"], 4)

        b_high = _make_synthetic_behavior(cond="A", pos_cls="high")
        vm_a = infer_valence_mapping(b_high["trials"])
        self.assertEqual(vm_a["pos_anchor_cls"], "high")
        self.assertEqual(vm_a["cls_to_rank"]["low"], 0)
        self.assertEqual(vm_a["cls_to_rank"]["high"], 4)

        # Missing `eff` must raise ValueError instead of silently assuming low=pos!
        bad_trials = [{"cls": "m2", "eff": None}]
        with self.assertRaises(ValueError):
            infer_valence_mapping(bad_trials)

    def test_zero_pulses_and_discontinuity_fail_loudly(self):
        beh = _make_synthetic_behavior()
        trials = beh["trials"]

        # 0 pulses -> MUST fail sync_ok and NEVER fabricate intercept=0!
        res0 = align_triggers_with_behavior(trials, [], sfreq=2000.0)
        self.assertFalse(res0["summary"]["sync_ok"])
        self.assertIsNone(res0["clock_model"])
        self.assertEqual(res0["summary"]["tone_unmatched"], len(trials))
        self.assertEqual(res0["summary"]["tone_wall_clock_recovered"], 0)
        self.assertTrue(all(t["tone_edf_s"] is None for t in res0["aligned_trials"]))

        # Synthesize clean pulses + drop 5 -> recovers within < 2 ms and passes sync_ok
        from backend.app.trigger_sync import parse_iso_ts
        t0 = parse_iso_ts(trials[0]["tone_onset_wall"])
        a_true, b_true = 1.000012, 400.5
        pulses = []
        for tr in trials:
            wt = parse_iso_ts(tr["tone_onset_wall"]) - t0
            tc = tr["extra_json"]["trigger"]["tone_code"]
            pulses.append({"sample": 0, "time_s": a_true * wt + b_true, "code": tc, "dur_ms": 10.0, "kind": "tone"})
            kw = parse_iso_ts(tr["extra_json"]["timing"]["keydown_wall"]) - t0
            rc = 35 if tr["choice_phys"] == "left" else 37
            pulses.append({"sample": 0, "time_s": a_true * kw + b_true, "code": rc, "dur_ms": 10.0, "kind": "response"})

        rng = random.Random(7)
        drop = set(rng.sample(range(len(pulses)), 5))
        kept = [p for i, p in enumerate(pulses) if i not in drop]
        res_ok = align_triggers_with_behavior(trials, kept, sfreq=2000.0, continuity={"has_discontinuity": False, "n_gaps": 0})
        self.assertTrue(res_ok["summary"]["sync_ok"])
        self.assertEqual(res_ok["summary"]["tone_unmatched"], 0)
        for tr, al in zip(trials, res_ok["aligned_trials"]):
            wt = parse_iso_ts(tr["tone_onset_wall"]) - t0
            self.assertLess(abs(al["tone_edf_s"] - (a_true * wt + b_true)) * 1000.0, 2.0)

        # EDF+D discontinuity -> MUST fail sync_ok!
        res_gap = align_triggers_with_behavior(trials, kept, sfreq=2000.0, continuity={"has_discontinuity": True, "n_gaps": 1})
        self.assertFalse(res_gap["summary"]["sync_ok"])
        self.assertTrue(res_gap["summary"]["edf_has_discontinuity"])

    def test_fir_notch_attenuation_at_250_500_1000hz(self):
        """Verify > 30 dB suppression at 50/100/150 Hz across 250, 500, and 1000 Hz."""
        for sfreq, test_freqs in ((250.0, [50.0]), (500.0, [50.0, 100.0, 150.0]), (1000.0, [50.0, 100.0, 150.0])):
            t = np.arange(int(sfreq * 10)) / sfreq
            sig = np.sin(2 * np.pi * 10.0 * t)  # 10 Hz physiological signal (should pass!)
            for f0 in test_freqs:
                sig = sig + np.sin(2 * np.pi * f0 * t)
            cleaned = apply_pnas2025_filters(sig[None, :], sfreq=sfreq, line_freq=50.0, hp_freq=0.3)[0]
            # Measure FFT amplitude in middle 6 seconds (away from edges)
            sl = slice(int(sfreq * 2), int(sfreq * 8))
            freqs = np.fft.rfftfreq(sl.stop - sl.start, 1.0 / sfreq)
            amp_in = np.abs(np.fft.rfft(sig[sl]))
            amp_out = np.abs(np.fft.rfft(cleaned[sl]))
            # 10 Hz passband must stay intact (< 1 dB loss)
            i10 = int(np.argmin(np.abs(freqs - 10.0)))
            self.assertLess(abs(20 * np.log10(amp_out[i10] / amp_in[i10])), 1.5)
            # Every line harmonic must have > 30 dB attenuation!
            for f0 in test_freqs:
                idx = int(np.argmin(np.abs(freqs - f0)))
                atten_db = -20 * np.log10((amp_out[idx] + 1e-12) / (amp_in[idx] + 1e-12))
                self.assertGreater(atten_db, 30.0, f"Weak notch at {f0} Hz for sfreq={sfreq}: {atten_db:.1f} dB")

    def test_synthetic_doctor_excel_merged_cells(self):
        with tempfile.TemporaryDirectory() as td:
            xlsx = Path(td) / "synthetic_doctor_table.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(["电极", "靶点", "入点", "点数", 1, 2, 3, 4, 5, 6])
            ws.append(["H-R", "海马-R", "颞中回-R", 6, "海马", None, "白质", None, "颞中回", None])
            ws.merge_cells("E2:F2")
            ws.merge_cells("G2:H2")
            ws.merge_cells("I2:J2")
            wb.save(xlsx)

            res = parse_doctor_electrode_table(xlsx)
            self.assertEqual(res["n_electrodes"], 1)
            e = res["electrodes"][0]
            self.assertEqual(e["name"], "H-R")
            self.assertEqual(e["n_contacts"], 6)
            self.assertEqual(e["contact_regions"]["1"], "海马")
            self.assertEqual(e["contact_regions"]["2"], "海马")
            self.assertEqual(e["contact_regions"]["6"], "颞中回")

    def test_workspace_store_isolation_duplicate_and_union_csv(self):
        with tempfile.TemporaryDirectory() as td:
            st = WorkspaceStore(td)
            st.upsert_subject("s1", "S1", {})
            # Duplicate without overwrite must fail!
            with self.assertRaises(FileExistsError):
                st.upsert_subject("s1", "S1-dup", {})

            # Corrupt JSON must raise instead of silently returning {}
            st.index_path.write_text("{corrupt", encoding="utf-8")
            with self.assertRaises(Exception):
                st.list_subjects()
            st.index_path.write_text("{}", encoding="utf-8")

            # Group export union columns: s1 has NO clinical fields, s2 HAS clinical fields
            st.upsert_subject("s1", "S1", {})
            st.upsert_subject("s2", "S2", {})
            beh_json = Path(td) / "beh.json"
            beh_json.write_text(json.dumps(_make_synthetic_behavior()), encoding="utf-8")
            analyze_behavior(beh_json, output_dir=st.subject_out_dir("s1") / "behavior")
            analyze_behavior(beh_json, output_dir=st.subject_out_dir("s2") / "behavior")
            from backend.app.clinical_data import save_clinical_record
            save_clinical_record(st.subject_out_dir("s2") / "clinical", {"age": "28", "bdi_score": "14"})

            exp = st.export_group_tables()
            with open(exp["behavior_csv"], "r", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertIn("clin_bdi_score", rows[0])
            s2_row = next(r for r in rows if r["subject_id"] == "s2")
            self.assertEqual(s2_row["clin_bdi_score"], "14")

    def test_isolated_api_clinical_and_deidentified_ai_context(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_store = WorkspaceStore(Path(td) / "ws")
            tmp_ai = app_main.AIConfigStore(Path(td) / "ai_config.json")
            tmp_store.upsert_subject("sub-test", "SECRET_REAL_NAME_DO_NOT_LEAK", {})

            with patch.object(app_main, "store", tmp_store), patch.object(app_main, "ai_store", tmp_ai):
                client = TestClient(app_main.app)

                # Duplicate subject API returns 409
                r_dup = client.post("/api/subjects", json={"subject_id": "sub-test", "display_name": "dup"})
                self.assertEqual(r_dup.status_code, 409)

                # Clinical photo upload + save
                buf = io.BytesIO()
                Image.new("RGB", (64, 64), (30, 80, 140)).save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                r = client.post(
                    "/api/subjects/sub-test/clinical/upload",
                    json={"files": [{"name": "scale.jpg", "data_base64": f"data:image/jpeg;base64,{b64}"}]},
                )
                self.assertEqual(r.status_code, 200)
                r2 = client.put("/api/subjects/sub-test/clinical", json={"structured": {"age": "26", "bdi_score": "14"}})
                self.assertEqual(r2.status_code, 200)

                # Multimodal payload check
                parts, used = build_multimodal_clinical_payload(tmp_store.subject_out_dir("sub-test") / "clinical", "备注")
                self.assertTrue(any(p["type"] == "image_url" for p in parts))
                self.assertEqual(len(used), 1)

                # De-identified AI context: real name MUST NOT appear, PRIMARY_behavior MUST appear!
                beh_p = Path(td) / "b.json"
                beh_p.write_text(json.dumps(_make_synthetic_behavior()), encoding="utf-8")
                analyze_behavior(beh_p, output_dir=tmp_store.subject_out_dir("sub-test") / "behavior")
                ctx = app_main._build_deidentified_ai_context("sub-test", "behavior")
                ctx_str = json.dumps(ctx, ensure_ascii=False)
                self.assertNotIn("SECRET_REAL_NAME_DO_NOT_LEAK", ctx_str)
                self.assertIn("PRIMARY_behavior", ctx)
                self.assertLess(len(ctx_str), 6000)

                # AI config defaults
                r_ai = client.get("/api/ai/config")
                self.assertEqual(r_ai.json()["base_url"], "https://opencode.ai/zen/go/v1")
                self.assertEqual(r_ai.json()["model"], "deepseek-v4.1-flash-1017")

                # Plugin discovery
                plugins = discover_plugins()
                self.assertIn("erp_preview", plugins)
                self.assertIn("band_power_profile", plugins)

                parsed = parse_ai_json_response('```json\n{"age": "28", "bdi_score": "12"}\n```')
                self.assertEqual(parsed["age"], "28")


if __name__ == "__main__":
    unittest.main()
