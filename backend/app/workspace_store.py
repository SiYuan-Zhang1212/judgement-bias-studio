"""Thread-safe, atomic multi-subject workspace registry & cross-subject aggregation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import json
import os
import threading


class WorkspaceStore:
    def __init__(self, root_dir: str | Path):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.subjects_dir = self.root / "subjects"
        self.subjects_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir = self.root / "group_exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "subjects_index.json"
        self._lock = threading.Lock()
        if not self.index_path.exists():
            self._write_index({})

    def _read_index(self) -> dict[str, Any]:
        """Read subjects index strictly; raise on corruption instead of silently wiping records."""
        if not self.index_path.exists():
            return {}
        raw = self.index_path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"Corrupt subjects index (expected JSON object): {self.index_path}")
        return data

    def _write_index(self, data: dict[str, Any]) -> None:
        """Atomic write via temp file + os.replace() to prevent partial-write corruption."""
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def list_subjects(self) -> list[dict[str, Any]]:
        with self._lock:
            idx = self._read_index()
        return sorted(idx.values(), key=lambda x: x.get("updated_at", ""), reverse=True)

    def get_subject(self, subject_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read_index().get(subject_id)

    def subject_out_dir(self, subject_id: str) -> Path:
        p = self.subjects_dir / subject_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def upsert_subject(
        self,
        subject_id: str,
        display_name: str,
        paths: dict[str, str],
        notes: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            idx = self._read_index()
            if subject_id in idx and not overwrite:
                raise FileExistsError(f"被试编号 '{subject_id}' 已存在，请换一个编号或明确选择覆盖。")
            now = datetime.now().isoformat(timespec="seconds")
            rec = {
                "subject_id": subject_id,
                "display_name": display_name or subject_id,
                "paths": {
                    "root_dir": paths.get("root_dir", ""),
                    "mri_dir": paths.get("mri_dir", ""),
                    "ct_dir": paths.get("ct_dir", ""),
                    "edf_path": paths.get("edf_path", ""),
                    "behavior_json": paths.get("behavior_json", ""),
                    "electrode_excel": paths.get("electrode_excel", ""),
                },
                "notes": notes,
                "steps": {
                    "intake": {"status": "pending"},
                    "clinical": {"status": "pending"},
                    "behavior": {"status": "pending"},
                    "localization": {"status": "pending"},
                    "preprocess": {"status": "pending"},
                },
                "plugins": {},
                "created_at": now,
                "updated_at": now,
            }
            idx[subject_id] = rec
            self._write_index(idx)
            return rec

    def refresh_subject_paths(self, subject_id: str) -> dict[str, Any]:
        """Re-discover missing optional paths (like doctor electrode_excel) inside root_dir."""
        with self._lock:
            idx = self._read_index()
            if subject_id not in idx:
                raise KeyError(f"Subject {subject_id} not found")
            rec = idx[subject_id]
            root_dir = rec.get("paths", {}).get("root_dir", "")
            if root_dir and Path(root_dir).exists():
                disc = self.auto_discover_folder(root_dir)
                for k, v in disc.items():
                    if v and not rec["paths"].get(k):
                        rec["paths"][k] = v
                rec["updated_at"] = datetime.now().isoformat(timespec="seconds")
                self._write_index(idx)
            return rec

    def update_step(self, subject_id: str, step_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            idx = self._read_index()
            if subject_id not in idx:
                raise KeyError(f"Subject {subject_id} not found")
            now = datetime.now().isoformat(timespec="seconds")
            idx[subject_id].setdefault("steps", {})[step_key] = {
                **payload,
                "updated_at": now,
            }
            idx[subject_id]["updated_at"] = now
            self._write_index(idx)
            return idx[subject_id]

    def update_plugin_result(self, subject_id: str, plugin_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            idx = self._read_index()
            if subject_id not in idx:
                raise KeyError(f"Subject {subject_id} not found")
            now = datetime.now().isoformat(timespec="seconds")
            idx[subject_id].setdefault("plugins", {})[plugin_id] = {
                **payload,
                "updated_at": now,
            }
            idx[subject_id]["updated_at"] = now
            self._write_index(idx)
            return idx[subject_id]

    def delete_subject(self, subject_id: str) -> None:
        with self._lock:
            idx = self._read_index()
            if subject_id in idx:
                del idx[subject_id]
                self._write_index(idx)

    def auto_discover_folder(self, folder_path: str | Path) -> dict[str, str]:
        """Auto-detect MRI dir, CT dir, EDF file, behavior JSON, and optional doctor table."""
        root = Path(folder_path)
        if not root.exists():
            raise FileNotFoundError(f"Folder does not exist: {root}")

        mri_dir = ""
        ct_dir = ""
        edf_path = ""
        behavior_json = ""
        electrode_excel = ""

        for p in sorted(root.iterdir()):
            if p.name.startswith("~$") or p.name.startswith("."):
                continue
            name_low = p.name.lower()
            if p.is_dir():
                if "mr" in name_low or "t1" in name_low:
                    mri_dir = str(p)
                elif "ct" in name_low:
                    ct_dir = str(p)
            elif p.is_file():
                if name_low.endswith(".edf") and not edf_path:
                    edf_path = str(p)
                elif name_low.endswith(".json") and not behavior_json:
                    behavior_json = str(p)
                elif name_low.endswith((".xlsx", ".xlsm", ".csv")) and not electrode_excel:
                    electrode_excel = str(p)

        return {
            "root_dir": str(root),
            "mri_dir": mri_dir,
            "ct_dir": ct_dir,
            "edf_path": edf_path,
            "behavior_json": behavior_json,
            "electrode_excel": electrode_excel,
        }

    def export_group_tables(self) -> dict[str, Any]:
        """Merge all completed subjects into group behavior & electrode CSVs using full key union."""
        subjects = self.list_subjects()
        beh_rows: list[dict[str, Any]] = []
        elec_rows: list[dict[str, Any]] = []

        for sub in subjects:
            sid = sub["subject_id"]
            out_dir = self.subject_out_dir(sid)
            beh_file = out_dir / "behavior" / "behavior_summary.json"
            if beh_file.exists():
                b = json.loads(beh_file.read_text(encoding="utf-8"))
                m = b.get("metrics", {})
                vm = b.get("valence_mapping", {})
                clin_file = out_dir / "clinical" / "clinical_record.json"
                clin = {}
                if clin_file.exists():
                    try:
                        cs = json.loads(clin_file.read_text(encoding="utf-8")).get("structured", {})
                        clin = {
                            f"clin_{k}": cs.get(k, "")
                            for k in (
                                "sex", "age", "education_years", "epilepsy_duration_years",
                                "seizure_onset_zone", "implantation_side", "asm_medications",
                                "bdi_score", "hamd_score", "hama_score", "gad7_score", "moca_mmse",
                            )
                        }
                    except Exception:
                        pass
                beh_rows.append(
                    {
                        "subject_id": sid,
                        "display_name": sub.get("display_name", sid),
                        "cond": b.get("participant", {}).get("cond"),
                        "pos_anchor_cls": vm.get("pos_anchor_cls"),
                        "neg_anchor_cls": vm.get("neg_anchor_cls"),
                        "qc_pass": b.get("qc_pass"),
                        **m,
                        **clin,
                    }
                )
            loc_csv = out_dir / "localization" / "electrode_contacts_mni.csv"
            if loc_csv.exists():
                with open(loc_csv, "r", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        elec_rows.append({"subject_id": sid, **row})

        def _union_keys(rows: list[dict[str, Any]]) -> list[str]:
            seen: dict[str, None] = {}
            for r in rows:
                for k in r.keys():
                    seen.setdefault(k, None)
            return list(seen.keys())

        beh_csv = self.exports_dir / "group_behavior_summary.csv"
        if beh_rows:
            keys = _union_keys(beh_rows)
            with open(beh_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(beh_rows)

        elec_csv = self.exports_dir / "group_electrode_coverage.csv"
        if elec_rows:
            keys = _union_keys(elec_rows)
            with open(elec_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(elec_rows)

        return {
            "n_behavior_subjects": len(beh_rows),
            "n_electrode_contacts": len(elec_rows),
            "behavior_csv": str(beh_csv) if beh_rows else None,
            "electrode_csv": str(elec_csv) if elec_rows else None,
        }
