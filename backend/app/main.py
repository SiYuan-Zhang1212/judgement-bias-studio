"""FastAPI backend for Judgement Bias SEEG Standardized SOP Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import base64
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .workspace_store import WorkspaceStore
from .trigger_sync import sync_subject_triggers
from .behavior_analysis import analyze_behavior
from .seeg_preprocess import inspect_edf_channels_qc, run_pnas2025_preprocess
from .electrode_pipeline import (
    auto_pick_series,
    run_electrode_localization,
    apply_bindings_and_regenerate_contacts,
    build_series_volume3d,
)
from .doctor_table import parse_doctor_electrode_table
from .ai_assistant import AIConfigStore, ask_opencode_go, ask_opencode_go_multimodal
from .clinical_data import (
    load_clinical_record,
    save_clinical_record,
    build_multimodal_clinical_payload,
    parse_ai_json_response,
)
from .plugin_base import PluginContext, discover_plugins, LAST_PLUGIN_ERRORS


STUDIO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_DATA = STUDIO_ROOT / "workspace_data"
FRONTEND_DIR = STUDIO_ROOT / "frontend"
SINO_FRONTEND_DIR = STUDIO_ROOT.parent / "seeg-workbench" / "frontend"
DEFAULT_DEMO_DIR = STUDIO_ROOT.parent / "JudgementBiasDEMO"

store = WorkspaceStore(WORKSPACE_DATA)
ai_store = AIConfigStore(WORKSPACE_DATA / "ai_config.json")
PLUGINS = discover_plugins()

_SERIES_CATALOG: dict[str, dict[str, Any]] = {}
_VOL3D_CACHE: dict[str, tuple[dict[str, Any], bytes]] = {}

app = FastAPI(title="Judgement Bias SEEG Studio", version="1.3.0")


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
if SINO_FRONTEND_DIR.exists():
    app.mount("/sino-static", StaticFiles(directory=str(SINO_FRONTEND_DIR)), name="sino-static")


def _seed_demo_subject() -> None:
    if not store.list_subjects() and DEFAULT_DEMO_DIR.exists():
        paths = store.auto_discover_folder(DEFAULT_DEMO_DIR)
        store.upsert_subject(
            subject_id="sub-001",
            display_name="sub-001 (DEMO)",
            paths=paths,
            notes="内置匿名示例被试（JudgementBiasDEMO）",
        )


def _register_dicom_catalog(dicom_info: dict[str, Any]) -> None:
    for s in dicom_info.get("mr_series", []) + dicom_info.get("ct_series", []):
        if s.get("uid") and s.get("folder"):
            _SERIES_CATALOG[s["uid"]] = {
                "folder": s["folder"],
                "kind": s.get("kind", "mr"),
                "description": s.get("description", ""),
            }


def _ensure_series_in_catalog(uid: str) -> dict[str, Any]:
    if uid in _SERIES_CATALOG:
        return _SERIES_CATALOG[uid]
    for sub in store.list_subjects():
        p = store.subject_out_dir(sub["subject_id"]) / "intake" / "intake_report.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                _register_dicom_catalog(d.get("dicom", {}))
            except Exception:
                pass
    if uid in _SERIES_CATALOG:
        return _SERIES_CATALOG[uid]
    raise HTTPException(404, f"Series {uid} not found; run Step 1 Intake first.")


_seed_demo_subject()


class SubjectCreateReq(BaseModel):
    subject_id: str
    display_name: str = ""
    folder_path: str = ""
    mri_dir: str = ""
    ct_dir: str = ""
    edf_path: str = ""
    behavior_json: str = ""
    electrode_excel: str = ""
    notes: str = ""
    overwrite: bool = False


class LocalizationReq(BaseModel):
    mr_uid: str | None = None
    ct_uid: str | None = None
    hu_threshold: float = 2500.0
    contact_spacing_mm: float = 3.5


class BindingUpdateReq(BaseModel):
    bindings: dict[str, str]


class PreprocessReq(BaseModel):
    reref: str = "car"
    resample_hz: float = 250.0
    line_freq: float = 50.0
    hp_freq: float = 0.3
    fir_order: int = 180
    tmin: float = -1.0
    tmax: float = 2.5
    baseline_mode: str = "whole_epoch"
    iqr_weight: float = 2.3
    min_bad_chs_for_trial_drop: int = 3
    bad_channels: list[str] = []
    use_auto_bad: bool = True
    phase_filter: str = "main"
    epoch_lock: str = "tone"


class PluginRunReq(BaseModel):
    params: dict[str, Any] = {}


class Volume3dReq(BaseModel):
    mask_path: str = ""


class ClinicalFileItem(BaseModel):
    name: str
    data_base64: str


class ClinicalUploadReq(BaseModel):
    files: list[ClinicalFileItem]


class ClinicalSaveReq(BaseModel):
    structured: dict[str, Any] = {}


class ClinicalAIExtractReq(BaseModel):
    extra_note: str = ""
    overwrite: bool = False
    api_key: str = ""


class AIConfigReq(BaseModel):
    base_url: str = "https://opencode.ai/zen/go/v1"
    model: str = "deepseek-v4.1-flash-1017"
    api_key: str = ""


class AIAskReq(BaseModel):
    subject_id: str = ""
    step: str = "general"
    question: str
    api_key: str = ""


@app.get("/", response_class=HTMLResponse)
def index():
    idx = FRONTEND_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(idx.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "n_subjects": len(store.list_subjects()),
        "n_plugins": len(PLUGINS),
        "plugin_errors": list(LAST_PLUGIN_ERRORS),
        "ai": ai_store.public_view(),
    }


@app.get("/api/subjects")
def list_subjects():
    return {"subjects": store.list_subjects(), "default_demo_dir": str(DEFAULT_DEMO_DIR)}


@app.post("/api/subjects/discover")
def discover_folder(req: dict[str, str]):
    folder = req.get("folder_path", "").strip()
    if not folder:
        raise HTTPException(400, "folder_path is required")
    try:
        return store.auto_discover_folder(folder)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/subjects")
def create_subject(req: SubjectCreateReq):
    paths = {
        "root_dir": req.folder_path,
        "mri_dir": req.mri_dir,
        "ct_dir": req.ct_dir,
        "edf_path": req.edf_path,
        "behavior_json": req.behavior_json,
        "electrode_excel": req.electrode_excel,
    }
    if req.folder_path and not (req.mri_dir or req.ct_dir or req.edf_path or req.behavior_json):
        discovered = store.auto_discover_folder(req.folder_path)
        for k, v in discovered.items():
            if not paths.get(k):
                paths[k] = v
    try:
        rec = store.upsert_subject(
            req.subject_id,
            req.display_name or req.subject_id,
            paths,
            req.notes,
            overwrite=req.overwrite,
        )
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    return {"subject": rec}


@app.delete("/api/subjects/{subject_id}")
def delete_subject(subject_id: str):
    store.delete_subject(subject_id)
    return {"deleted": subject_id}


@app.get("/api/subjects/{subject_id}")
def get_subject(subject_id: str):
    sub = store.get_subject(subject_id)
    if not sub:
        raise HTTPException(404, "Subject not found")
    return {"subject": sub}


# --- STEP 1: Intake & Trigger Sync + Optional Doctor Table ---
@app.post("/api/subjects/{subject_id}/intake")
def run_intake(subject_id: str):
    sub = store.get_subject(subject_id)
    if not sub:
        raise HTTPException(404, "Subject not found")
    # Refresh any newly added files (such as doctor electrode_excel) inside root_dir
    sub = store.refresh_subject_paths(subject_id)
    paths = sub["paths"]
    out_dir = store.subject_out_dir(subject_id) / "intake"
    out_dir.mkdir(parents=True, exist_ok=True)

    doctor_tbl = None
    doc_path = paths.get("electrode_excel", "")
    if doc_path and Path(doc_path).exists():
        try:
            doctor_tbl = parse_doctor_electrode_table(doc_path)
        except Exception as e:
            doctor_tbl = {"error": str(e), "source_file": doc_path}

    dicom_info = auto_pick_series(paths.get("mri_dir", ""), paths.get("ct_dir", ""))
    _register_dicom_catalog(dicom_info)
    edf_qc = (
        inspect_edf_channels_qc(paths["edf_path"], doctor_table=doctor_tbl)
        if paths.get("edf_path") and Path(paths["edf_path"]).exists()
        else {}
    )
    sync_res = (
        sync_subject_triggers(paths["edf_path"], paths["behavior_json"])
        if paths.get("edf_path") and paths.get("behavior_json")
        else {}
    )

    intake_data = {
        "dicom": dicom_info,
        "edf": edf_qc,
        "doctor_table": doctor_tbl,
        "trigger_sync": {
            "edf_info": sync_res.get("edf_info"),
            "clock_model": sync_res.get("clock_model"),
            "summary": sync_res.get("summary"),
        },
    }
    (out_dir / "intake_report.json").write_text(
        json.dumps(intake_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if sync_res:
        (out_dir / "aligned_triggers.json").write_text(
            json.dumps(sync_res, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    sync_sum = sync_res.get("summary", {}) if sync_res else {}
    sync_ok = bool(sync_sum.get("sync_ok", False))
    store.update_step(
        subject_id,
        "intake",
        {
            "status": "done" if sync_ok else "error",
            "sync_ok": sync_ok,
            "sync_error": sync_sum.get("sync_error", ""),
            "summary": {
                "mr_series_count": len(dicom_info.get("mr_series", [])),
                "ct_series_count": len(dicom_info.get("ct_series", [])),
                "n_seeg_channels": edf_qc.get("n_seeg", 0),
                "n_shafts": len(edf_qc.get("shafts", [])),
                "auto_bad_count": len(edf_qc.get("auto_bad_channels", [])),
                "review_warn_count": len(edf_qc.get("review_warning_channels", [])),
                "has_doctor_table": bool(doctor_tbl and not doctor_tbl.get("error")),
                "tone_hw_matched": sync_sum.get("tone_hardware_matched", 0),
                "tone_wall_recovered": sync_sum.get("tone_wall_clock_recovered", 0),
                "median_resid_ms": sync_sum.get("median_clock_resid_ms"),
            },
        },
    )
    return intake_data


@app.get("/api/subjects/{subject_id}/intake")
def get_intake(subject_id: str):
    p = store.subject_out_dir(subject_id) / "intake" / "intake_report.json"
    if not p.exists():
        return {"exists": False}
    data = json.loads(p.read_text(encoding="utf-8"))
    _register_dicom_catalog(data.get("dicom", {}))
    return {"exists": True, "data": data}


# --- STEP 2: Clinical Data ---
@app.get("/api/subjects/{subject_id}/clinical")
def get_clinical(subject_id: str):
    if not store.get_subject(subject_id):
        raise HTTPException(404, "Subject not found")
    cdir = store.subject_out_dir(subject_id) / "clinical"
    return {"data": load_clinical_record(cdir)}


@app.put("/api/subjects/{subject_id}/clinical")
def put_clinical(subject_id: str, req: ClinicalSaveReq):
    if not store.get_subject(subject_id):
        raise HTTPException(404, "Subject not found")
    cdir = store.subject_out_dir(subject_id) / "clinical"
    rec = save_clinical_record(cdir, req.structured)
    filled = sum(1 for v in rec["structured"].values() if str(v).strip())
    store.update_step(
        subject_id,
        "clinical",
        {"status": "done" if (filled > 0 or rec["files"]) else "pending", "filled_fields": filled, "n_files": len(rec["files"])},
    )
    return {"data": rec}


@app.post("/api/subjects/{subject_id}/clinical/upload")
def upload_clinical_files(subject_id: str, req: ClinicalUploadReq):
    if not store.get_subject(subject_id):
        raise HTTPException(404, "Subject not found")
    cdir = store.subject_out_dir(subject_id) / "clinical"
    up_dir = cdir / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    saved_names: list[str] = []
    for item in req.files:
        raw_name = Path(item.name or "upload.bin").name
        safe_name = raw_name.replace("/", "_").replace("\\", "_")
        dest = up_dir / safe_name
        if dest.exists():
            stem, suf = dest.stem, dest.suffix
            idx = 1
            while (up_dir / f"{stem}_{idx}{suf}").exists():
                idx += 1
            dest = up_dir / f"{stem}_{idx}{suf}"
        b64 = item.data_base64.split(",", 1)[-1]
        dest.write_bytes(base64.b64decode(b64))
        saved_names.append(dest.name)

    rec = load_clinical_record(cdir)
    store.update_step(subject_id, "clinical", {"status": "done", "n_files": len(rec["files"])})
    return {"uploaded": saved_names, "data": rec}


@app.delete("/api/subjects/{subject_id}/clinical/files/{filename}")
def delete_clinical_file(subject_id: str, filename: str):
    if not store.get_subject(subject_id):
        raise HTTPException(404, "Subject not found")
    cdir = store.subject_out_dir(subject_id) / "clinical"
    target = (cdir / "uploads" / Path(filename).name).resolve()
    if not str(target).startswith(str((cdir / "uploads").resolve())):
        raise HTTPException(403, "Invalid filename")
    if target.exists():
        target.unlink()
    return {"deleted": filename, "data": load_clinical_record(cdir)}


@app.post("/api/subjects/{subject_id}/clinical/ai_extract")
def ai_extract_clinical(subject_id: str, req: ClinicalAIExtractReq):
    if not store.get_subject(subject_id):
        raise HTTPException(404, "Subject not found")
    cdir = store.subject_out_dir(subject_id) / "clinical"
    rec_before = load_clinical_record(cdir)
    if not rec_before["files"] and not req.extra_note.strip():
        raise HTTPException(400, "请先拍照/上传临床资料文件，或输入一段临床文字再点击 AI 提取。")

    content_parts, used_files = build_multimodal_clinical_payload(cdir, extra_note=req.extra_note)
    cfg = ai_store.load()
    try:
        ai_res = ask_opencode_go_multimodal(cfg, content_parts, override_api_key=req.api_key)
    except Exception as e:
        raise HTTPException(400, str(e))

    parsed = parse_ai_json_response(ai_res["answer"])
    notes = str(parsed.pop("extraction_notes", "") or "").strip()
    cur_struct = dict(rec_before["structured"])
    for k, v in parsed.items():
        if k in cur_struct and str(v or "").strip():
            if req.overwrite or not str(cur_struct[k]).strip():
                cur_struct[k] = str(v).strip()

    rec_after = save_clinical_record(cdir, cur_struct, ai_extract_notes=notes or f"已基于 {len(used_files)} 份文件提取")
    filled = sum(1 for v in rec_after["structured"].values() if str(v).strip())
    store.update_step(subject_id, "clinical", {"status": "done", "filled_fields": filled, "n_files": len(rec_after["files"])})
    return {"data": rec_after, "model": ai_res["model"], "used_files": used_files, "raw_answer": ai_res["answer"]}


# --- STEP 3: Behavioral Analysis ---
@app.post("/api/subjects/{subject_id}/behavior")
def run_behavior(subject_id: str):
    sub = store.get_subject(subject_id)
    if not sub:
        raise HTTPException(404, "Subject not found")
    beh_path = sub["paths"].get("behavior_json")
    if not beh_path or not Path(beh_path).exists():
        raise HTTPException(400, "Behavior JSON path missing")
    out_dir = store.subject_out_dir(subject_id) / "behavior"
    try:
        res = analyze_behavior(beh_path, output_dir=out_dir)
    except ValueError as e:
        raise HTTPException(400, str(e))
    store.update_step(
        subject_id,
        "behavior",
        {
            "status": "done",
            "qc_pass": res["qc_pass"],
            "summary": {
                "dprime": res["metrics"]["dprime"],
                "bias_score": res["metrics"]["bias_score"],
                "p_pos_on_mids": res["metrics"]["p_pos_on_mids"],
                "omission_rate": res["metrics"]["omission_rate"],
                "pos_anchor_cls": res["valence_mapping"]["pos_anchor_cls"],
            },
        },
    )
    return res


@app.get("/api/subjects/{subject_id}/behavior")
def get_behavior(subject_id: str):
    p = store.subject_out_dir(subject_id) / "behavior" / "behavior_summary.json"
    if not p.exists():
        return {"exists": False}
    return {"exists": True, "data": json.loads(p.read_text(encoding="utf-8"))}


# --- STEP 4: Electrode Localization + 3D Volume ---
@app.post("/api/volume3d/{uid}")
def api_volume3d(uid: str, req: Volume3dReq):
    meta = _ensure_series_in_catalog(uid)
    ckey = f"{uid}|{req.mask_path}"
    if ckey in _VOL3D_CACHE:
        info, payload = _VOL3D_CACHE[ckey]
    else:
        try:
            info, payload = build_series_volume3d(
                folder=meta["folder"],
                uid=uid,
                kind=meta["kind"],
                description=meta["description"],
                mask_path=req.mask_path,
            )
        except Exception as e:
            raise HTTPException(400, str(e))
        if len(_VOL3D_CACHE) >= 6:
            _VOL3D_CACHE.clear()
        _VOL3D_CACHE[ckey] = (info, payload)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"X-Volume": json.dumps(info)},
    )


@app.post("/api/subjects/{subject_id}/localization")
def run_localization(subject_id: str, req: LocalizationReq):
    sub = store.get_subject(subject_id)
    if not sub:
        raise HTTPException(404, "Subject not found")
    sub = store.refresh_subject_paths(subject_id)
    paths = sub["paths"]
    out_dir = store.subject_out_dir(subject_id) / "localization"

    intake_p = store.subject_out_dir(subject_id) / "intake" / "intake_report.json"
    edf_shafts = []
    doctor_tbl = None
    if intake_p.exists():
        intake_d = json.loads(intake_p.read_text(encoding="utf-8"))
        _register_dicom_catalog(intake_d.get("dicom", {}))
        edf_shafts = intake_d.get("edf", {}).get("shafts", [])
        doctor_tbl = intake_d.get("doctor_table")
    if (not doctor_tbl or doctor_tbl.get("error")) and paths.get("electrode_excel") and Path(paths["electrode_excel"]).exists():
        try:
            doctor_tbl = parse_doctor_electrode_table(paths["electrode_excel"])
        except Exception:
            doctor_tbl = None

    res = run_electrode_localization(
        mri_dir=paths["mri_dir"],
        ct_dir=paths["ct_dir"],
        output_dir=out_dir,
        mr_uid=req.mr_uid,
        ct_uid=req.ct_uid,
        hu_threshold=req.hu_threshold,
        contact_spacing_mm=req.contact_spacing_mm,
        edf_shafts=edf_shafts,
        doctor_table=doctor_tbl,
    )
    store.update_step(
        subject_id,
        "localization",
        {
            "status": "done",
            "summary": {
                "n_electrodes": res["n_electrodes"],
                "n_contacts": res["n_contacts"],
                "atlas_source": res["atlas_source"],
                "has_doctor_table": bool(doctor_tbl and not doctor_tbl.get("error")),
                "n_warnings": len(res.get("qc_warnings", [])),
            },
        },
    )
    return res


@app.get("/api/subjects/{subject_id}/localization")
def get_localization(subject_id: str):
    p = store.subject_out_dir(subject_id) / "localization" / "electrode_localization.json"
    if not p.exists():
        return {"exists": False}
    return {"exists": True, "data": json.loads(p.read_text(encoding="utf-8"))}


@app.post("/api/subjects/{subject_id}/localization/bindings")
def save_shaft_bindings(subject_id: str, req: BindingUpdateReq):
    out_dir = store.subject_out_dir(subject_id) / "localization"
    p = out_dir / "electrode_localization.json"
    if not p.exists():
        raise HTTPException(404, "Run localization first")
    data = json.loads(p.read_text(encoding="utf-8"))
    # Regenerate exact contact counts per bound EDF/Doctor shaft + apply doctor labels!
    updated = apply_bindings_and_regenerate_contacts(data, req.bindings, out_dir)
    return {"saved": True, "bindings": req.bindings, "data": updated}


# --- STEP 5: SEEG Preprocessing (PNAS 2025 SOP) ---
@app.post("/api/subjects/{subject_id}/preprocess")
def run_preprocess(subject_id: str, req: PreprocessReq):
    sub = store.get_subject(subject_id)
    if not sub:
        raise HTTPException(404, "Subject not found")
    paths = sub["paths"]
    out_dir = store.subject_out_dir(subject_id) / "preprocess"

    sync_p = store.subject_out_dir(subject_id) / "intake" / "aligned_triggers.json"
    sync_res = json.loads(sync_p.read_text(encoding="utf-8")) if sync_p.exists() else None
    intake_p = store.subject_out_dir(subject_id) / "intake" / "intake_report.json"
    doc_tbl = json.loads(intake_p.read_text(encoding="utf-8")).get("doctor_table") if intake_p.exists() else None

    try:
        res = run_pnas2025_preprocess(
            edf_path=paths["edf_path"],
            behavior_json_path=paths["behavior_json"],
            output_dir=out_dir,
            params=req.model_dump(),
            sync_result=sync_res,
            doctor_table=doc_tbl,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    store.update_step(
        subject_id,
        "preprocess",
        {
            "status": "done",
            "summary": {
                "reref": req.reref,
                "output_sfreq": res["output_sfreq"],
                "n_channels_retained": res["n_channels_retained"],
                "n_epochs_retained": res["n_epochs_retained"],
                "n_epochs_rejected": res["n_epochs_rejected_iqr"],
            },
        },
    )
    return res


@app.get("/api/subjects/{subject_id}/preprocess")
def get_preprocess(subject_id: str):
    p = store.subject_out_dir(subject_id) / "preprocess" / "preprocess_qc.json"
    if not p.exists():
        return {"exists": False}
    return {"exists": True, "data": json.loads(p.read_text(encoding="utf-8"))}


# --- STEP 6: Extensible Plugins ---
@app.get("/api/plugins")
def list_plugins():
    global PLUGINS
    PLUGINS = discover_plugins()
    return {
        "plugins": [p.descriptor() for p in PLUGINS.values()],
        "plugin_errors": list(LAST_PLUGIN_ERRORS),
    }


@app.post("/api/subjects/{subject_id}/plugins/{plugin_id}/run")
def run_plugin(subject_id: str, plugin_id: str, req: PluginRunReq):
    sub = store.get_subject(subject_id)
    if not sub:
        raise HTTPException(404, "Subject not found")
    if plugin_id not in PLUGINS:
        raise HTTPException(404, f"Plugin '{plugin_id}' not found")

    plugin = PLUGINS[plugin_id]
    sdir = store.subject_out_dir(subject_id)
    pdir = sdir / "plugins" / plugin_id
    pdir.mkdir(parents=True, exist_ok=True)

    pre_json = sdir / "preprocess" / "preprocess_qc.json"
    epo_path = None
    if pre_json.exists():
        ep_str = json.loads(pre_json.read_text(encoding="utf-8")).get("artifacts", {}).get("epochs_fif")
        if ep_str:
            epo_path = Path(ep_str)

    ctx = PluginContext(
        subject_id=subject_id,
        subject_dir=sdir,
        output_dir=pdir,
        behavior_summary_path=sdir / "behavior" / "behavior_summary.json",
        epochs_fif_path=epo_path,
        localization_json_path=sdir / "localization" / "electrode_localization.json",
    )
    try:
        res = plugin.run(ctx, req.params)
    except Exception as e:
        raise HTTPException(400, str(e))

    store.update_plugin_result(subject_id, plugin_id, res)
    return {"plugin_id": plugin_id, "result": res}


# --- AI Assistant (De-identified, Step-Focused Context) ---
def _build_deidentified_ai_context(subject_id: str, step: str) -> dict[str, Any]:
    """Build compact, de-identified context prioritized for the active step (< 5 KB).

    Strips patient real names, raw file paths, and bulky channel/contact/DICOM lists so
    important clinical/behavioral/QC metrics are NEVER truncated and PII is never leaked.
    """
    ctx: dict[str, Any] = {"subject_id": subject_id or "unknown", "active_step": step}
    if not subject_id:
        return ctx
    sdir = store.subject_out_dir(subject_id)

    def _load(rel: str) -> dict[str, Any] | None:
        fp = sdir / rel
        return json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None

    builders: dict[str, Any] = {}

    if (d := _load("intake/intake_report.json")) is not None:
        edf = d.get("edf") or {}
        builders["intake"] = {
            "n_mr_series": len((d.get("dicom") or {}).get("mr_series", [])),
            "n_ct_series": len((d.get("dicom") or {}).get("ct_series", [])),
            "edf_sfreq": edf.get("sfreq"),
            "edf_duration_s": edf.get("duration_s"),
            "n_seeg_channels": edf.get("n_seeg"),
            "n_shafts": len(edf.get("shafts", [])),
            "shafts": [{"shaft": s["shaft"], "side": s["side"], "n": s["n_contacts"], "missing": s["missing_indices"]} for s in edf.get("shafts", [])],
            "auto_bad_channels": edf.get("auto_bad_channels", []),
            "review_warning_channels": edf.get("review_warning_channels", []),
            "has_doctor_table": bool((d.get("doctor_table") or {}).get("n_electrodes")),
            "trigger_sync_summary": (d.get("trigger_sync") or {}).get("summary"),
            "clock_model": (d.get("trigger_sync") or {}).get("clock_model"),
        }

    if (d := _load("clinical/clinical_record.json")) is not None:
        st = {k: v for k, v in (d.get("structured") or {}).items() if str(v).strip()}
        builders["clinical"] = {"structured": st, "n_files": len(d.get("files", []))}

    if (d := _load("behavior/behavior_summary.json")) is not None:
        builders["behavior"] = {
            "cond": (d.get("participant") or {}).get("cond"),
            "valence_mapping": d.get("valence_mapping"),
            "metrics": d.get("metrics"),
            "psychometric": d.get("psychometric"),
            "by_block": d.get("by_block"),
            "mood_series": d.get("mood_series"),
            "qc_pass": d.get("qc_pass"),
            "qc_flags": d.get("qc_flags"),
        }

    if (d := _load("localization/electrode_localization.json")) is not None:
        builders["localization"] = {
            "n_electrodes": d.get("n_electrodes"),
            "n_contacts": d.get("n_contacts"),
            "mni_approx_only": d.get("mni_approx_only"),
            "qc_warnings": d.get("qc_warnings"),
            "electrodes": d.get("electrodes"),
        }

    if (d := _load("preprocess/preprocess_qc.json")) is not None:
        builders["preprocess"] = {
            "params": d.get("params"),
            "orig_sfreq": d.get("orig_sfreq"),
            "output_sfreq": d.get("output_sfreq"),
            "n_channels_retained": d.get("n_channels_retained"),
            "bad_channels": d.get("bad_channels"),
            "review_warning_channels": d.get("review_warning_channels"),
            "n_epochs_initial": d.get("n_epochs_initial"),
            "n_epochs_rejected_iqr": d.get("n_epochs_rejected_iqr"),
            "n_epochs_retained": d.get("n_epochs_retained"),
            "iqr_stats": d.get("iqr_stats"),
        }

    # Put active step FIRST so it is guaranteed top priority
    if step in builders:
        ctx[f"PRIMARY_{step}"] = builders.pop(step)
    # Attach compact cross-step summaries for the remaining steps
    for k, v in builders.items():
        ctx[k] = v
    return ctx


@app.get("/api/ai/config")
def get_ai_config():
    return ai_store.public_view()


@app.post("/api/ai/config")
def save_ai_config(req: AIConfigReq):
    return ai_store.save(req.base_url, req.model, req.api_key)


@app.post("/api/ai/ask")
def ai_ask(req: AIAskReq):
    cfg = ai_store.load()
    ctx = _build_deidentified_ai_context(req.subject_id, req.step)
    try:
        return ask_opencode_go(cfg, req.step, req.question, ctx, override_api_key=req.api_key)
    except Exception as e:
        raise HTTPException(400, str(e))


# --- Artifacts & Group Exports ---
@app.get("/api/artifact")
def get_artifact(path: str):
    p = Path(path).resolve()
    if not str(p).startswith(str(WORKSPACE_DATA.resolve())):
        raise HTTPException(403, "Access denied outside workspace_data")
    if not p.exists():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(str(p))


@app.post("/api/group/export")
def export_group():
    return store.export_group_tables()
