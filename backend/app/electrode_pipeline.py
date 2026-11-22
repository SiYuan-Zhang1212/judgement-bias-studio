"""电极定位、深浅端物理定向、电极杆绑定与近似 MNI 坐标映射。

为什么把同级 `seeg-workbench` 模块改为惰性加载：
本工作台仅 Step 4（CT 电极自动检测、刚体配准、3D 脑预览）依赖同级仓库 `seeg-workbench`。
如果把 `seeg-workbench` 的导入放在模块顶层，一旦用户单独 clone 本仓库而未安装同级目录，
就会在 `import backend.app.main` 阶段直接抛错，导致整个 FastAPI 后端与 Step 1–3/5–6 全部瘫痪。
改为调用期惰性加载后，缺失同级文件会明确抛出 `Step4Unavailable`（API 层转为 503），
而模块已存在但内部存在语法错误或导入异常时则原样冒泡，避免掩盖真实故障。

关键科学/工程约束（保持原有口径不变）：
1. 深端（1 号触点）与浅端（入点）基于 CT 头体 3D 距离场（`edt_mm`）定向，不依赖 Z 轴正负。
2. 绑定 EDF/医生表电极杆后基于已持久化的几何轴与 4×4 刚体矩阵重算触点，无需依赖 sino 模块。
3. 近似 MNI 坐标基于规范 LPS 脑实质包围盒估计，明确标注 `approx_only=True`，医生表优先。
4. CT 检出电极数与 EDF/医生表不一致时显式输出 QC 警告，重跑定位时保留历史绑定。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import importlib.util
import json
import re
import sys
import types
import numpy as np
from scipy import ndimage as ndi


_SIBLING_APP_DIR = Path(__file__).resolve().parents[3] / "seeg-workbench" / "backend" / "app"
_REQUIRED_SINO_MODULES = ("electrode_detect", "pipeline", "volume_preview")

# 仅在三个模块全部加载成功后才写入缓存：(加载时的目录 Path, (detect, pipe, vol))。
# 失败或半初始化状态绝不写入缓存，且目录变更（如测试 monkeypatch）时自动失效。
_SINO_MODULES_CACHE: tuple[Path, tuple[Any, Any, Any]] | None = None


class Step4Unavailable(RuntimeError):
    """同级依赖仓库 `seeg-workbench` 或其所需模块文件缺失时抛出。

    仅用于区分“可选依赖未安装”与“依赖已存在但加载时报语法/运行错误”：
    前者由 API 层转为 503 并给出安装路径指引，后者必须原样向上冒泡。
    """


def _missing_sino_files() -> list[Path]:
    """返回缺失的同级 sino 模块文件路径列表（纯文件存在性检查，不执行任何 import）。"""
    return [
        _SIBLING_APP_DIR / f"{name}.py"
        for name in _REQUIRED_SINO_MODULES
        if not (_SIBLING_APP_DIR / f"{name}.py").is_file()
    ]


def is_step4_available() -> bool:
    """低开销检查 Step 4 所需的同级模块文件是否齐全（不做真实 import，无副作用）。"""
    return len(_missing_sino_files()) == 0


def step4_unavailable_detail() -> str:
    """生成面向用户的可操作错误说明（包含缺失文件、期望路径及修复方法）。"""
    missing = _missing_sino_files()
    missing_names = ", ".join(p.name for p in missing) if missing else ", ".join(
        f"{m}.py" for m in _REQUIRED_SINO_MODULES
    )
    return (
        f"Step 4 不可用：缺少同级依赖文件 [{missing_names}]，"
        f"期望目录路径为 {_SIBLING_APP_DIR}。"
        f"修复方法：请将 `seeg-workbench` 仓库放置到与 `judgement-bias-studio` 相同的父目录下后重试。"
    )


def _reset_sino_cache() -> None:
    """清空惰性加载缓存及 `_sino_app` 命名空间，仅供测试隔离使用。"""
    global _SINO_MODULES_CACHE
    _SINO_MODULES_CACHE = None
    for key in [k for k in list(sys.modules.keys()) if k == "_sino_app" or k.startswith("_sino_app.")]:
        sys.modules.pop(key, None)


def _load_sino_module(mod_name: str):
    """按需加载单个同级 sino 模块；文件缺失抛 `Step4Unavailable`，其他异常原样冒泡。"""
    mod_path = _SIBLING_APP_DIR / f"{mod_name}.py"
    if not mod_path.is_file():
        raise Step4Unavailable(step4_unavailable_detail())

    pkg_name = "_sino_app"
    # 如果 `_sino_app` 之前指向不同目录（如测试切换 tmp_path），刷新其搜索路径
    if pkg_name not in sys.modules or list(getattr(sys.modules[pkg_name], "__path__", [])) != [str(_SIBLING_APP_DIR)]:
        for key in [k for k in list(sys.modules.keys()) if k == pkg_name or k.startswith(f"{pkg_name}.")]:
            sys.modules.pop(key, None)
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_SIBLING_APP_DIR)]
        sys.modules[pkg_name] = pkg

    full = f"{pkg_name}.{mod_name}"
    if full in sys.modules:
        return sys.modules[full]

    spec = importlib.util.spec_from_file_location(full, str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为同级模块构造导入规范：{mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # 避免把半初始化的失败模块留在 sys.modules 里
        sys.modules.pop(full, None)
        raise
    return mod


def _get_sino_modules() -> tuple[Any, Any, Any]:
    """惰性加载并缓存 (electrode_detect, pipeline, volume_preview) 三个同级模块。"""
    global _SINO_MODULES_CACHE
    if _SINO_MODULES_CACHE is not None and _SINO_MODULES_CACHE[0] == _SIBLING_APP_DIR:
        return _SINO_MODULES_CACHE[1]

    # 先整体校验文件是否存在，防止只加载了一部分模块后才发现后续文件缺失
    if not is_step4_available():
        raise Step4Unavailable(step4_unavailable_detail())

    detect_mod = _load_sino_module("electrode_detect")
    pipe_mod = _load_sino_module("pipeline")
    vol_mod = _load_sino_module("volume_preview")
    loaded = (detect_mod, pipe_mod, vol_mod)
    _SINO_MODULES_CACHE = (_SIBLING_APP_DIR, loaded)
    return loaded


def build_series_volume3d(
    folder: str,
    uid: str,
    kind: str,
    description: str = "",
    mask_path: str = "",
) -> tuple[dict[str, Any], bytes]:
    """Build WebGL2-ready LPS 3D volume preview (reusing Sinovation volume_preview.py)."""
    _, _, sino_vol = _get_sino_modules()
    img = _read_dicom_series_sitk(folder, uid)
    vkind = "ct" if kind == "ct" else ("mr-t1" if ("t1" in (description or "").lower() or kind == "mr") else "mr")
    return sino_vol.preview(img, vkind, mask_path=mask_path)


def _scan_dicom_folder(folder: Path) -> list[dict[str, Any]]:
    import pydicom

    if not folder or not folder.exists():
        return []
    by_uid: dict[str, dict[str, Any]] = {}
    for p in folder.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if not uid:
                continue
            if uid not in by_uid:
                mod = str(getattr(ds, "Modality", "") or "").upper()
                desc = str(getattr(ds, "SeriesDescription", "") or "")
                thick = getattr(ds, "SliceThickness", None)
                by_uid[uid] = {
                    "uid": uid,
                    "modality": mod,
                    "kind": "ct" if mod == "CT" else "mr",
                    "description": desc,
                    "slice_thickness_mm": float(thick) if thick is not None else None,
                    "rows": int(getattr(ds, "Rows", 0) or 0),
                    "cols": int(getattr(ds, "Columns", 0) or 0),
                    "num_slices": 0,
                    "folder": str(p.parent),
                }
            by_uid[uid]["num_slices"] += 1
        except Exception:
            continue
    return list(by_uid.values())


def _read_dicom_series_sitk(folder: str, uid: str):
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    fnames = reader.GetGDCMSeriesFileNames(folder, uid)
    if not fnames:
        raise ValueError(f"No DICOM slices found for SeriesInstanceUID={uid} in {folder}")
    reader.SetFileNames(fnames)
    return sitk.Cast(reader.Execute(), sitk.sitkFloat32)


def auto_pick_series(mri_dir: str | Path, ct_dir: str | Path) -> dict[str, Any]:
    """Scan MRI and CT DICOM directories and recommend best thin-slice 3D T1 & post-op CT."""
    mr_list = _scan_dicom_folder(Path(mri_dir)) if mri_dir else []
    ct_list = _scan_dicom_folder(Path(ct_dir)) if ct_dir else []

    def score_mr(s: dict[str, Any]) -> tuple[int, int, int]:
        desc = (s["description"] or "").lower()
        is_t1 = 2 if ("t1" in desc and "3d" in desc) else (1 if "t1" in desc else 0)
        not_wip = 1 if ("vwip" not in desc and "survey" not in desc) else 0
        return (is_t1, not_wip, s["num_slices"])

    def score_ct(s: dict[str, Any]) -> tuple[int, int]:
        desc = (s["description"] or "").lower()
        is_mpr_or_bone = 2 if ("mpr" in desc or "bone" in desc) else 1
        return (is_mpr_or_bone, s["num_slices"])

    mr_sorted = sorted([s for s in mr_list if s["num_slices"] >= 20], key=score_mr, reverse=True)
    ct_sorted = sorted([s for s in ct_list if s["num_slices"] >= 20], key=score_ct, reverse=True)

    return {
        "mr_series": mr_sorted,
        "ct_series": ct_sorted,
        "recommended_mr_uid": mr_sorted[0]["uid"] if mr_sorted else None,
        "recommended_ct_uid": ct_sorted[0]["uid"] if ct_sorted else None,
    }


def _build_ct_head_depth_field(ct_img) -> tuple[np.ndarray, np.ndarray, Any]:
    """Compute 3D distance-from-head-surface (mm) on a 2x downsampled CT volume.

    Used to reliably distinguish Deep Intracranial Tip (larger distance from skin/air surface)
    from Superficial Skull Entry (smaller distance to surface / outside head), completely
    independent of patient orientation or electrode angle.
    """
    import SimpleITK as sitk

    _, _, sino_vol = _get_sino_modules()
    # Downsample by ~2mm isotropic for fast, smooth EDT
    ref, _ = sino_vol.lps_grid(ct_img, limit=160)
    vol = sitk.Resample(ct_img, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(vol)  # (Z, Y, X) in canonical LPS
    step = float(ref.GetSpacing()[0])

    head = arr > -300.0
    labs, n = ndi.label(head)
    if n > 1:
        counts = np.bincount(labs.ravel())
        counts[0] = 0
        head = labs == int(np.argmax(counts))
    head = ndi.binary_fill_holes(head)
    edt_mm = ndi.distance_transform_edt(head).astype(np.float32) * step

    # Intracranial centroid (deepest 15% of head interior) as tie-breaker
    deep_mask = edt_mm >= np.percentile(edt_mm[head], 85) if np.any(head) else head
    zz, yy, xx = np.where(deep_mask)
    if len(zz):
        c_idx = (float(xx.mean()), float(yy.mean()), float(zz.mean()))
        centroid_lps = np.array(ref.TransformContinuousIndexToPhysicalPoint(c_idx), dtype=float)
    else:
        centroid_lps = np.zeros(3, dtype=float)
    return edt_mm, centroid_lps, ref


def _sample_depth_mm(edt_mm: np.ndarray, ref_img, pt_lps: tuple[float, float, float]) -> float:
    try:
        idx = ref_img.TransformPhysicalPointToIndex(tuple(map(float, pt_lps)))
        x, y, z = idx
        D, H, W = edt_mm.shape
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            return float(edt_mm[z, y, x])
    except Exception:
        pass
    return 0.0


def orient_electrodes_deep_to_superficial(detected: list[Any], ct_img) -> list[dict[str, Any]]:
    """Orient every detected electrode so `tip_xyz` (Contact #1) is ALWAYS the deep end."""
    edt_mm, centroid_lps, ref_img = _build_ct_head_depth_field(ct_img)
    oriented: list[dict[str, Any]] = []

    for e in detected:
        p_a = np.array(e.tip_xyz, dtype=float)
        p_b = np.array(e.entry_xyz, dtype=float)
        d_a = _sample_depth_mm(edt_mm, ref_img, tuple(p_a))
        d_b = _sample_depth_mm(edt_mm, ref_img, tuple(p_b))

        # Primary: farther from outer head surface = deeper tip. Tie-breaker: closer to brain centroid.
        if abs(d_a - d_b) > 1.5:
            a_is_deep = d_a > d_b
        else:
            a_is_deep = float(np.linalg.norm(p_a - centroid_lps)) < float(np.linalg.norm(p_b - centroid_lps))

        tip = p_a if a_is_deep else p_b
        entry = p_b if a_is_deep else p_a
        tip_depth = d_a if a_is_deep else d_b
        entry_depth = d_b if a_is_deep else d_a

        vec = tip - entry
        nrm = float(np.linalg.norm(vec))
        # axis points from entry -> tip so `tip - axis * dist` walks from deepest #1 outward to entry!
        axis = (vec / nrm) if nrm > 1e-8 else np.array([0.0, 0.0, 1.0])

        oriented.append(
            {
                "uid": e.uid,
                "tip_xyz": [round(float(v), 2) for v in tip],
                "entry_xyz": [round(float(v), 2) for v in entry],
                "axis": [float(v) for v in axis],
                "length_mm": round(float(e.length_mm), 1),
                "tip_surface_depth_mm": round(tip_depth, 1),
                "entry_surface_depth_mm": round(entry_depth, 1),
            }
        )
    return oriented


def _generate_contacts_for_shaft(
    shaft: dict[str, Any],
    n_contacts: int,
    contact_spacing_mm: float = 3.5,
) -> list[dict[str, Any]]:
    """Generate 1..n_contacts along oriented shaft from deep tip (#1) outward toward entry."""
    tip = np.array(shaft["tip_xyz"], dtype=float)
    axis = np.array(shaft["axis"], dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    out = []
    for ci in range(n_contacts):
        pos = tip - axis * (ci * contact_spacing_mm)
        out.append(
            {
                "electrode": shaft["uid"],
                "contact_index": ci + 1,
                "ct_lps": [round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2]), 2)],
            }
        )
    return out


def _estimate_mr_to_mni_lps_brain(mr_img) -> dict[str, Any]:
    """Orientation-invariant linear MNI approximation using canonical LPS brain parenchyma mask.

    Resamples MRI to canonical LPS first (fixing sagittal/coronal voxel axis bugs!) and runs
    `t1_candidate` to isolate brain parenchyma without neck/scalp. Explicitly tagged as
    `approx_t1_brain_bbox_unverified` (NOT nonlinear MNI152 registration).
    """
    import SimpleITK as sitk

    _, _, sino_vol = _get_sino_modules()
    ref, _ = sino_vol.lps_grid(mr_img, limit=180)
    vol = sitk.Resample(mr_img, ref, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(vol)
    step = float(ref.GetSpacing()[0])
    brain_prob = sino_vol.t1_candidate(arr, step)
    mask = brain_prob >= 0.5
    zz, yy, xx = np.where(mask)
    if len(zz) < 200:
        return {
            "source": "approx_fallback_unverified",
            "approx_only": True,
            "ras_min": [-70.0, -102.0, -50.0],
            "ras_max": [70.0, 70.0, 78.0],
            "mni_min": [-70.0, -102.0, -50.0],
            "mni_max": [70.0, 70.0, 78.0],
        }

    # Physical LPS bounding box of actual brain parenchyma (independent of acquisition orientation!)
    p_lo = [int(np.percentile(xx, 1)), int(np.percentile(yy, 1)), int(np.percentile(zz, 1))]
    p_hi = [int(np.percentile(xx, 99)), int(np.percentile(yy, 99)), int(np.percentile(zz, 99))]
    corners_lps = [
        ref.TransformIndexToPhysicalPoint((ix, iy, iz))
        for ix in (p_lo[0], p_hi[0])
        for iy in (p_lo[1], p_hi[1])
        for iz in (p_lo[2], p_hi[2])
    ]
    ras = np.array([[-p[0], -p[1], p[2]] for p in corners_lps], dtype=float)
    return {
        "source": "approx_t1_brain_bbox_unverified",
        "approx_only": True,
        "ras_min": [round(float(v), 2) for v in ras.min(axis=0)],
        "ras_max": [round(float(v), 2) for v in ras.max(axis=0)],
        # Standard MNI152 pial brain parenchyma extent (mm RAS)
        "mni_min": [-70.0, -102.0, -50.0],
        "mni_max": [70.0, 70.0, 78.0],
    }


def lps_to_mni_approx(
    lps_xyz: list[float] | tuple[float, float, float],
    mni_model: dict[str, Any],
) -> dict[str, Any]:
    """Map individual MR LPS (mm) to approximate MNI RAS (mm); clearly marks unverified status."""
    ras = np.array([-float(lps_xyz[0]), -float(lps_xyz[1]), float(lps_xyz[2])], dtype=float)
    r_min = np.array(mni_model["ras_min"], dtype=float)
    r_max = np.array(mni_model["ras_max"], dtype=float)
    m_min = np.array(mni_model["mni_min"], dtype=float)
    m_max = np.array(mni_model["mni_max"], dtype=float)
    span = np.maximum(r_max - r_min, 1.0)
    mni = m_min + (ras - r_min) / span * (m_max - m_min)
    hemi = "R" if mni[0] >= 0 else "L"
    return {
        "mni_approx_x": round(float(mni[0]), 2),
        "mni_approx_y": round(float(mni[1]), 2),
        "mni_approx_z": round(float(mni[2]), 2),
        "hemisphere": hemi,
        "region": f"{hemi}_待绑定医生表核对",
        "label_source": "unassigned_approx_only",
        "atlas_source": mni_model["source"],
    }


def _apply_m44(m44: list[list[float]], pt: list[float]) -> list[float]:
    M = np.array(m44, dtype=float)
    v = np.array([pt[0], pt[1], pt[2], 1.0], dtype=float)
    res = M @ v
    return [round(float(res[0]), 2), round(float(res[1]), 2), round(float(res[2]), 2)]


def _default_contacts_for_length(length_mm: float, spacing_mm: float) -> int:
    # Conservative initial contact estimate before binding to EDF/Doctor table
    est = int(round(min(length_mm, 56.0) / max(spacing_mm, 1.0))) + 1
    return max(8, min(16, est))


def _write_contacts_csv(csv_path: Path, contacts: list[dict[str, Any]]) -> None:
    fieldnames = [
        "electrode", "edf_shaft", "contact_index", "hemisphere", "region", "label_source",
        "mr_x", "mr_y", "mr_z", "mni_approx_x", "mni_approx_y", "mni_approx_z", "atlas_source",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in contacts:
            w.writerow(
                {
                    "electrode": c["electrode"],
                    "edf_shaft": c.get("edf_shaft", ""),
                    "contact_index": c["contact_index"],
                    "hemisphere": c["hemisphere"],
                    "region": c["region"],
                    "label_source": c.get("label_source", "unassigned_approx_only"),
                    "mr_x": c["mr_lps"][0],
                    "mr_y": c["mr_lps"][1],
                    "mr_z": c["mr_lps"][2],
                    "mni_approx_x": c["mni_approx_x"],
                    "mni_approx_y": c["mni_approx_y"],
                    "mni_approx_z": c["mni_approx_z"],
                    "atlas_source": c["atlas_source"],
                }
            )


def _norm_shaft_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name or "").upper()


def apply_bindings_and_regenerate_contacts(
    loc_data: dict[str, Any],
    bindings: dict[str, str],
    out_dir: Path,
) -> dict[str, Any]:
    """Bind CT electrodes to EDF/Doctor shafts, regenerate exact contact counts, and apply doctor labels."""
    shafts_geom = {s["uid"]: s for s in loc_data.get("shafts_geom", [])}
    m44 = loc_data.get("ct_to_mr_m44")
    mni_model = loc_data.get("mni_model")
    spacing = float(loc_data.get("contact_spacing_mm", 3.5))

    edf_by_norm = {
        _norm_shaft_name(sh["shaft"]): sh
        for sh in (loc_data.get("edf_shafts_available") or [])
    }
    doc_tbl = loc_data.get("doctor_table")
    doc_by_norm = (
        {_norm_shaft_name(de["name"]): de for de in doc_tbl.get("electrodes", [])}
        if (doc_tbl and not doc_tbl.get("error"))
        else {}
    )

    new_contacts: list[dict[str, Any]] = []
    for e in loc_data.get("electrodes", []):
        ct_uid = e["electrode"]
        bound = bindings.get(ct_uid, e.get("edf_shaft_binding", ""))
        e["edf_shaft_binding"] = bound
        bnorm = _norm_shaft_name(bound)
        edf_sh = edf_by_norm.get(bnorm)
        doc_e = doc_by_norm.get(bnorm)

        # Determine exact contact count from bound Doctor table / EDF shaft!
        if doc_e and doc_e.get("n_contacts"):
            n_c = int(doc_e["n_contacts"])
            e["contact_count_source"] = "doctor_table"
        elif edf_sh and edf_sh.get("max_index"):
            n_c = int(edf_sh["max_index"])
            e["contact_count_source"] = "edf_shaft"
        else:
            n_c = int(e.get("n_contacts", 12))
            e["contact_count_source"] = e.get("contact_count_source", "estimated_from_ct")
        e["n_contacts"] = n_c

        geom = shafts_geom.get(ct_uid)
        if geom and m44 and mni_model:
            raw_cs = _generate_contacts_for_shaft(geom, n_c, contact_spacing_mm=spacing)
            e_contacts = []
            for rc in raw_cs:
                mr_lps = _apply_m44(m44, rc["ct_lps"])
                info = lps_to_mni_approx(mr_lps, mni_model)
                c_rec = {
                    **rc,
                    "edf_shaft": edf_sh["shaft"] if edf_sh else bound,
                    "mr_lps": mr_lps,
                    **info,
                }
                if doc_e:
                    dreg = doc_e.get("contact_regions", {}).get(str(rc["contact_index"]))
                    if dreg:
                        c_rec["region"] = dreg
                        c_rec["label_source"] = "doctor_verified"
                e_contacts.append(c_rec)
            new_contacts.extend(e_contacts)
            e["hemisphere"] = e_contacts[0]["hemisphere"] if e_contacts else e["hemisphere"]
            if doc_e:
                e["doctor_target"] = doc_e.get("target", "")
                e["doctor_entry"] = doc_e.get("entry", "")
                c1_reg = doc_e.get("contact_regions", {}).get("1")
                if c1_reg:
                    e["tip_region"] = f"{c1_reg} (医生核对)"
            else:
                e["tip_region"] = e_contacts[0]["region"] if e_contacts else e["tip_region"]

    if new_contacts:
        loc_data["contacts"] = new_contacts
        loc_data["n_contacts"] = len(new_contacts)

    csv_path = out_dir / "electrode_contacts_mni.csv"
    _write_contacts_csv(csv_path, loc_data.get("contacts", []))
    (out_dir / "electrode_localization.json").write_text(
        json.dumps(loc_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return loc_data


def run_electrode_localization(
    mri_dir: str | Path,
    ct_dir: str | Path,
    output_dir: str | Path,
    mr_uid: str | None = None,
    ct_uid: str | None = None,
    hu_threshold: float = 2500.0,
    contact_spacing_mm: float = 3.5,
    edf_shafts: list[dict[str, Any]] | None = None,
    doctor_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run oriented CT electrode detection -> CT-to-MR rigid registration -> MNI approx + QC."""
    # 先行获取同级 sino 模块：缺失时立即抛 Step4Unavailable（无需浪费时间扫描 DICOM）
    sino_detect, sino_pipe, _ = _get_sino_modules()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Preserve existing user shaft bindings across re-runs!
    prev_bindings: dict[str, str] = {}
    prev_json = out / "electrode_localization.json"
    if prev_json.exists():
        try:
            old = json.loads(prev_json.read_text(encoding="utf-8"))
            for e in old.get("electrodes", []):
                if e.get("edf_shaft_binding"):
                    prev_bindings[e["electrode"]] = e["edf_shaft_binding"]
        except Exception:
            pass

    picked = auto_pick_series(mri_dir, ct_dir)
    mr_uid = mr_uid or picked["recommended_mr_uid"]
    ct_uid = ct_uid or picked["recommended_ct_uid"]

    mr_meta = next((s for s in picked["mr_series"] if s["uid"] == mr_uid), None)
    ct_meta = next((s for s in picked["ct_series"] if s["uid"] == ct_uid), None)
    if mr_meta is None or ct_meta is None:
        raise ValueError("Could not locate selected MRI or CT DICOM series.")

    mr_img = _read_dicom_series_sitk(mr_meta["folder"], mr_uid)
    ct_img = _read_dicom_series_sitk(ct_meta["folder"], ct_uid)

    # 1. Detect electrodes on post-op CT & orient Deep Tip (#1) -> Superficial Entry via EDT!
    detected_raw = sino_detect.detect_electrodes(ct_img, hu_threshold=hu_threshold)
    shafts_geom = orient_electrodes_deep_to_superficial(detected_raw, ct_img)

    # 2. Rigid CT -> MR registration (pipeline.register_ct_to_mr already returns forward CT->MR point transform!)
    tfm = sino_pipe.register_ct_to_mr(ct_img, mr_img)
    m44 = sino_pipe.transform_to_matrix4x4(tfm)

    # 3. Orientation-invariant canonical LPS T1 brain parenchyma MNI approx
    mni_model = _estimate_mr_to_mni_lps_brain(mr_img)

    # 4. Initial contact generation (will be refined to exact counts upon binding)
    localized_contacts: list[dict[str, Any]] = []
    elec_summary: list[dict[str, Any]] = []
    for sh in shafts_geom:
        n_est = _default_contacts_for_length(sh["length_mm"], contact_spacing_mm)
        raw_cs = _generate_contacts_for_shaft(sh, n_est, contact_spacing_mm=contact_spacing_mm)
        e_contacts = []
        for rc in raw_cs:
            mr_lps = _apply_m44(m44, rc["ct_lps"])
            info = lps_to_mni_approx(mr_lps, mni_model)
            e_contacts.append({**rc, "edf_shaft": "", "mr_lps": mr_lps, **info})
        localized_contacts.extend(e_contacts)
        elec_summary.append(
            {
                "electrode": sh["uid"],
                "hemisphere": e_contacts[0]["hemisphere"] if e_contacts else "?",
                "n_contacts": n_est,
                "contact_count_source": "estimated_from_ct",
                "length_mm": sh["length_mm"],
                "tip_surface_depth_mm": sh["tip_surface_depth_mm"],
                "entry_surface_depth_mm": sh["entry_surface_depth_mm"],
                "tip_region": e_contacts[0]["region"] if e_contacts else "Unknown",
                "edf_shaft_binding": prev_bindings.get(sh["uid"], ""),
            }
        )

    # 5. Explicit QC warnings for shaft count mismatch or unverified MNI approximation
    qc_warnings: list[str] = [
        "MNI 坐标当前为基于 T1 脑实质包围盒的近似估计（非非线性配准），精确脑区请务必绑定医生术后电极位置表。"
    ]
    n_ct = len(elec_summary)
    n_edf = len(edf_shafts or [])
    n_doc = int(doctor_table.get("n_electrodes", 0)) if (doctor_table and not doctor_table.get("error")) else 0
    if n_edf and n_ct != n_edf:
        qc_warnings.append(
            f"CT 自动检出 {n_ct} 根电极，但 EDF 记录包含 {n_edf} 根电极杆（差 {abs(n_ct - n_edf)} 根，可能存在交叉粘连或淡显电极，请在 3D 视图中核对）。"
        )
    if n_doc and n_ct != n_doc:
        qc_warnings.append(
            f"CT 自动检出 {n_ct} 根电极，但医生电极位置表包含 {n_doc} 根电极。"
        )

    csv_path = out / "electrode_contacts_mni.csv"
    result = {
        "mr_uid": mr_uid,
        "mr_description": mr_meta["description"],
        "ct_uid": ct_uid,
        "ct_description": ct_meta["description"],
        "hu_threshold": hu_threshold,
        "contact_spacing_mm": contact_spacing_mm,
        "n_electrodes": len(elec_summary),
        "n_contacts": len(localized_contacts),
        "atlas_source": mni_model["source"],
        "mni_approx_only": True,
        "qc_warnings": qc_warnings,
        "ct_to_mr_m44": m44,
        "mni_model": mni_model,
        "shafts_geom": shafts_geom,
        "electrodes": elec_summary,
        "contacts": localized_contacts,
        "edf_shafts_available": edf_shafts or [],
        "doctor_table": doctor_table,
        "artifacts": {
            "contacts_csv": str(csv_path),
            "localization_json": str(out / "electrode_localization.json"),
        },
    }

    # Re-apply preserved bindings immediately if any existed!
    if prev_bindings:
        result = apply_bindings_and_regenerate_contacts(result, prev_bindings, out)
    else:
        _write_contacts_csv(csv_path, localized_contacts)
        (out / "electrode_localization.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result
