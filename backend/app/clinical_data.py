"""Clinical data management: photo/file upload, document text extraction, and AI structuring."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import base64
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile


DEFAULT_CLINICAL_FIELDS: dict[str, Any] = {
    "sex": "",
    "age": "",
    "handedness": "",
    "education_years": "",
    "epilepsy_duration_years": "",
    "seizure_onset_zone": "",
    "implantation_side": "",
    "seizure_frequency": "",
    "asm_medications": "",
    "psych_medications": "",
    "bdi_score": "",
    "hamd_score": "",
    "hama_score": "",
    "gad7_score": "",
    "moca_mmse": "",
    "other_scales": "",
    "clinical_summary": "",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TEXT_EXTS = {".txt", ".md", ".csv", ".json"}


def load_clinical_record(clinical_dir: Path) -> dict[str, Any]:
    rec_path = clinical_dir / "clinical_record.json"
    uploads_dir = clinical_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    if rec_path.exists():
        data = json.loads(rec_path.read_text(encoding="utf-8"))
    else:
        data = {
            "structured": dict(DEFAULT_CLINICAL_FIELDS),
            "ai_extract_notes": "",
            "updated_at": "",
        }

    # Ensure all schema keys exist
    merged = dict(DEFAULT_CLINICAL_FIELDS)
    merged.update(data.get("structured") or {})
    data["structured"] = merged

    # Refresh live file listing
    files = []
    for p in sorted(uploads_dir.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        ext = p.suffix.lower()
        files.append(
            {
                "name": p.name,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024.0, 1),
                "is_image": ext in IMAGE_EXTS,
                "ext": ext,
            }
        )
    data["files"] = files
    return data


def save_clinical_record(
    clinical_dir: Path,
    structured: dict[str, Any],
    ai_extract_notes: str | None = None,
) -> dict[str, Any]:
    clinical_dir.mkdir(parents=True, exist_ok=True)
    cur = load_clinical_record(clinical_dir)
    merged = dict(DEFAULT_CLINICAL_FIELDS)
    merged.update(cur.get("structured") or {})
    for k, v in (structured or {}).items():
        if k in merged:
            merged[k] = "" if v is None else str(v).strip()
    cur["structured"] = merged
    if ai_extract_notes is not None:
        cur["ai_extract_notes"] = ai_extract_notes
    cur["updated_at"] = datetime.now().isoformat(timespec="seconds")

    rec_path = clinical_dir / "clinical_record.json"
    rec_path.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
    return load_clinical_record(clinical_dir)


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        texts = [el.text for el in root.iter() if el.text and el.tag.endswith("}t")]
        return "\n".join(texts)
    except Exception:
        return ""


def _extract_xlsx_text(path: Path) -> str:
    try:
        import openpyxl

        wb = openpyxl.load_workbook(str(path), data_only=True)
        lines: list[str] = []
        for name in wb.sheetnames[:3]:
            ws = wb[name]
            lines.append(f"[Sheet: {name}]")
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i >= 80:
                    break
                vals = [str(v).strip() for v in row if v is not None and str(v).strip()]
                if vals:
                    lines.append(" | ".join(vals))
        return "\n".join(lines)
    except Exception:
        return ""


def _image_to_data_url(path: Path, max_dim: int = 1600) -> str:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > max_dim:
            im.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


def build_multimodal_clinical_payload(
    clinical_dir: Path,
    extra_note: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build OpenAI-compatible multimodal content parts (text + base64 images) for AI extraction."""
    rec = load_clinical_record(clinical_dir)
    text_chunks: list[str] = []
    image_parts: list[dict[str, Any]] = []
    used_files: list[str] = []

    if extra_note.strip():
        text_chunks.append(f"【用户补充文字】\n{extra_note.strip()}")

    for f in rec["files"]:
        p = Path(f["path"])
        ext = f["ext"]
        used_files.append(f["name"])
        if ext in IMAGE_EXTS:
            if len(image_parts) < 6:  # cap at 6 images per request
                try:
                    data_url = _image_to_data_url(p)
                    image_parts.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )
                except Exception as e:
                    text_chunks.append(f"[图片 {f['name']} 读取失败: {e}]")
        elif ext in TEXT_EXTS:
            txt = p.read_text(encoding="utf-8", errors="ignore")[:6000]
            text_chunks.append(f"【文件: {f['name']}】\n{txt}")
        elif ext == ".docx":
            txt = _extract_docx_text(p)[:6000]
            text_chunks.append(f"【Word 文件: {f['name']}】\n{txt}")
        elif ext in (".xlsx", ".xlsm"):
            txt = _extract_xlsx_text(p)[:6000]
            text_chunks.append(f"【Excel 文件: {f['name']}】\n{txt}")
        else:
            text_chunks.append(
                f"【文件: {f['name']} ({ext})】暂未内置直接解析该格式，若为扫描 PDF 建议截图/拍照上传图片以启用 AI 视觉识别。"
            )

    schema_desc = json.dumps(DEFAULT_CLINICAL_FIELDS, ensure_ascii=False, indent=2)
    prompt_text = (
        "请从以下上传的临床资料（图片或文本，可能包含病历首页、出院小结、抑郁/焦虑/认知量表、抗癫痫用药、癫痫病史等）中，"
        "尽可能准确地提取结构化临床信息。\n"
        "请**严格**返回一个 JSON 对象（可用 ```json ... ``` 包裹），包含以下字段（未提及的字段请留空字符串 \"\"，不要编造）：\n"
        f"{schema_desc}\n"
        "此外请额外包含一个字段 `extraction_notes`（简短中文说明提取了哪些依据、哪些信息缺失或需人工核对）。\n\n"
        + ("\n\n".join(text_chunks) if text_chunks else "（无纯文本，请直接识别附带图片中的临床信息）")
    )

    content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    content_parts.extend(image_parts)
    return content_parts, used_files


def parse_ai_json_response(text: str) -> dict[str, Any]:
    """Extract JSON dict from LLM reply (handles fenced ```json blocks or raw JSON)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else text
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            pass
    return {"clinical_summary": text.strip(), "extraction_notes": "AI 未返回标准 JSON，已将全文写入临床摘要。"}
