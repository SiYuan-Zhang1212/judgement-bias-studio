"""Parser for doctor-verified post-op SEEG electrode location tables (.xlsx / .csv).

Supports the clinical Chinese hospital Excel layout (e.g. `电极位置表-*.xlsx`):
- Header row containing `电极`, `靶点`, `入点`, `点数`, followed by contact numbers `1, 2, 3, ...`
- Merged cells spanning multiple contacts are expanded so every contact index inherits its
  anatomical region label (e.g. `被盖中`, `WT`, `MCC`, `SMA`, ...).
- Optional data source: when present, takes priority over approximate MNI labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import re


def _clean_cell(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip()


def parse_doctor_electrode_table(file_path: str | Path) -> dict[str, Any]:
    """Parse doctor-verified electrode location table (.xlsx or .csv)."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Electrode table not found: {p}")

    ext = p.suffix.lower()
    if ext == ".csv":
        return _parse_csv(p)
    if ext in (".xlsx", ".xlsm"):
        return _parse_xlsx(p)
    raise ValueError(f"Unsupported electrode table format: {ext} (use .xlsx or .csv)")


def _parse_xlsx(p: Path) -> dict[str, Any]:
    import openpyxl

    wb = openpyxl.load_workbook(str(p), data_only=True)
    ws = wb.active

    # Expand merged cells into a 2D lookup grid
    max_r = ws.max_row or 0
    max_c = ws.max_column or 0
    grid: list[list[Any]] = [
        [ws.cell(r, c).value for c in range(1, max_c + 1)]
        for r in range(1, max_r + 1)
    ]
    for rng in ws.merged_cells.ranges:
        val = ws.cell(rng.min_row, rng.min_col).value
        for r in range(rng.min_row, rng.max_row + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                grid[r - 1][c - 1] = val

    # Locate header row containing "电极" or contact numbers 1, 2, 3
    header_r = -1
    contact_cols: dict[int, int] = {}
    col_elec, col_target, col_entry, col_count = 0, 1, 2, 3

    for r_idx, row in enumerate(grid[:10]):
        strs = [_clean_cell(v) for v in row]
        if any("电极" in s for s in strs) or (1 in row and 2 in row and 3 in row):
            header_r = r_idx
            for c_idx, val in enumerate(row):
                s = _clean_cell(val)
                if "电极" in s:
                    col_elec = c_idx
                elif "靶点" in s or "target" in s.lower():
                    col_target = c_idx
                elif "入点" in s or "entry" in s.lower():
                    col_entry = c_idx
                elif "点数" in s or "触点" in s or "count" in s.lower():
                    col_count = c_idx
                else:
                    try:
                        num = int(float(s))
                        if 1 <= num <= 24:
                            contact_cols[num] = c_idx
                    except Exception:
                        pass
            break

    if header_r < 0:
        header_r = 0

    electrodes: list[dict[str, Any]] = []
    for r_idx in range(header_r + 1, len(grid)):
        row = grid[r_idx]
        ename = _clean_cell(row[col_elec]) if col_elec < len(row) else ""
        if not ename:
            continue
        # Skip sub-notes rows that don't look like an electrode name or have no count/contacts
        n_contacts_raw = row[col_count] if col_count < len(row) else None
        try:
            n_contacts = int(float(n_contacts_raw)) if n_contacts_raw is not None else 0
        except Exception:
            n_contacts = 0

        target = _clean_cell(row[col_target]) if col_target < len(row) else ""
        entry = _clean_cell(row[col_entry]) if col_entry < len(row) else ""

        contact_regions: dict[str, str] = {}
        last_label = ""
        for c_num in sorted(contact_cols.keys()):
            if n_contacts and c_num > n_contacts:
                break
            c_idx = contact_cols[c_num]
            lbl = _clean_cell(row[c_idx]) if c_idx < len(row) else ""
            if lbl:
                last_label = lbl
                contact_regions[str(c_num)] = lbl
            elif last_label and c_num <= n_contacts:
                # Forward-fill unmerged sparse hospital sheets where label starts at boundary contact
                contact_regions[str(c_num)] = last_label

        if not n_contacts and not contact_regions:
            continue
        if not n_contacts:
            n_contacts = len(contact_regions)

        side = "R" if re.search(r"[-_]?R$", ename, re.I) else ("L" if re.search(r"[-_]?L$", ename, re.I) else "?")
        electrodes.append(
            {
                "name": ename,
                "side": side,
                "target": target,
                "entry": entry,
                "n_contacts": n_contacts,
                "contact_regions": contact_regions,
            }
        )

    return {
        "source_file": str(p),
        "file_name": p.name,
        "n_electrodes": len(electrodes),
        "electrodes": electrodes,
    }


def _parse_csv(p: Path) -> dict[str, Any]:
    electrodes_map: dict[str, dict[str, Any]] = {}
    with open(p, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ename = _clean_cell(row.get("electrode") or row.get("电极") or row.get("shaft"))
            if not ename:
                continue
            cidx = _clean_cell(row.get("contact_index") or row.get("contact") or row.get("触点"))
            reg = _clean_cell(row.get("region") or row.get("脑区") or row.get("anatomy"))
            if ename not in electrodes_map:
                side = "R" if re.search(r"[-_]?R$", ename, re.I) else ("L" if re.search(r"[-_]?L$", ename, re.I) else "?")
                electrodes_map[ename] = {
                    "name": ename,
                    "side": side,
                    "target": _clean_cell(row.get("target") or row.get("靶点")),
                    "entry": _clean_cell(row.get("entry") or row.get("入点")),
                    "n_contacts": 0,
                    "contact_regions": {},
                }
            if cidx and reg:
                electrodes_map[ename]["contact_regions"][str(cidx)] = reg
                electrodes_map[ename]["n_contacts"] = max(
                    electrodes_map[ename]["n_contacts"], int(float(cidx))
                )
    elecs = list(electrodes_map.values())
    return {
        "source_file": str(p),
        "file_name": p.name,
        "n_electrodes": len(elecs),
        "electrodes": elecs,
    }


def apply_doctor_table_to_localization(
    loc_data: dict[str, Any],
    doctor_table: dict[str, Any] | None,
    bindings: dict[str, str],
) -> dict[str, Any]:
    """Override/enrich contact regions with doctor-verified anatomical labels where bound."""
    if not doctor_table:
        return loc_data
    doc_by_name = {e["name"]: e for e in doctor_table.get("electrodes", [])}
    # Also allow matching normalized names (e.g. 'SMA-R' <-> 'SMAR')
    doc_norm = {re.sub(r"[^A-Za-z0-9]", "", k).upper(): v for k, v in doc_by_name.items()}

    for c in loc_data.get("contacts", []):
        ct_name = c["electrode"]
        bound = bindings.get(ct_name, "")
        if not bound:
            continue
        doc_e = doc_by_name.get(bound) or doc_norm.get(re.sub(r"[^A-Za-z0-9]", "", bound).upper())
        if not doc_e:
            continue
        reg = doc_e.get("contact_regions", {}).get(str(c["contact_index"]))
        if reg:
            c["doctor_region"] = reg
            c["region"] = reg
            c["label_source"] = "doctor_verified"

    for e in loc_data.get("electrodes", []):
        bound = bindings.get(e["electrode"], "")
        if bound:
            e["edf_shaft_binding"] = bound
            doc_e = doc_by_name.get(bound) or doc_norm.get(re.sub(r"[^A-Za-z0-9]", "", bound).upper())
            if doc_e:
                e["doctor_target"] = doc_e.get("target", "")
                e["doctor_entry"] = doc_e.get("entry", "")
                c1 = doc_e.get("contact_regions", {}).get("1")
                if c1:
                    e["tip_region"] = f"{c1} (医生核对)"
    return loc_data
