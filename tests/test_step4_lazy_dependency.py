"""Step 4 同级依赖（seeg-workbench）惰性加载与优雅降级测试。

为什么独立构造测试：
使用 monkeypatch + tmp_path 模拟“未安装 seeg-workbench”与“已安装但模块内部损坏”两种场景，
全程不依赖真实 seeg-workbench 仓库，也不加载重型影像模块，确保在 CI 或单独 clone 环境下可复现。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import electrode_pipeline as ep
from backend.app import main as app_main
from backend.app.workspace_store import WorkspaceStore


@pytest.fixture()
def isolated_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """隔离 store / ai_store / 3D 缓存与 sino 缓存，避免污染真实 workspace_data。"""
    tmp_store = WorkspaceStore(tmp_path / "ws")
    tmp_ai = app_main.AIConfigStore(tmp_path / "ai_config.json")
    tmp_store.upsert_subject("sub-lazy", "sub-lazy", {})

    monkeypatch.setattr(app_main, "store", tmp_store)
    monkeypatch.setattr(app_main, "ai_store", tmp_ai)
    monkeypatch.setattr(app_main, "_SERIES_CATALOG", {})
    monkeypatch.setattr(app_main, "_VOL3D_CACHE", {})
    ep._reset_sino_cache()
    yield tmp_store
    ep._reset_sino_cache()


def test_import_main_does_not_require_sino_sibling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """同级 seeg-workbench 不存在时，导入 backend.app.main 及 electrode_pipeline 必须成功且不加载 _sino_app。"""
    import subprocess

    studio_root = Path(__file__).resolve().parents[1]
    # 在全新子进程中把 seeg-workbench 路径重定向到不存在目录前先导入，验证模块顶层绝不加载 _sino_app
    code = (
        "import sys; "
        "from pathlib import Path; "
        "orig_is_file = Path.is_file; "
        "Path.is_file = lambda self: False if 'seeg-workbench' in str(self) else orig_is_file(self); "
        "import backend.app.main as m; "
        "assert '_sino_app' not in sys.modules; "
        "assert m.is_step4_available() is False; "
        "print('IMPORT_OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(studio_root),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "IMPORT_OK" in proc.stdout

    missing_dir = tmp_path / "nonexistent_seeg_workbench" / "backend" / "app"
    monkeypatch.setattr(ep, "_SIBLING_APP_DIR", missing_dir)
    ep._reset_sino_cache()
    assert app_main.app is not None
    assert "_sino_app" not in sys.modules
    assert ep.is_step4_available() is False


def test_health_exposes_step4_available_without_importing(
    isolated_app: WorkspaceStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """GET /api/health 仅做文件存在性检查：缺失返 False，文件存在但有导入异常时仍返 True 且不触发 import。"""
    client = TestClient(app_main.app)

    # 1. 目录缺失 -> step4_available: False
    missing_dir = tmp_path / "missing_sino"
    monkeypatch.setattr(ep, "_SIBLING_APP_DIR", missing_dir)
    r_false = client.get("/api/health")
    assert r_false.status_code == 200
    assert r_false.json()["step4_available"] is False
    assert "_sino_app" not in sys.modules

    # 2. 三个 .py 文件存在，但故意写入导入即报错的代码 -> /api/health 绝不 import，仍返回 True
    fake_dir = tmp_path / "fake_sino_no_import"
    fake_dir.mkdir()
    for name in ("electrode_detect", "pipeline", "volume_preview"):
        (fake_dir / f"{name}.py").write_text("raise RuntimeError('health must not import me')\n", encoding="utf-8")
    monkeypatch.setattr(ep, "_SIBLING_APP_DIR", fake_dir)

    r_true = client.get("/api/health")
    assert r_true.status_code == 200
    assert r_true.json()["step4_available"] is True
    assert "_sino_app" not in sys.modules


def test_step4_endpoints_return_503_when_sino_missing(
    isolated_app: WorkspaceStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """同级模块缺失时，Step 4 端点必须返回 503 且包含缺失文件、期望路径与修复指引。"""
    missing_dir = tmp_path / "no_seeg_workbench" / "backend" / "app"
    monkeypatch.setattr(ep, "_SIBLING_APP_DIR", missing_dir)
    ep._reset_sino_cache()

    client = TestClient(app_main.app)

    # POST /api/volume3d/{uid}
    r_vol = client.post("/api/volume3d/dummy-uid", json={"mask_path": ""})
    assert r_vol.status_code == 503
    detail_vol = r_vol.json()["detail"]
    assert "electrode_detect.py" in detail_vol
    assert str(missing_dir) in detail_vol
    assert "seeg-workbench" in detail_vol

    # POST /api/subjects/{subject_id}/localization
    r_loc = client.post("/api/subjects/sub-lazy/localization", json={"hu_threshold": 2500.0})
    assert r_loc.status_code == 503
    detail_loc = r_loc.json()["detail"]
    assert "electrode_detect.py" in detail_loc
    assert str(missing_dir) in detail_loc
    assert "seeg-workbench" in detail_loc


def test_bindings_endpoint_does_not_depend_on_sino(
    isolated_app: WorkspaceStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """/localization/bindings 仅操作已保存 JSON，不依赖 seeg-workbench，缺失 sino 时仍应正常工作。"""
    missing_dir = tmp_path / "no_sino"
    monkeypatch.setattr(ep, "_SIBLING_APP_DIR", missing_dir)
    ep._reset_sino_cache()

    loc_dir = isolated_app.subject_out_dir("sub-lazy") / "localization"
    loc_dir.mkdir(parents=True, exist_ok=True)
    seed_loc = {
        "contact_spacing_mm": 3.5,
        "ct_to_mr_m44": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "mni_model": {
            "source": "approx_t1_brain_bbox_unverified",
            "approx_only": True,
            "ras_min": [-70.0, -102.0, -50.0],
            "ras_max": [70.0, 70.0, 78.0],
            "mni_min": [-70.0, -102.0, -50.0],
            "mni_max": [70.0, 70.0, 78.0],
        },
        "shafts_geom": [
            {"uid": "E01", "tip_xyz": [20.0, 0.0, 0.0], "entry_xyz": [60.0, 0.0, 0.0], "axis": [-1.0, 0.0, 0.0], "length_mm": 40.0}
        ],
        "electrodes": [
            {"electrode": "E01", "hemisphere": "L", "n_contacts": 10, "tip_region": "L_待绑定医生表核对", "edf_shaft_binding": ""}
        ],
        "contacts": [],
        "edf_shafts_available": [{"shaft": "HR", "side": "R", "n_contacts": 8, "max_index": 8}],
        "doctor_table": None,
    }
    (loc_dir / "electrode_localization.json").write_text(json.dumps(seed_loc), encoding="utf-8")

    client = TestClient(app_main.app)
    r = client.post("/api/subjects/sub-lazy/localization/bindings", json={"bindings": {"E01": "HR"}})
    assert r.status_code == 200
    assert r.json()["data"]["electrodes"][0]["n_contacts"] == 8
    assert "_sino_app" not in sys.modules


def test_broken_sino_module_raises_original_runtime_error_not_503(
    isolated_app: WorkspaceStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """模块文件存在但导入时抛 RuntimeError：绝不能伪装成 503，必须原样冒泡且不缓存半初始化状态。"""
    broken_dir = tmp_path / "broken_sino"
    broken_dir.mkdir()
    (broken_dir / "electrode_detect.py").write_text(
        "raise RuntimeError('simulated internal sino bug')\n", encoding="utf-8"
    )
    (broken_dir / "pipeline.py").write_text("X = 1\n", encoding="utf-8")
    (broken_dir / "volume_preview.py").write_text("Y = 1\n", encoding="utf-8")

    monkeypatch.setattr(ep, "_SIBLING_APP_DIR", broken_dir)
    ep._reset_sino_cache()

    # 1. 直接调用 _get_sino_modules()：必须抛出原始 RuntimeError（不是 Step4Unavailable）
    with pytest.raises(RuntimeError, match="simulated internal sino bug") as exc_info:
        ep._get_sino_modules()
    assert not isinstance(exc_info.value, ep.Step4Unavailable)
    assert ep._SINO_MODULES_CACHE is None
    assert "_sino_app.electrode_detect" not in sys.modules

    # 2. 通过 /localization 端点调用：不能返回 503，应原样冒泡 RuntimeError
    client_raise = TestClient(app_main.app, raise_server_exceptions=True)
    with pytest.raises(RuntimeError, match="simulated internal sino bug"):
        client_raise.post("/api/subjects/sub-lazy/localization", json={})

    client_http = TestClient(app_main.app, raise_server_exceptions=False)
    r_loc = client_http.post("/api/subjects/sub-lazy/localization", json={})
    assert r_loc.status_code == 500
    assert r_loc.status_code != 503

    # 3. 通过 /api/volume3d/{uid} 端点调用：同样不能返回 503 或 400，应原样冒泡 RuntimeError
    app_main._SERIES_CATALOG["uid-broken"] = {"folder": str(tmp_path), "kind": "mr", "description": "T1"}
    with pytest.raises(RuntimeError, match="simulated internal sino bug"):
        client_raise.post("/api/volume3d/uid-broken", json={"mask_path": ""})

    r_vol = client_http.post("/api/volume3d/uid-broken", json={"mask_path": ""})
    assert r_vol.status_code == 500
    assert r_vol.status_code != 503


def test_partial_missing_sino_files_do_not_cache_half_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """三个模块缺一个时直接抛 Step4Unavailable，不加载其余文件也不写入缓存。"""
    partial_dir = tmp_path / "partial_sino"
    partial_dir.mkdir()
    (partial_dir / "electrode_detect.py").write_text("LOADED = True\n", encoding="utf-8")
    (partial_dir / "pipeline.py").write_text("LOADED = True\n", encoding="utf-8")
    # 故意缺少 volume_preview.py

    monkeypatch.setattr(ep, "_SIBLING_APP_DIR", partial_dir)
    ep._reset_sino_cache()

    assert ep.is_step4_available() is False
    with pytest.raises(ep.Step4Unavailable) as exc_info:
        ep._get_sino_modules()
    assert "volume_preview.py" in str(exc_info.value)
    assert ep._SINO_MODULES_CACHE is None
    assert "_sino_app.electrode_detect" not in sys.modules
