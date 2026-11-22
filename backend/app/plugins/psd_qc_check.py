"""Example Plugin #2: Canonical Frequency-Band Power Profile across SEEG Shafts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
import os
import re
import numpy as np
from scipy.signal import welch

from ..plugin_base import AnalysisPlugin, PluginContext

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_jbs")

BANDS = [
    ("Theta (4-8 Hz)", 4.0, 8.0),
    ("Alpha (8-12 Hz)", 8.0, 12.0),
    ("Beta (13-30 Hz)", 13.0, 30.0),
    ("Gamma (35-80 Hz)", 35.0, 80.0),
]


class BandPowerSummaryPlugin(AnalysisPlugin):
    plugin_id = "band_power_profile"
    name = "各电极杆频段功率分布（Theta/Alpha/Beta/Gamma）"
    version = "1.0.0"
    description = "按电极杆汇总预处理后试次在 Theta(4-8Hz)、Alpha(8-12Hz)、Beta(13-30Hz)、Gamma(35-80Hz) 的平均功率热图。"
    category = "electrophysiology"
    requires = ["preprocess"]
    params_schema = [
        {
            "key": "tmin",
            "label": "分析窗口起点 (s)",
            "type": "number",
            "default": 0.0,
        },
        {
            "key": "tmax",
            "label": "分析窗口终点 (s)",
            "type": "number",
            "default": 1.5,
        },
    ]

    def run(self, ctx: PluginContext, params: dict[str, Any]) -> dict[str, Any]:
        import mne
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not ctx.epochs_fif_path or not ctx.epochs_fif_path.exists():
            raise FileNotFoundError("请先完成 Step 4（SEEG 预处理）以生成 Epochs .fif 文件。")

        epochs = mne.read_epochs(str(ctx.epochs_fif_path), preload=True, verbose="ERROR")
        tmin = float(params.get("tmin", 0.0))
        tmax = float(params.get("tmax", 1.5))
        ep_crop = epochs.copy().crop(
            tmin=max(tmin, float(epochs.tmin)),
            tmax=min(tmax, float(epochs.tmax)),
        )
        data = ep_crop.get_data() * 1e6  # (n_ep, n_ch, n_t)
        sfreq = float(ep_crop.info["sfreq"])
        freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq), data.shape[2]), axis=2)
        mean_psd = psd.mean(axis=0)  # (n_ch, n_f)

        shaft_map: dict[str, list[int]] = defaultdict(list)
        for i, ch in enumerate(ep_crop.ch_names):
            m = re.match(r"^([A-Za-z]+)", ch)
            shaft = m.group(1) if m else ch
            shaft_map[shaft].append(i)

        shafts = sorted(shaft_map.keys())
        mat = np.zeros((len(shafts), len(BANDS)), dtype=float)
        for si, sh in enumerate(shafts):
            idxs = shaft_map[sh]
            for bi, (_, f0, f1) in enumerate(BANDS):
                fm = (freqs >= f0) & (freqs <= f1)
                if np.any(fm):
                    mat[si, bi] = float(10 * np.log10(np.mean(mean_psd[idxs][:, fm]) + 1e-12))

        out_png = ctx.output_dir / "plugin_band_power.png"
        fig, ax = plt.subplots(figsize=(7.5, max(4.2, len(shafts) * 0.35)), dpi=150, facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels([b[0] for b in BANDS], color="#e2e8f0", fontsize=9)
        ax.set_yticks(range(len(shafts)))
        ax.set_yticklabels(shafts, color="#e2e8f0", fontsize=9)
        ax.set_title(f"Band Power by SEEG Shaft ({tmin:.2f}–{tmax:.2f} s, dB µV²/Hz)", color="#f8fafc")
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors="#cbd5e1")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "summary": f"已统计 {len(shafts)} 根电极杆在 [{tmin:.2f}, {tmax:.2f}] s 窗口的 4 频段功率。",
            "metrics": {"n_shafts": len(shafts), "n_epochs": int(len(ep_crop))},
            "image_path": str(out_png),
        }
