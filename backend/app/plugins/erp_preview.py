"""Example Plugin #1: Phase-Locked Evoked Potential (ERP) GFP Preview.

Correctly averages across trials FIRST to extract phase-locked evoked potentials per channel
`(n_ch, n_times)`, then computes Global Field Power (RMS across channels) — avoiding the
common pitfall of averaging single-trial RMS (which reflects ongoing total amplitude).
"""

from __future__ import annotations

from typing import Any
import os
import numpy as np

from ..plugin_base import AnalysisPlugin, PluginContext

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_jbs")


class ErpPreviewPlugin(AnalysisPlugin):
    plugin_id = "erp_preview"
    name = "条件平均 ERP（锁相诱发电位 GFP）预览"
    version = "1.1.0"
    description = "先按试次平均得到各通道锁相诱发电位（Evoked），再计算跨通道 GFP/RMS，对比锚点音与模糊中间音（或正/负选择）。"
    category = "electrophysiology"
    requires = ["preprocess"]
    params_schema = [
        {
            "key": "contrast",
            "label": "对比条件",
            "type": "select",
            "default": "anchor_vs_mid",
            "options": [
                {"value": "anchor_vs_mid", "label": "锚点音 vs 模糊中间音"},
                {"value": "pos_vs_neg_choice", "label": "正性选择 vs 负性选择"},
            ],
        },
        {
            "key": "shaft_prefix",
            "label": "筛选电极杆前缀（留空=全脑通道）",
            "type": "text",
            "default": "",
        },
    ]

    def run(self, ctx: PluginContext, params: dict[str, Any]) -> dict[str, Any]:
        import mne
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not ctx.epochs_fif_path or not ctx.epochs_fif_path.exists():
            raise FileNotFoundError("请先完成 Step 5（SEEG 预处理）以生成 Epochs .fif 文件。")

        epochs = mne.read_epochs(str(ctx.epochs_fif_path), preload=True, verbose="ERROR")
        prefix = str(params.get("shaft_prefix", "") or "").strip().upper()
        if prefix:
            picks = [c for c in epochs.ch_names if c.upper().startswith(prefix)]
            if not picks:
                raise ValueError(f"未找到前缀为 '{prefix}' 的通道。")
            epochs = epochs.copy().pick(picks)

        contrast = params.get("contrast", "anchor_vs_mid")
        meta = epochs.metadata
        if contrast == "pos_vs_neg_choice":
            m1 = meta["choice_sem"] == "pos"
            m2 = meta["choice_sem"] == "neg"
            l1, l2 = "Positive Choice", "Negative Choice"
        else:
            m1 = meta["is_ambiguous"] == False  # noqa: E712
            m2 = meta["is_ambiguous"] == True   # noqa: E712
            l1, l2 = "Anchor Tones", "Ambiguous Mids"

        data = epochs.get_data() * 1e6  # (n_trials, n_ch, n_times) in uV
        times = epochs.times

        def evoked_gfp(mask_arr: np.ndarray) -> np.ndarray:
            if not np.any(mask_arr):
                return np.zeros(len(times), dtype=float)
            # Average across trials FIRST -> (n_ch, n_times), then RMS across channels (GFP)
            evoked = np.mean(data[mask_arr], axis=0)
            return np.sqrt(np.mean(evoked**2, axis=0))

        g1 = evoked_gfp(m1.values)
        g2 = evoked_gfp(m2.values)

        out_png = ctx.output_dir / f"plugin_erp_{contrast}.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150, facecolor="#0f172a")
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")
        ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

        ax.plot(times, g1, color="#38bdf8", lw=2.2, label=f"{l1} (n={int(m1.sum())})")
        ax.plot(times, g2, color="#f472b6", lw=2.2, label=f"{l2} (n={int(m2.sum())})")
        ax.axvline(0.0, color="#fbbf24", linestyle="--", lw=1.2)
        ax.set_xlabel("Time relative to event (s)")
        ax.set_ylabel("Evoked GFP / RMS (µV)")
        title_scope = f"Shaft {prefix}" if prefix else f"All {len(epochs.ch_names)} channels"
        ax.set_title(f"Phase-Locked Evoked ERP (GFP) — {title_scope}")
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

        return {
            "summary": f"已计算锁相诱发 GFP：{l1} ({int(m1.sum())} trials) vs {l2} ({int(m2.sum())} trials)，通道数 {len(epochs.ch_names)}。",
            "metrics": {
                "n_channels": len(epochs.ch_names),
                "n_group1": int(m1.sum()),
                "n_group2": int(m2.sum()),
            },
            "image_path": str(out_png),
        }
