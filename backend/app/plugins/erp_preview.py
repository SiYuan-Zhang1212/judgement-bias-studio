"""Example Plugin #1: Phase-Locked Evoked Potential (ERP) GFP Preview.

Correctly averages across trials FIRST to extract phase-locked evoked potentials per channel
`(n_ch, n_times)`, then computes Global Field Power (RMS across channels) — avoiding the
common pitfall of averaging single-trial RMS (which reflects ongoing total amplitude).

Compliance note (SOP rule R1)
-----------------------------
`anchor_vs_mid` is NOT a clean condition contrast.  Anchor tones carry a fixed correct
key (so they co-vary with hand), always carry feedback (win/avoid), and are the only
tones with a valence label; ambiguous mid tones have none of those.  The contrast is
therefore labelled "锚点音 vs 模糊中音（手侧/反馈/效价共同变化）" and the plugin reports
the anchor hand-agreement rate alongside it.  For a defensible condition effect use the
`condition_stats` plugin, which reports A/B/C flavours and the hand diagnostics.
"""

from __future__ import annotations

from typing import Any
import os
import numpy as np

from .. import stats_core as sc
from ..plugin_base import AnalysisPlugin, PluginContext
from .condition_stats import build_design_columns, hand_agreement

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_jbs")


class ErpPreviewPlugin(AnalysisPlugin):
    plugin_id = "erp_preview"
    name = "条件平均 ERP（锁相诱发电位 GFP）预览"
    version = "1.2.0"
    description = "先按试次平均得到各通道锁相诱发电位（Evoked），再计算跨通道 GFP/RMS。这是描述性预览，不是条件效应检验；做效应请用 condition_stats 插件。"
    category = "electrophysiology"
    requires = ["preprocess"]
    params_schema = [
        {
            "key": "contrast",
            "label": "对比条件",
            "type": "select",
            "default": "anchor_vs_mid",
            "options": [
                {"value": "anchor_vs_mid", "label": "锚点音 vs 模糊中间音（手侧/反馈/效价共同变化）"},
                {"value": "pos_vs_neg_choice", "label": "正性选择 vs 负性选择（与 RT/运动相关）"},
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
        if meta is None:
            raise ValueError(
                "Epochs 缺少 metadata，无法按条件分组。请用 Step 5 重新生成（勾选保留 metadata）。"
            )

        # R1: quantify how entangled the contrast is with hand before drawing anything.
        cols = build_design_columns(meta)
        agree = hand_agreement(cols)
        rate = agree["agree_rate"]
        hand_entangled = bool(np.isfinite(rate) and rate >= sc.thresholds()["hand_agree_95"])

        if contrast == "pos_vs_neg_choice":
            m1 = np.asarray([c == "pos" for c in cols["choice_sem"]])
            m2 = np.asarray([c == "neg" for c in cols["choice_sem"]])
            l1, l2 = "Positive Choice", "Negative Choice"
            contrast_note = (
                "正性 vs 负性选择与按键手侧几乎完全共变（右手答正/左手的映射每个被试固定），"
                "且与 RT 相关；本图只能当运动/反应相关的描述，不能当效价效应。"
            )
        else:
            m1 = ~cols["is_mid"]
            m2 = cols["is_mid"]
            l1, l2 = "Anchor Tones", "Ambiguous Mids"
            contrast_note = (
                "锚点音与模糊中音同时改变了手侧（锚点按键固定）、反馈（锚点有赢/避，中音无）"
                "和效价标签（只有锚点有）；三者无法在本对比内分离。"
            )

        data = epochs.get_data() * 1e6  # (n_trials, n_ch, n_times) in uV
        times = epochs.times

        def evoked_gfp(mask_arr: np.ndarray) -> np.ndarray:
            if not np.any(mask_arr):
                return np.zeros(len(times), dtype=float)
            # Average across trials FIRST -> (n_ch, n_times), then RMS across channels (GFP)
            evoked = np.mean(data[mask_arr], axis=0)
            return np.sqrt(np.mean(evoked**2, axis=0))

        g1 = evoked_gfp(m1)
        g2 = evoked_gfp(m2)

        out_png = ctx.output_dir / f"plugin_erp_{contrast}.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.9), dpi=150, facecolor="#0f172a")
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
        title = f"Evoked GFP — {title_scope}"
        if contrast == "anchor_vs_mid" and hand_entangled:
            title += "\n[条件（与手侧共变）· 非效价/决策效应]"
        ax.set_title(title, fontsize=10)
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", fontsize=8)
        fig.text(
            0.01, 0.01,
            contrast_note
            + (f"  锚点手侧符合率 {agree['n_agree']}/{agree['n_anchor']}。" if agree["available"] else ""),
            color="#94a3b8", fontsize=6.6, wrap=True,
        )
        fig.tight_layout(rect=[0, 0.055, 1, 1])
        fig.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

        label = "条件（与手侧共变）" if hand_entangled else "条件"
        return {
            "summary": (
                f"已计算锁相诱发 GFP：{l1} ({int(m1.sum())} trials) vs {l2} ({int(m2.sum())} trials)，"
                f"通道数 {len(epochs.ch_names)}。对比标签：{label}；"
                "这是描述性预览，条件效应请用 condition_stats 插件（A/B/C 三口径）。"
            ),
            "metrics": {
                "n_channels": len(epochs.ch_names),
                "n_group1": int(m1.sum()),
                "n_group2": int(m2.sum()),
                "contrast_label": label,
                "hand_agrees_with_mapping": f"{agree['n_agree']}/{agree['n_anchor']}"
                if agree["available"] else "不可用",
                "hand_agree_rate": agree["agree_rate"],
                "descriptive_only": True,
            },
            "image_path": str(out_png),
        }
