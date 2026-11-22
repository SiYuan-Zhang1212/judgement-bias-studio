"""Shared matplotlib styling: CJK font resolution + the Studio dark theme.

The repo's original figures used English axis labels, so no CJK font was ever
configured; any Chinese label renders as tofu boxes.  Reports and figures from the
statistics layer are Chinese (the SOP is written in Chinese and the hand-off
audience reads Chinese), so font resolution has to happen once, in one place.

The font list covers macOS (PingFang / Heiti / Songti) and the Linux boxes used
for the local SEEG pipeline (Noto CJK / WenQuanYi).  If nothing is found we keep
matplotlib's default and emit a warning instead of silently drawing boxes.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_jbs")

# Checked in order; the first one that exists wins.
_CJK_CANDIDATES = (
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux (this workstation / most clusters)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    # Windows (in case someone runs the Studio from a Windows box)
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)

_FONT_FAMILY: str | None = None
_RESOLVED = False

# Dark theme, matching the pre-existing QC figures in seeg_preprocess.py.
BG = "#0f172a"
PANEL = "#1e293b"
FG = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "#94a3b8"
SPINE = "#334155"
ACCENT = ("#64748b", "#38bdf8", "#34d399", "#f472b6", "#fbbf24", "#f43f5e")


def cjk_font_family() -> str:
    """Register the first available CJK font and return its family name.

    Returns ``"DejaVu Sans"`` (matplotlib's default) when nothing is installed, so
    callers always get a usable family string.  Resolution happens once per process.
    """
    global _FONT_FAMILY, _RESOLVED
    if _RESOLVED:
        return _FONT_FAMILY or "DejaVu Sans"

    _RESOLVED = True
    try:
        from matplotlib import font_manager as fm
    except Exception:
        _FONT_FAMILY = None
        return "DejaVu Sans"

    for path in _CJK_CANDIDATES:
        p = Path(path)
        if not p.exists():
            continue
        try:
            fm.fontManager.addfont(str(p))
            _FONT_FAMILY = fm.FontProperties(fname=str(p)).get_name()
            return _FONT_FAMILY
        except Exception:
            continue

    warnings.warn(
        "未找到中文字体，图中中文可能显示为方框。macOS 装 PingFang 即可；"
        "Linux 可安装 fonts-noto-cjk 或 fonts-wqy-microhei。",
        RuntimeWarning,
        stacklevel=2,
    )
    return "DejaVu Sans"


def apply_dark_theme(plt) -> None:
    """Global rcParams for the Studio's dark figures (CJK-safe font stack)."""
    family = cjk_font_family()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = SPINE
    plt.rcParams["axes.linewidth"] = 0.8


def style_axes(ax) -> None:
    """Per-axes cosmetics (facecolor, ticks, grid, spines)."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#cbd5e1", labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color("#f8fafc")
    ax.grid(True, linestyle="--", alpha=0.2, color=GRID)
    for s in ax.spines.values():
        s.set_color(SPINE)
