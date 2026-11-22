"""Fault-tolerant pluggable analysis module architecture for Judgement Bias Studio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib
import pkgutil
import sys


@dataclass
class PluginContext:
    subject_id: str
    subject_dir: Path
    output_dir: Path
    behavior_summary_path: Path | None
    epochs_fif_path: Path | None
    localization_json_path: Path | None


class AnalysisPlugin:
    plugin_id: str = ""
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    category: str = "electrophysiology"
    requires: list[str] = []
    params_schema: list[dict[str, Any]] = []

    def descriptor(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "requires": list(self.requires),
            "params_schema": list(self.params_schema),
        }

    def run(self, ctx: PluginContext, params: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


LAST_PLUGIN_ERRORS: list[dict[str, str]] = []


def discover_plugins() -> dict[str, AnalysisPlugin]:
    """Auto-discover all AnalysisPlugin subclasses in `backend.app.plugins` safely.

    A syntax or import error in one plugin logs into `LAST_PLUGIN_ERRORS` and never crashes startup.
    """
    from . import plugins as plugins_pkg

    LAST_PLUGIN_ERRORS.clear()
    registry: dict[str, AnalysisPlugin] = {}
    for modinfo in pkgutil.iter_modules(plugins_pkg.__path__):
        if modinfo.name.startswith("_"):
            continue
        mod_full = f"{plugins_pkg.__name__}.{modinfo.name}"
        try:
            mod = importlib.reload(sys.modules[mod_full]) if mod_full in sys.modules else importlib.import_module(mod_full)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, AnalysisPlugin)
                    and obj is not AnalysisPlugin
                    and getattr(obj, "plugin_id", "")
                ):
                    inst = obj()
                    registry[inst.plugin_id] = inst
        except Exception as exc:
            LAST_PLUGIN_ERRORS.append({"module": modinfo.name, "error": str(exc)})
    return registry
