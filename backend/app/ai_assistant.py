"""AI Assistant powered by OpenCode Go (OpenAI-compatible) + DeepSeek Flash 1017."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import urllib.request
import urllib.error


DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash-1017"

SYSTEM_PROMPT = (
    "你是 Judgement Bias SEEG 认知神经科学实验的资深数据分析助手，专门协助研究生和师弟完成"
    "标准化 SOP 质控与结果解读。你的回答必须基于提供的真实数据指标，简洁、专业、直接给出"
    "质控结论（合格/需注意）、可能原因和具体下一步建议，使用中文 Markdown 格式。"
)


class AIConfigStore:
    def __init__(self, config_path: Path):
        self.path = config_path

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return {
                    "base_url": data.get("base_url") or DEFAULT_BASE_URL,
                    "model": data.get("model") or DEFAULT_MODEL,
                    "api_key": data.get("api_key") or "",
                }
            except Exception:
                pass
        return {"base_url": DEFAULT_BASE_URL, "model": DEFAULT_MODEL, "api_key": ""}

    def save(self, base_url: str, model: str, api_key: str | None = None) -> dict[str, Any]:
        cur = self.load()
        new_cfg = {
            "base_url": (base_url or DEFAULT_BASE_URL).strip().rstrip("/"),
            "model": (model or DEFAULT_MODEL).strip(),
            "api_key": cur["api_key"] if api_key is None or api_key == "" else api_key.strip(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.public_view(new_cfg)

    def public_view(self, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        c = cfg or self.load()
        key = c.get("api_key", "")
        masked = (key[:6] + "…" + key[-4:]) if len(key) > 12 else ("已配置" if key else "")
        return {
            "base_url": c["base_url"],
            "model": c["model"],
            "has_api_key": bool(key),
            "api_key_masked": masked,
        }


def ask_opencode_go(
    cfg: dict[str, Any],
    step: str,
    question: str,
    context_payload: dict[str, Any],
    override_api_key: str = "",
) -> dict[str, Any]:
    """Send a context-aware analysis query to OpenCode Go (/chat/completions)."""
    api_key = (override_api_key or cfg.get("api_key") or "").strip()
    if not api_key:
        raise ValueError("请先在右上角「AI 设置」中粘贴您的 OpenCode Go API Key。")

    base_url = (cfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    model = (cfg.get("model") or DEFAULT_MODEL).strip()
    url = f"{base_url}/chat/completions"

    ctx_str = json.dumps(context_payload, ensure_ascii=False, indent=2)
    if len(ctx_str) > 12000:
        ctx_str = ctx_str[:12000] + "\n...(truncated)"

    user_msg = (
        f"当前所在 SOP 步骤：{step}\n"
        f"当前被试数据摘要（JSON）：\n```json\n{ctx_str}\n```\n\n"
        f"用户问题 / 请求：{question}"
    )

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.3,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "JudgementBiasStudio/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"OpenCode Go API 返回 HTTP {e.code}: {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"连接 OpenCode Go 失败: {e}") from e

    choices = raw.get("choices") or []
    answer = choices[0]["message"]["content"] if choices else str(raw)
    return {
        "model": raw.get("model", model),
        "answer": answer,
        "usage": raw.get("usage"),
    }


def ask_opencode_go_multimodal(
    cfg: dict[str, Any],
    content_parts: list[dict[str, Any]],
    override_api_key: str = "",
) -> dict[str, Any]:
    """Send multimodal (text + base64 image_url) request to OpenCode Go for clinical record extraction."""
    api_key = (override_api_key or cfg.get("api_key") or "").strip()
    if not api_key:
        raise ValueError("请先在右上角「⚙ AI 设置」中粘贴您的 OpenCode Go API Key。")

    base_url = (cfg.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    model = (cfg.get("model") or DEFAULT_MODEL).strip()
    url = f"{base_url}/chat/completions"

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是临床神经外科与认知神经科学病历结构化助手。严格依据提供的图片与文字提取客观临床信息，绝不编造未出现的数据。",
                },
                {"role": "user", "content": content_parts},
            ],
            "temperature": 0.1,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "JudgementBiasStudio/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"OpenCode Go API 返回 HTTP {e.code}: {err_body}") from e
    except Exception as e:
        raise RuntimeError(f"连接 OpenCode Go 失败: {e}") from e

    choices = raw.get("choices") or []
    answer = choices[0]["message"]["content"] if choices else str(raw)
    return {"model": raw.get("model", model), "answer": answer, "usage": raw.get("usage")}
