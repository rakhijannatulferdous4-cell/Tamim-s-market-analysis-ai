"""
AI Trading Debate — crash-proof, future-proof.

Pipeline
────────
 Step 1  Chart Vision  : live configured vision models
                          → live Groq vision model fallback
                          → text-fallback if vision APIs fail
 Step 2  Analyst Votes : every activated vision-capable model
                         (OpenRouter, Groq, Together AI, Gemini, or Hugging Face)
                        <think>…</think> tokens stripped before JSON parsing
Step 3  Final Verdict : Top Groq text model synthesis
                        → FINAL_DECISION + candle recommendation box
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from html import escape
import json
import os
import re
import time

import requests
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Page config — sidebar expanded so the model manager is immediately visible
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Debate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Secret helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_secret(key: str) -> str:
    # User-entered provider keys stay in the server-side Streamlit session.
    # They are never written into browser-side HTML or sent to the client.
    provider_keys = st.session_state.get("provider_keys", {})
    if key in provider_keys and provider_keys[key]:
        return str(provider_keys[key])
    val = ""
    try:
        val = st.secrets.get(key, "") or ""
    except Exception:
        pass
    if not val:
        val = os.environ.get(key, "") or ""
    return val


def require_secret(key: str) -> str:
    val = get_secret(key)
    if not val:
        st.error(
            f"**{key}** is missing.\n\n"
            "Add it to Streamlit secrets or the deployment environment."
        )
        st.stop()
    return val


def _api_key(key: str) -> str:
    """Read a provider key without relying on platform-specific services."""
    value = get_secret(key)
    if not value:
        raise RuntimeError(f"{key} is not configured.")
    return value


def _gemini_client(api_key: str | None = None) -> genai.Client:
    return genai.Client(api_key=api_key or _api_key("GEMINI_API_KEY"))


def _groq_client(api_key: str | None = None) -> Groq:
    return Groq(api_key=api_key or _api_key("GROQ_API_KEY"))


def _deepseek_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(
        api_key=api_key or _api_key("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON parser — strips <think> tokens, markdown fences, and bare objects
# ─────────────────────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict:
    raw = text or ""
    # 1. Strip <think>…</think> (Qwen-QwQ / DeepSeek-R1 style)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # 2. Strip markdown fences
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    # 3. Extract first {...} block
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start >= 0:
        try:
            return json.loads(raw[start:end])
        except Exception:
            pass
    # 4. Last-resort text extraction — some models output structured prose
    #    instead of JSON. Honor an explicit vote before scanning explanation
    #    keywords, because a DOWN answer may still discuss bullish risk.
    aliases = {
        "BULLISH": "UP",
        "BUY": "UP",
        "UP": "UP",
        "BEARISH": "DOWN",
        "SELL": "DOWN",
        "DOWN": "DOWN",
        "NEUTRAL": "WAIT",
        "HOLD": "WAIT",
        "WAIT": "WAIT",
    }
    vote_words = "|".join(aliases)
    explicit = re.findall(
        r"\b(?:final\s+)?(?:vote|direction|signal|decision|stance)\b"
        r"\s*(?:is|should\s+be|[:=-])\s*[\"']?("
        + vote_words
        + r")\b",
        raw,
        flags=re.IGNORECASE,
    )
    if explicit:
        vote = aliases[explicit[-1].upper()]
    else:
        txt_up = raw.upper()
        has_up = bool(re.search(r"\bBULLISH\b|\bBUY\b|\bUP\b", txt_up))
        has_down = bool(re.search(r"\bBEARISH\b|\bSELL\b|\bDOWN\b", txt_up))
        vote = "UP" if has_up and not has_down else (
            "DOWN" if has_down and not has_up else "WAIT"
        )
    conf_m = re.search(r'(\d{1,3})\s*%', raw)
    conf   = min(int(conf_m.group(1)), 100) if conf_m else 50
    lines  = [l.strip() for l in raw.split("\n") if len(l.strip()) > 25]
    rsn    = lines[0][:140] if lines else "Model output could not be parsed as JSON."
    return {"vote": vote, "confidence": conf, "reasoning": rsn,
            "_text_extraction": True}


def _safe_error(error: object) -> str:
    """Make provider errors useful without ever rendering credentials."""
    message = str(error or "The provider did not return a response.")
    message = re.sub(
        r"([?&](?:key|api_key|apikey|token|access_token)=)[^&\s]+",
        r"\1[redacted]",
        message,
        flags=re.IGNORECASE,
    )
    message = re.sub(
        r"(Bearer\s+|AIza)[A-Za-z0-9._\-/+=-]+",
        r"\1[redacted]",
        message,
        flags=re.IGNORECASE,
    )
    return message[:240]


def _normalise_vote_fields(result: dict, default_reason: str) -> dict:
    """Keep model-specific vote formats consistent in the UI and prompts."""
    raw_vote = str(
        result.get("vote")
        or result.get("gemini_vote")
        or result.get("direction")
        or "WAIT"
    ).upper()
    result["vote"] = {
        "BULLISH": "UP",
        "BUY": "UP",
        "BEARISH": "DOWN",
        "SELL": "DOWN",
        "NEUTRAL": "WAIT",
        "HOLD": "WAIT",
    }.get(raw_vote, raw_vote if raw_vote in {"UP", "DOWN", "WAIT"} else "WAIT")

    raw_conf = result.get("confidence") or result.get("gemini_confidence") or 50
    try:
        confidence = float(raw_conf)
        if 0 < confidence <= 1:
            confidence *= 100
        result["confidence"] = max(0, min(100, int(round(confidence))))
    except (TypeError, ValueError):
        result["confidence"] = 50

    result["reasoning"] = (
        result.get("reasoning")
        or result.get("gemini_reasoning")
        or default_reason
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Groq model discovery helpers
# ─────────────────────────────────────────────────────────────────────────────
_GROQ_TEXT_EXCLUDE = frozenset(
    # Exclude non-text, non-chat, and reasoning/TTS/embed models.
    # Also exclude third-party models that require separate terms acceptance
    # (canopylabs, orpheus) and compound which hits 413 payload limits.
    [
        "vision", "whisper", "guard", "embed", "tts", "distil",
        "qwq", "think",                          # reasoning chain-of-thought
        "canopylabs", "orpheus",                 # require extra terms acceptance
        "compound",                              # context/payload too large (413)
        "arabic", "saudi",                       # specialized non-JSON models
    ]
)

# Hard fallback list used when the live API call fails
GEMINI_MODEL_ID = "gemini-3-flash-preview"
GEMINI_MODEL_LABEL = "Gemini 3 Flash Preview"


def _discover_groq_vision_models() -> list[str]:
    """
    Return an ordered list of vision-capable Groq model IDs.
    Uses Groq's live input-modality metadata. Never returns retired model IDs.
    """
    try:
        if not get_secret("GROQ_API_KEY"):
            return []
        client = _groq_client()
        listing = client.models.list()
        vision_models = [
            m for m in listing.data
            if "image" in (getattr(m, "input_modalities", []) or [])
        ]
        vision_models.sort(
            key=lambda m: (
                "qwen" not in m.id.lower(),
                "3.6" not in m.id.lower(),
                m.id,
            )
        )
        return [m.id for m in vision_models]
    except Exception:
        return []


def _discover_groq_text_models(n: int = 3) -> list[tuple[str, str]]:
    """
    Return up to `n` text-only model IDs from Groq's live catalog,
    ranked by a quality heuristic. Explicit preferences always rank #1 and #2.
    Returns an empty list when the live catalog cannot be read. This prevents
    retired model IDs from being presented as active models.
    """
    try:
        if not get_secret("GROQ_API_KEY"):
            return []
        client = _groq_client()
        listing = client.models.list()

        candidates: list[tuple[int, str]] = []
        for m in listing.data:
            mid_lower = m.id.lower()
            if any(x in mid_lower for x in _GROQ_TEXT_EXCLUDE):
                continue
            score = 0
            if "qwen"    in mid_lower:     score += 115
            elif "llama-3.3" in mid_lower: score += 110
            elif "llama-3.2" in mid_lower: score += 80
            elif "llama-3.1" in mid_lower: score += 70
            elif "llama-3"  in mid_lower:  score += 60
            elif "gpt-oss" in mid_lower:   score += 55
            elif "mixtral"  in mid_lower:  score += 55
            elif "gemma"    in mid_lower:  score += 50
            else:                           score += 10
            if   any(x in mid_lower for x in ["70b", "72b", "32b"]): score += 30
            elif any(x in mid_lower for x in ["34b", "13b"]):         score += 20
            elif "8b" in mid_lower: score += 15
            elif "7b" in mid_lower: score += 12
            candidates.append((score, m.id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        result = [(mid, mid) for _, mid in candidates[:n]]
        return result
    except Exception:
        return []


def _short_label(model_id: str) -> str:
    """Human-readable short label for any Groq model ID."""
    mid = model_id.lower()
    # Handle meta-llama/ prefix
    base = mid.split("/")[-1]
    for size in ["120b", "72b", "70b", "34b", "32b", "27b", "20b", "13b", "9b", "8b", "7b"]:
        if size in base:
            if "llama-4-scout"    in base: return f"Llama 4 Scout {size.upper()}"
            if "llama-4-maverick" in base: return f"Llama 4 Maverick {size.upper()}"
            if "llama-3.3"  in base: return f"Llama 3.3 {size.upper()}"
            if "llama-3.2"  in base: return f"Llama 3.2 {size.upper()}"
            if "llama-3.1"  in base: return f"Llama 3.1 {size.upper()}"
            if "llama-3"    in base: return f"Llama 3 {size.upper()}"
            if "mixtral"    in base: return f"Mixtral {size.upper()}"
            if "gemma"      in base: return f"Gemma {size.upper()}"
            if "qwen"       in base: return f"Qwen {size.upper()}"
    return model_id.split("/")[-1].replace("-", " ").title()


def _provider_configured(provider: str) -> bool:
    """Return whether the provider has a credential available at runtime."""
    key_by_provider = {
        "Groq": "GROQ_API_KEY",
        "OpenRouter": "OPENROUTER_API_KEY",
        "Together AI": "TOGETHER_API_KEY",
        "Google Gemini": "GEMINI_API_KEY",
        "Hugging Face": "HF_TOKEN",
        "GitHub Models": "GITHUB_TOKEN",
        "Cerebras Cloud": "CEREBRAS_API_KEY",
        "Cloudflare Workers AI": "CLOUDFLARE_API_KEY",
        "Mistral AI": "MISTRAL_API_KEY",
        "Fireworks AI": "FIREWORKS_API_KEY",
        "DeepInfra": "DEEPINFRA_API_KEY",
        "SambaNova Cloud": "SAMBANOVA_API_KEY",
        "Novita AI": "NOVITA_API_KEY",
        "Hyperbolic": "HYPERBOLIC_API_KEY",
        "GlHF (glhf.chat)": "GLHF_API_KEY",
    }
    key_name = key_by_provider.get(provider, "")
    return bool(get_secret(key_name) or (
        provider == "Hugging Face" and get_secret("HUGGINGFACE_API_KEY")
    ))


_PROVIDER_CONFIG = {
    "OpenRouter": {
        "label": "OpenRouter · recommended",
        "key": "OPENROUTER_API_KEY",
        "models_url": "https://openrouter.ai/api/v1/models",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "kind": "openai",
    },
    "Groq": {
        "key": "GROQ_API_KEY",
        "models_url": "https://api.groq.com/openai/v1/models",
        "chat_url": "https://api.groq.com/openai/v1/chat/completions",
        "kind": "openai",
    },
    "Together AI": {
        "key": "TOGETHER_API_KEY",
        "models_url": "https://api.together.xyz/v1/models",
        "chat_url": "https://api.together.xyz/v1/chat/completions",
        "kind": "openai",
    },
    "Google Gemini": {
        "key": "GEMINI_API_KEY",
        "models_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "kind": "gemini",
    },
    "Hugging Face": {
        "key": "HF_TOKEN",
        "models_url": "https://huggingface.co/api/models",
        "chat_url": "https://router.huggingface.co/v1/chat/completions",
        "kind": "openai",
    },
    "GitHub Models": {
        "label": "GitHub Models",
        "key": "GITHUB_TOKEN",
        "models_url": "https://models.github.ai/catalog/models",
        "chat_url": "https://models.github.ai/inference/chat/completions",
        "kind": "openai",
    },
    "Cerebras Cloud": {
        "label": "Cerebras Cloud",
        "key": "CEREBRAS_API_KEY",
        "models_url": "https://api.cerebras.ai/v1/models",
        "chat_url": "https://api.cerebras.ai/v1/chat/completions",
        "kind": "openai",
    },
    "Cloudflare Workers AI": {
        "label": "Cloudflare Workers AI · requires Account ID",
        "key": "CLOUDFLARE_API_KEY",
        "account_key": "CLOUDFLARE_ACCOUNT_ID",
        "models_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search",
        "chat_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions",
        "kind": "openai",
    },
    "Mistral AI": {
        "label": "Mistral AI",
        "key": "MISTRAL_API_KEY",
        "models_url": "https://api.mistral.ai/v1/models",
        "chat_url": "https://api.mistral.ai/v1/chat/completions",
        "kind": "openai",
    },
    "Fireworks AI": {
        "label": "Fireworks AI",
        "key": "FIREWORKS_API_KEY",
        "models_url": "https://api.fireworks.ai/inference/v1/models",
        "chat_url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "kind": "openai",
    },
    "DeepInfra": {
        "label": "DeepInfra",
        "key": "DEEPINFRA_API_KEY",
        "models_url": "https://api.deepinfra.com/v1/openai/models",
        "chat_url": "https://api.deepinfra.com/v1/openai/chat/completions",
        "kind": "openai",
    },
    "SambaNova Cloud": {
        "label": "SambaNova Cloud",
        "key": "SAMBANOVA_API_KEY",
        "models_url": "https://api.sambanova.ai/v1/models",
        "chat_url": "https://api.sambanova.ai/v1/chat/completions",
        "kind": "openai",
    },
    "Novita AI": {
        "label": "Novita AI",
        "key": "NOVITA_API_KEY",
        "models_url": "https://api.novita.ai/openai/v1/models",
        "chat_url": "https://api.novita.ai/openai/v1/chat/completions",
        "kind": "openai",
    },
    "Hyperbolic": {
        "label": "Hyperbolic",
        "key": "HYPERBOLIC_API_KEY",
        "models_url": "https://api.hyperbolic.xyz/v1/models",
        "chat_url": "https://api.hyperbolic.xyz/v1/chat/completions",
        "kind": "openai",
    },
    "GlHF (glhf.chat)": {
        "label": "GlHF (glhf.chat)",
        "key": "GLHF_API_KEY",
        "models_url": "https://glhf.chat/api/openai/v1/models",
        "chat_url": "https://glhf.chat/api/openai/v1/chat/completions",
        "kind": "openai",
    },
}


def _provider_key(provider: str) -> str:
    """Read a configured provider key without ever rendering its value."""
    config = _PROVIDER_CONFIG[provider]
    key = get_secret(config["key"])
    if provider == "Hugging Face" and not key:
        key = get_secret("HUGGINGFACE_API_KEY")
    return key


def _provider_account_id(provider: str) -> str:
    account_key = _PROVIDER_CONFIG[provider].get("account_key", "")
    return get_secret(account_key) if account_key else ""


def _provider_label(provider: str) -> str:
    """Return the display name without exposing provider credentials."""
    config = _PROVIDER_CONFIG.get(provider, {})
    return str(config.get("label") or provider)


def _provider_chat_url(provider: str, account_id_override: str | None = None) -> str:
    """Resolve provider-specific URL placeholders before making an inference call."""
    url = _PROVIDER_CONFIG[provider]["chat_url"]
    account_id = (
        account_id_override
        if account_id_override is not None
        else _provider_account_id(provider)
    )
    if "{account_id}" in url:
        if not account_id:
            raise RuntimeError(
                "Cloudflare Workers AI also requires CLOUDFLARE_ACCOUNT_ID "
                "in Streamlit secrets or the deployment environment."
            )
        url = url.replace("{account_id}", account_id)
    return url


def _model_supports_images(provider: str, item: dict) -> bool:
    """Conservatively identify models that can receive image inputs."""
    modalities = item.get("input_modalities") or item.get("modalities") or []
    if isinstance(modalities, str):
        modalities = re.split(r"[, +]", modalities.lower())
    modalities = {str(value).lower() for value in modalities}
    architecture = item.get("architecture") or {}
    if isinstance(architecture, dict):
        architecture_modalities = architecture.get("input_modalities") or []
        if isinstance(architecture_modalities, str):
            architecture_modalities = re.split(r"[, +]", architecture_modalities.lower())
        modalities.update(str(value).lower() for value in architecture_modalities)
        modality_text = str(architecture.get("modality", "")).lower()
    else:
        modality_text = str(architecture).lower()
    pipeline = str(item.get("pipeline_tag", "")).lower()
    model_id = str(item.get("id") or item.get("name") or "").lower()
    explicit = "image" in modalities or "vision" in modalities
    described = "image" in modality_text or "vision" in modality_text
    hf_vision = pipeline in {"image-text-to-text", "visual-question-answering"}
    known_vision_id = any(token in model_id for token in (
        "vision", "vl-", "-vl", "llava", "qwen3.6", "qwen2-vl",
        "qwen2.5-vl", "gemma-3", "pixtral", "internvl", "minicpm-v",
    ))
    if provider == "Google Gemini":
        methods = {
            str(v).lower() for v in item.get("supported_generation_methods", [])
        }
        return (
            "generatecontent" in methods
            and (
                explicit
                or described
                or model_id.startswith("gemini-")
            )
            and not any(
                token in model_id
                for token in (
                    "embedding", "aqa", "tts", "imagen", "veo",
                    "antigravity", "deep-research", "research",
                )
            )
        )
    return explicit or described or hf_vision or known_vision_id


def _normalise_provider_model(provider: str, item: dict) -> dict | None:
    model_id = str(item.get("id") or item.get("model") or item.get("name") or "").strip()
    if not model_id:
        return None
    if model_id.startswith("models/"):
        model_id = model_id[7:]
    item = {**item, "id": model_id}
    return {
        "provider": provider,
        "id": model_id,
        "name": _short_label(model_id),
        "vision": _model_supports_images(provider, item),
        "kind": _PROVIDER_CONFIG[provider]["kind"],
    }


def _model_is_chat_capable(model_id: str, item: dict) -> bool:
    """Keep catalog entries that can actually answer a debate prompt."""
    identifier = model_id.lower()
    blocked = (
        "whisper", "embed", "embedding", "rerank", "moderation",
        "safety", "tts", "text-to-speech", "image-generation",
        "flux", "stable-diffusion", "prompt-guard", "prompt_guard",
        "orpheus",
    )
    if any(token in identifier for token in blocked):
        return False
    capabilities = item.get("capabilities") or item.get("supported_generation_methods") or []
    if isinstance(capabilities, str):
        capabilities = [capabilities]
    if capabilities and not any(
        token in str(cap).lower()
        for cap in capabilities
        for token in ("chat", "generate", "completion", "generatecontent")
    ):
        return False
    return True


def _discover_provider_models(provider: str, api_key: str) -> list[dict]:
    """Fetch the provider's usable model catalog, retaining text-only models."""
    config = _PROVIDER_CONFIG[provider]
    headers = {"Authorization": "Bearer " + api_key}
    params = {}
    models_url = config["models_url"]
    account_id = _provider_account_id(provider)
    if "{account_id}" in models_url:
        if not account_id:
            raise RuntimeError(
                "Cloudflare Workers AI also requires CLOUDFLARE_ACCOUNT_ID "
                "in Streamlit secrets or the deployment environment."
            )
        models_url = models_url.replace("{account_id}", account_id)
    if provider == "Google Gemini":
        # Keep the credential out of request URLs and therefore out of
        # exception messages, proxy logs, and the rendered UI.
        headers = {"x-goog-api-key": api_key}
    if provider == "Hugging Face":
        params = {"limit": 200}
    response = requests.get(
        models_url,
        headers=headers,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        raw_models = (
            payload.get("data")
            or payload.get("models")
            or payload.get("items")
            or payload.get("result")
            or payload
        )
    else:
        raw_models = payload
    if provider == "Google Gemini":
        raw_models = payload.get("models", [])
        for item in raw_models:
            item["supported_generation_methods"] = item.get("supportedGenerationMethods", [])
    if not isinstance(raw_models, list):
        raise RuntimeError("The provider returned an unexpected /models response.")
    models = []
    for item in raw_models:
        if isinstance(item, str):
            item = {"id": item}
        if not isinstance(item, dict):
            continue
        if item.get("active") is False:
            continue
        model = _normalise_provider_model(provider, item)
        if model and _model_is_chat_capable(model["id"], item):
            models.append(model)
    models.sort(key=lambda model: (model["provider"], model["name"].lower(), model["id"]))
    if not models:
        raise RuntimeError("The provider returned no usable models for this API key.")
    return models


def _active_models() -> list[dict]:
    return list(st.session_state.get("active_models", []))


def _retire_model(model: dict) -> None:
    """Remove a model that proved unavailable from both active and catalog lists."""
    identity = (model.get("provider"), model.get("id"))
    for state_key in ("active_models", "available_models"):
        st.session_state[state_key] = [
            item for item in st.session_state.get(state_key, [])
            if (item.get("provider"), item.get("id")) != identity
        ]


def _model_label(model: dict) -> str:
    return f"{model['name']} · {model['provider']}"


def _openai_vision_request(
    model: dict,
    api_key: str,
    image_bytes: bytes,
    mime: str,
    prompt: str,
    account_id_override: str | None = None,
) -> str:
    config = _PROVIDER_CONFIG[model["provider"]]
    data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    response = requests.post(
        _provider_chat_url(model["provider"], account_id_override),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json={
            "model": model["id"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert chart-reading AI. Return valid JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "temperature": 0.3,
            "max_tokens": 1800,
        },
        timeout=(8, 60),
    )
    response.raise_for_status()
    body = response.json()
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The provider returned no model response.") from exc


def _gemini_vision_request(
    model: dict, api_key: str, image_bytes: bytes, mime: str, prompt: str
) -> str:
    response = _gemini_client(api_key).models.generate_content(
        model=model["id"],
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1200,
            response_mime_type="application/json",
        ),
    )
    if not response.text:
        raise RuntimeError("Gemini returned no model response.")
    return response.text


def _call_activated_model(
    model: dict,
    image_bytes: bytes,
    mime: str,
    prompt: str,
    api_key_override: str | None = None,
    account_id_override: str | None = None,
) -> dict:
    """Send the image to vision models; text models are handled separately."""
    api_key = (
        api_key_override
        if api_key_override is not None
        else _provider_key(model["provider"])
    )
    if not api_key:
        raise RuntimeError(model["provider"] + " API key is not configured.")
    if model["kind"] == "gemini":
        raw = _gemini_vision_request(model, api_key, image_bytes, mime, prompt)
    else:
        raw = _openai_vision_request(
            model, api_key, image_bytes, mime, prompt, account_id_override
        )
    result = parse_json(raw)
    result.setdefault("model", model["name"])
    result["_platform"] = model["provider"]
    result["_provider"] = model["provider"]
    result["_model_id"] = model["id"]
    result["_active_model"] = model
    result["vision_model"] = _model_label(model)
    result["vision_provider"] = model["provider"]
    return _normalise_vote_fields(
        result, "Image analysis completed by " + model["name"] + "."
    )


def _openai_text_request(
    model: dict,
    api_key: str,
    prompt: str,
    account_id_override: str | None = None,
) -> str:
    payload = {
        "model": model["id"],
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 700,
    }
    response = None
    for attempt in range(4):
        response = requests.post(
            _provider_chat_url(model["provider"], account_id_override),
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(8, 60),
        )
        if response.status_code != 429 or attempt == 3:
            break
        retry_after = response.headers.get("retry-after", "")
        try:
            delay = min(max(float(retry_after), 2.0), 30.0)
        except ValueError:
            delay = 5.0 * (attempt + 1)
        time.sleep(delay)
    response.raise_for_status()
    body = response.json()
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The provider returned no model response.") from exc


def _call_activated_text_model(
    model: dict,
    prompt: str,
    api_key_override: str | None = None,
    account_id_override: str | None = None,
) -> dict:
    api_key = (
        api_key_override
        if api_key_override is not None
        else _provider_key(model["provider"])
    )
    if not api_key:
        raise RuntimeError(model["provider"] + " API key is not configured.")
    if model["kind"] == "gemini":
        response = _gemini_client(api_key).models.generate_content(
            model=model["id"],
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=900,
                response_mime_type="application/json",
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no model response.")
        raw = response.text
    else:
        raw = _openai_text_request(
            model, api_key, prompt, account_id_override
        )
    result = parse_json(raw)
    result.setdefault("model", model["name"])
    result["_platform"] = model["provider"]
    result["_provider"] = model["provider"]
    result["_model_id"] = model["id"]
    result["_active_model"] = model
    return _normalise_vote_fields(result, "Model completed the debate response.")


def _call_activated_chat(
    model: dict, image_bytes: bytes, mime: str, message: str
) -> str:
    prompt = (
        "You are testing an AI trading chart reader. Carefully inspect the attached "
        "chart image, then answer the trader's question. Mention uncertainty when "
        "labels or indicators are not visible. Do not invent values.\n\n"
        "Trader question:\n" + message
    )
    api_key = _provider_key(model["provider"])
    if not api_key:
        raise RuntimeError(model["provider"] + " API key is not configured.")
    if model["kind"] == "gemini":
        return _gemini_vision_request(model, api_key, image_bytes, mime, prompt)
    return _openai_vision_request(model, api_key, image_bytes, mime, prompt)


def _extract_user_timeframe(extra: str) -> tuple[str, int] | None:
    """Extract an explicitly supplied candle timeframe from trader notes."""
    text = " ".join(str(extra or "").split())
    patterns = (
        r"\b(\d{1,4})\s*[- ]?(?:minute|minutes|min|mins)\b",
        r"\b(\d{1,4})\s*m(?:in)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            minutes = int(match.group(1))
            if 1 <= minutes <= 1440:
                return f"{minutes}-minute", minutes
    return None


def _apply_user_context(chart_data: dict, extra: str) -> dict:
    """Make trader-supplied context authoritative over visual guesses."""
    context = str(extra or "").strip()
    chart_data["user_context"] = context
    timeframe = _extract_user_timeframe(context)
    if timeframe:
        label, minutes = timeframe
        chart_data["timeframe"] = label
        chart_data["timeframe_minutes"] = minutes
        chart_data["timeframe_source"] = "Trader-provided"
        summary = str(chart_data.get("technical_summary") or "").strip()
        confirmation = f"Trader-confirmed chart timeframe: {label}."
        if confirmation not in summary:
            chart_data["technical_summary"] = (
                confirmation + (" " + summary if summary else "")
            )
    return chart_data


def _authoritative_context_block(extra: str = "") -> str:
    context = str(extra or "").strip()
    if not context:
        return ""
    return (
        "\n\n=== TRADER-PROVIDED CONTEXT — AUTHORITATIVE ===\n"
        + context
        + "\n=== END TRADER-PROVIDED CONTEXT ===\n"
        "The text between those markers is required input, not optional metadata. "
        "Read every quoted sentence and direct question before analysing. Treat "
        "explicit asset, timeframe, session, and question details above as facts "
        "supplied by the trader. Do not replace them with a visual guess. "
        "Use the supplied timeframe for every hold-duration and trade recommendation, "
        "and explicitly acknowledge it in your reasoning. If the trader asks where "
        "or whether to trade, answer that question directly with UP, DOWN, or WAIT, "
        "the relevant chart level or condition when available, and the reason.\n"
    )


def _activated_chart_prompt(extra: str = "") -> str:
    return (
        "Read the attached trading chart image carefully. Every activated model "
        "must analyze the image itself; do not rely on another model's output. "
        "Do not invent chart levels that are not visible. "
        + _authoritative_context_block(extra)
        + "\n"
        + CHART_PROMPT_TEXT + "\n" + CHART_ANALYSIS_JSON_SPEC + "\n"
        "Also include these committee fields: "
        '"analysis":"brief chart-based analysis",'
        '"key_risks":["specific visible or uncertainty risk"],'
        '"vote":"UP"|"DOWN"|"WAIT",'
        '"confidence":0-100,'
        '"reasoning":"one clear sentence for the vote".\n'
        "Return ONLY valid JSON."
    )


def step1_analyze_active_models(
    image_bytes: bytes, mime: str, extra: str, models: list[dict]
) -> tuple[dict, list[dict], list[tuple[str, str]], str]:
    """Send the original image to every activated vision model."""
    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    prompt = _activated_chart_prompt(extra)
    for model in models:
        try:
            results.append(_call_activated_model(model, image_bytes, mime, prompt))
        except Exception as exc:
            failures.append((_model_label(model), _safe_error(exc)))

    if not results:
        fallback = dict(_TEXT_FALLBACK_DATA)
        fallback["vision_provider"] = "No active vision model"
        fallback["vision_model"] = "Offline"
        return fallback, results, failures, "🚨 No activated vision model could read the chart."

    chart_data = dict(results[0])
    chart_data["vision_model"] = _model_label(results[0]["_active_model"])
    chart_data["vision_provider"] = results[0]["_platform"]
    chart_data["active_model_count"] = len(results)
    chart_data["vision_results"] = results
    chart_data["groq_vision_model"] = "Not used"
    chart_data["groq_vision_vote"] = "—"
    chart_data["groq_vision_confidence"] = 0
    chart_data["groq_vision_reasoning"] = "Activated model results are shown below."
    status = (
        "✅ " + str(len(results)) + " activated vision model(s) read the chart."
        if not failures
        else "⚠️ " + str(len(results)) + " model(s) read the chart; "
        + str(len(failures)) + " failed."
    )
    return chart_data, results, failures, status


def _activated_vote_prompt(chart_data: dict, extra: str = "") -> str:
    """Give text-only models chart facts without seeding another model's vote."""
    fact_fields = {
        "asset",
        "timeframe",
        "timeframe_minutes",
        "current_price",
        "trend",
        "support",
        "resistance",
        "indicators",
        "patterns",
        "live_news",
        "news_summary",
        "news_context",
        "news_searched",
    }
    safe_data = {
        key: value for key, value in chart_data.items() if key in fact_fields
    }
    return (
        "You are an independent trading analyst. A vision pass transcribed factual "
        "observations from the uploaded chart into the record below. The record "
        "intentionally excludes the vision model's vote, confidence, reasoning, "
        "analysis, and recommendation. Do not claim that you personally saw pixels. "
        "Derive your own signal from the chart facts and news: UP, DOWN, or WAIT. "
        "DOWN is a valid outcome when bearish evidence is stronger. Return valid JSON "
        "with vote, confidence 0-100, analysis, key_risks, and reasoning. Be specific "
        "and acknowledge uncertainty.\n\n"
        "CHART RECORD:\n" + json.dumps(safe_data, ensure_ascii=False, default=str)
        + _authoritative_context_block(extra)
    )


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Chart vision
# ─────────────────────────────────────────────────────────────────────────────
CHART_ANALYSIS_JSON_SPEC = """\
{
  "asset": "<ticker/name>",
  "timeframe": "<e.g. 10-min, 1H>",
  "timeframe_minutes": <integer minutes per candle>,
  "current_price": "<price visible on chart>",
  "trend": "<Bullish|Bearish|Sideways>",
  "support": ["<level>", "<level>"],
  "resistance": ["<level>", "<level>"],
  "indicators": {"<indicator name>": "<reading and interpretation>"},
  "patterns": ["<pattern>"],
  "technical_summary": "<3-4 sentence technical narrative>",
  "live_news": [
    {"headline": "<text>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<src>"},
    {"headline": "<text>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<src>"},
    {"headline": "<text>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<src>"}
  ],
  "news_summary": "<2-sentence overall news sentiment summary>",
  "vision_model": "<which model performed this analysis>"
}"""

CHART_PROMPT_TEXT = (
    "You are a senior technical analyst and financial journalist.\n\n"
    "Analyse the trading chart image in full detail. "
    "Include support/resistance levels, trend direction, indicators, and chart patterns. "
    "Also provide the latest relevant market news for the asset shown "
    "(use your best knowledge).\n\n"
    "Return ONLY valid JSON — no markdown, no extra text:\n"
)

_TEXT_FALLBACK_DATA: dict = {
    "asset": "Unknown (vision APIs unavailable)",
    "timeframe": "Unknown",
    "timeframe_minutes": 5,
    "current_price": "Unknown",
    "trend": "Unknown — treat as highly volatile",
    "support": [],
    "resistance": [],
    "indicators": {},
    "patterns": [],
    "technical_summary": (
        "Chart image could not be analysed — all vision APIs are currently "
        "unavailable. The AI committee will debate using strict risk-management "
        "assumptions: highly volatile market, no confirmed trend."
    ),
    "live_news": [],
    "news_summary": "No live news available. Assume high uncertainty.",
    "vision_model": "Text Fallback (no vision API available)",
}


def _gemini_vision(image_bytes: bytes, mime: str, extra: str) -> dict:
    """Use the configured Gemini API key to analyse the uploaded chart."""
    client = _gemini_client()
    extra_line = (f"\n\nExtra context: {extra.strip()}") if extra.strip() else ""
    prompt = (
        CHART_PROMPT_TEXT
        + extra_line
        + "\n"
        + CHART_ANALYSIS_JSON_SPEC
        + "\n\nNote: Fill live_news with your best knowledge of recent market events."
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
        ],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    result = parse_json(response.text or "")
    result["vision_model"] = GEMINI_MODEL_LABEL
    result["vision_provider"] = "Gemini"
    result["gemini_vision_model"] = GEMINI_MODEL_LABEL
    _normalise_vote_fields(result, "Gemini chart analysis.")
    result["gemini_vision_vote"] = result["vote"]
    result["gemini_vision_confidence"] = result["confidence"]
    result["gemini_vision_reasoning"] = result["reasoning"]
    return result


def _groq_vision(image_bytes: bytes, mime: str, extra: str,
                 model_id: str, model_label: str) -> dict:
    """
    Call a Groq vision model.
    Image is base64-encoded and sent as a data-URI inside the OpenAI-compatible
    image_url content block — the only format Groq accepts.
    """
    client = _groq_client()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_uri  = f"data:{mime};base64,{b64_image}"

    extra_line = (f"\n\nExtra context: {extra.strip()}") if extra.strip() else ""
    prompt = (
        CHART_PROMPT_TEXT
        + extra_line
        + "\n"
        + CHART_ANALYSIS_JSON_SPEC
        + "\n\nNote: Fill live_news with your best knowledge of recent market events."
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=0.3,
        max_tokens=1800,
    )
    result = parse_json(response.choices[0].message.content)
    result["vision_model"] = model_label
    result["vision_provider"] = "Groq"
    _normalise_vote_fields(result, "Groq vision analysis.")
    return result




# ─────────────────────────────────────────────────────────────────────────────
# Real-time news search via DuckDuckGo (no API key required)
# ─────────────────────────────────────────────────────────────────────────────
def _search_market_news(asset: str, max_results: int = 8) -> list[dict]:
    """
    Fetch the latest market news for `asset` from DuckDuckGo.
    Tries news endpoint first, falls back to text search.
    Never raises — returns [] on any failure.
    """
    if not asset or "Unknown" in asset or len(asset) < 2:
        return []
    try:
        from duckduckgo_search import DDGS
        query = asset + " price market trading analysis today"
        with DDGS() as ddgs:
            results = list(ddgs.news(keywords=query, max_results=max_results, timelimit="w"))
        if results:
            return results
        # Retry without time limit if nothing found in last week
        with DDGS() as ddgs:
            return list(ddgs.news(keywords=asset + " trading outlook", max_results=max_results))
    except Exception:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                return list(ddgs.text(
                    keywords=asset + " price today technical analysis signal",
                    max_results=max_results,
                ))
        except Exception:
            return []


def _inject_real_news(chart_data: dict) -> dict:
    """
    Perform a DuckDuckGo news search for the charted asset and inject
    real headlines into chart_data. Replaces the AI-hallucinated live_news
    with actual web results.  Mutates and returns chart_data.
    """
    asset = chart_data.get("asset", "")
    raw   = _search_market_news(asset)
    if not raw:
        chart_data.setdefault("live_news", [])
        chart_data.setdefault("news_context", "")
        chart_data["news_searched"] = False
        return chart_data

    live_news = []
    context_lines = []
    for n in raw[:8]:
        title  = (n.get("title") or n.get("body", ""))[:160]
        source = n.get("source") or n.get("href", "")[:60]
        date   = n.get("date", "")[:10]
        url    = n.get("url") or n.get("href", "")
        body   = n.get("body", "")[:200]
        live_news.append({
            "headline": title,
            "sentiment": "Neutral",   # analysts will judge sentiment
            "source": source,
            "date": date,
            "url": url,
            "body": body,
        })
        context_lines.append(
            "  • [" + date + "] " + title + " — " + source
        )

    chart_data["live_news"]     = live_news
    chart_data["news_context"]  = "\n".join(context_lines)
    chart_data["news_searched"] = True
    return chart_data


def step1_analyze_chart(image_bytes: bytes, mime: str, extra: str) -> tuple[dict, str]:
    """
    Runs the current live Groq vision model.
    Returns (chart_data, status_message). Never raises.
    """
    groq_data       = None
    groq_error      = ""
    groq_model_used = ""
    for model_id in _discover_groq_vision_models():
        try:
            groq_data       = _groq_vision(image_bytes, mime, extra, model_id, model_id)
            groq_model_used = model_id
            break
        except Exception as exc:
            groq_error += f"{model_id}: {_safe_error(exc)[:60]}; "

    if groq_data:
        chart_data = groq_data
        chart_data["groq_vision_model"]      = groq_model_used
        chart_data["_vision_model_id"]       = groq_model_used
        chart_data["groq_vision_vote"]       = groq_data.get("vote", "WAIT")
        chart_data["groq_vision_confidence"] = groq_data.get("confidence", 50)
        chart_data["groq_vision_reasoning"]  = groq_data.get("reasoning", "")
        status = "✅ **" + groq_model_used.split("/")[-1] + "** ⚡ analyzed the chart."
    else:
        chart_data = dict(_TEXT_FALLBACK_DATA)
        if extra.strip():
            chart_data["technical_summary"] += "  User context: " + extra.strip()
        chart_data["groq_vision_model"]      = "Groq Vision"
        chart_data["_vision_model_id"]       = ""
        chart_data["groq_vision_vote"]       = "Offline"
        chart_data["groq_vision_confidence"] = 0
        chart_data["groq_vision_reasoning"]  = "Vision API unavailable."
        chart_data["vision_provider"]        = "Text fallback"
        status = (
            "🚨 **Groq vision unavailable** — text-fallback mode.\n\n"
            "Groq: " + groq_error[:220]
        )

    return chart_data, status


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Independent analyst votes
# ─────────────────────────────────────────────────────────────────────────────
def build_analyst_prompt(model_name: str, g: dict, short: bool = False) -> str:
    indicators = (
        "\n".join("    " + k + ": " + str(v)
                  for k, v in g.get("indicators", {}).items())
        or "    Not visible in chart image"
    )

    # Prefer real web-searched news; fall back to AI-generated news list
    # short=True is used for 413-retry — truncate news to save tokens
    news_items = g.get("live_news", [])
    if short:
        news_items = news_items[:2]
    news_ctx = g.get("news_context", "")
    if short and news_ctx:
        news_ctx = "\n".join(news_ctx.split("\n")[:3])  # first 3 lines only
    if not news_ctx:
        news_ctx = (
            "\n".join(
                "  • [" + n.get("sentiment", "?") + "] "
                + n.get("headline", "") + " — " + n.get("source", "")
                + (" (" + n.get("date", "") + ")" if n.get("date") else "")
                for n in news_items
            )
            or "  No live news available at this time."
        )

    # Trader's own notes — these override/supplement everything else
    user_ctx = (g.get("user_context") or "").strip()
    user_section = (
        "══════════════════════════════════════════════════════\n"
        "  ⚠️  TRADER'S NOTES — AUTHORITATIVE; READ BEFORE ANALYSING\n"
        "══════════════════════════════════════════════════════\n"
        + user_ctx + "\n"
        "Treat the trader's explicit timeframe and question as facts. "
        "Do not substitute a timeframe guessed from the image. Answer the trader's "
        "direct question instead of giving a generic chart summary.\n\n"
    ) if user_ctx else ""

    if short:
        # Compact version for models with small context windows (413 retry)
        return (
            "You are " + model_name + ", an AI trading analyst.\n"
            + (("TRADER NOTES: " + user_ctx + "\n\n") if user_ctx else "")
            + "Asset: " + g.get("asset", "?") + " | TF: " + g.get("timeframe", "?")
            + " | Trend: " + g.get("trend", "?") + "\n"
            + "Support: " + (", ".join(g.get("support", [])) or "N/A") + "\n"
            + "Resistance: " + (", ".join(g.get("resistance", [])) or "N/A") + "\n"
            + "Summary: " + g.get("technical_summary", "")[:300] + "\n"
            + "News: " + news_ctx[:200] + "\n\n"
            + 'Return ONLY valid JSON: {"model":"' + model_name + '",'
            '"vote":"UP"|"DOWN"|"WAIT","confidence":0-100,'
            '"reasoning":"<one sentence>","analysis":"<brief>","key_risks":["r1"],'
            '"news_sentiment":"BULLISH"|"BEARISH"|"NEUTRAL","technical_score":0}'
        )

    return (
        "You are " + model_name + ", a senior AI quant-analyst and trading strategist.\n\n"
        "Below is everything you need: live chart data, real-time news, and vision AI\n"
        "readings. Study ALL of it rigorously, then produce a high-conviction trade signal.\n\n"

        + user_section +

        "══════════════════════════════════════════════════════\n"
        "  SECTION 1 — CHART TECHNICAL DATA\n"
        "══════════════════════════════════════════════════════\n"
        "Asset        : " + g.get("asset", "Unknown") + "\n"
        "Timeframe    : " + g.get("timeframe", "Unknown") + "\n"
        "Current Price: " + g.get("current_price", "?") + "\n"
        "Trend        : " + g.get("trend", "Unknown") + "\n"
        "Support      : " + (", ".join(g.get("support", [])) or "None identified") + "\n"
        "Resistance   : " + (", ".join(g.get("resistance", [])) or "None identified") + "\n"
        "Indicators:\n" + indicators + "\n"
        "Chart Patterns: " + (", ".join(g.get("patterns", [])) or "None observed") + "\n"
        "Technical Summary:\n  " + g.get("technical_summary", "") + "\n\n"
        "Vision AI Reading (" + g.get("vision_provider", "Vision AI")
        + " read the chart):\n"
        "  " + g.get("vision_model", "Vision Model").split("/")[-1] + " → "
        + str(g.get("gemini_vision_vote", g.get("groq_vision_vote", "?")))
        + " (" + str(g.get("gemini_vision_confidence",
                            g.get("groq_vision_confidence", 0))) + "%) — "
        + g.get("gemini_vision_reasoning",
                g.get("groq_vision_reasoning", "")) + "\n\n"

        "══════════════════════════════════════════════════════\n"
        "  SECTION 2 — REAL-TIME MARKET NEWS (live web search)\n"
        "══════════════════════════════════════════════════════\n"
        + news_ctx + "\n"
        "News Summary (vision AI): " + g.get("news_summary", "N/A") + "\n\n"

        "══════════════════════════════════════════════════════\n"
        "  SECTION 3 — YOUR ANALYSIS CHECKLIST (answer all 8)\n"
        "══════════════════════════════════════════════════════\n"
        "Work through each point; summarise findings in the 'analysis' field. "
        "Then answer the trader's direct question from the notes explicitly in "
        "that analysis and reasoning; do not ignore or paraphrase it away.\n\n"
        "1. TREND STRUCTURE — uptrend, downtrend, or range? Higher highs/lows?\n"
        "2. RSI — overbought (>70), oversold (<30), neutral? Divergence?\n"
        "3. MACD — bullish or bearish crossover? Histogram expanding or contracting?\n"
        "4. SUPPORT / RESISTANCE — price near key level? Breaking out or rejecting?\n"
        "5. VOLUME — confirming the trend? Volume spike, dry-up, or climax?\n"
        "6. PATTERNS — reversal (H&S, double top/bottom) or continuation (flag, pennant)?\n"
        "7. NEWS SENTIMENT — are headlines bullish, bearish, or neutral?\n"
        "8. SYNTHESIS — weighing ALL above: what is the highest-probability next move?\n\n"
        "Return ONLY valid JSON — absolutely no markdown, no extra text:\n"
        '{\n'
        '  "model": "' + model_name + '",\n'
        '  "analysis": "<4-sentence analysis covering ALL 8 checklist points above>",\n'
        '  "key_risks": ["<precise risk 1>", "<precise risk 2>", "<precise risk 3>"],\n'
        '  "vote": "UP"|"DOWN"|"WAIT",\n'
        '  "confidence": 0-100,\n'
        '  "reasoning": "<single clear sentence: the strongest reason for your vote>",\n'
        '  "news_sentiment": "BULLISH"|"BEARISH"|"NEUTRAL",\n'
        '  "technical_score": <integer -100 to 100, negative=bearish, positive=bullish>\n'
        '}'
    )


def _call_groq(model_id: str, model_name: str, chart_data: dict) -> dict:
    client  = _groq_client()
    prompt  = build_analyst_prompt(model_name, chart_data)
    sys_msg = {"role": "system",
               "content": "You are an expert AI trading analyst. Respond with valid JSON only. No markdown, no explanation — pure JSON."}
    user_msg = {"role": "user", "content": prompt}

    # Try with response_format=json_object first (forces compliant models to output JSON)
    raw = None
    try:
        chat = client.chat.completions.create(
            model=model_id,
            messages=[sys_msg, user_msg],
            temperature=0.4,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        raw = chat.choices[0].message.content
    except Exception as e:
        err = str(e)
        # 400 terms-acceptance — fail clearly so user sees which model needs terms
        if "400" in err and "terms" in err.lower():
            raise RuntimeError(
                "Error 400 — model requires terms acceptance at console.groq.com. "
                "Open Groq playground for this model and accept the terms, then retry."
            ) from e
        # 413 request too large — retry with shorter (news-trimmed) prompt
        if "413" in err or "request_too_large" in err:
            short_prompt = build_analyst_prompt(model_name, chart_data, short=True)
            try:
                chat = client.chat.completions.create(
                    model=model_id,
                    messages=[sys_msg, {"role": "user", "content": short_prompt}],
                    temperature=0.4,
                    max_tokens=600,
                )
                raw = chat.choices[0].message.content
            except Exception as e2:
                raise RuntimeError(
                    "Error 413 — prompt too large for this model's context window. "
                    "Remove it or use a model with a larger context."
                ) from e2
        else:
            # Other errors: retry without response_format (some models don't support it)
            try:
                chat = client.chat.completions.create(
                    model=model_id,
                    messages=[sys_msg, user_msg],
                    temperature=0.4,
                    max_tokens=1200,
                )
                raw = chat.choices[0].message.content
            except Exception:
                raise

    result = parse_json(raw)
    result.setdefault("model", model_name)
    result["_platform"] = "Groq"
    result["_model_id"]  = model_id          # stored for Round 2 deliberation
    return result


def _call_gemini(chart_data: dict) -> dict:
    """Run Gemini as an independent analyst in the committee."""
    prompt = build_analyst_prompt(GEMINI_MODEL_LABEL, chart_data)
    client = _gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    result = parse_json(response.text or "")
    result.setdefault("model", GEMINI_MODEL_LABEL)
    result["_platform"] = "Gemini"
    result["_model_id"] = GEMINI_MODEL_ID
    return result


def _call_deepseek(chart_data: dict) -> dict:
    prompt  = build_analyst_prompt("DeepSeek-Chat", chart_data)
    try:
        chat = _deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": "You are an expert AI trading analyst. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1200,
        )
    except Exception as exc:
        if getattr(exc, "status_code", None) == 402:
            raise RuntimeError(
                "💳 DeepSeek is waiting for a top-up (402 Payment Required). "
                "Skipping — remaining analysts will carry the debate."
            ) from exc
        raise
    result = parse_json(chat.choices[0].message.content)
    result.setdefault("model", "DeepSeek-Chat")
    result["_platform"] = "DeepSeek"
    result["_model_id"]  = "deepseek-chat"   # stored for Round 2 deliberation
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DELIBERATION ROUND — AIs share all reasoning and submit final votes
# ─────────────────────────────────────────────────────────────────────────────
def _call_groq_raw(model_id: str, model_name: str, prompt: str) -> dict:
    """Call Groq with a custom prompt (used for deliberation round)."""
    client   = _groq_client()
    sys_msg  = {"role": "system",
                "content": "You are an AI trading analyst in a structured debate. Respond with valid JSON only."}
    user_msg = {"role": "user", "content": prompt}
    raw = None
    try:
        chat = client.chat.completions.create(
            model=model_id, messages=[sys_msg, user_msg],
            temperature=0.3, max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = chat.choices[0].message.content
    except Exception:
        chat = client.chat.completions.create(
            model=model_id, messages=[sys_msg, user_msg],
            temperature=0.3, max_tokens=500,
        )
        raw = chat.choices[0].message.content
    result = parse_json(raw)
    result.setdefault("model", model_name)
    result["_platform"] = "Groq"
    return result


def _call_gemini_raw(model_name: str, prompt: str) -> dict:
    """Call Gemini with a custom prompt for the deliberation round."""
    client = _gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    result = parse_json(response.text or "")
    result.setdefault("model", model_name)
    result["_platform"] = "Gemini"
    result["_model_id"] = GEMINI_MODEL_ID
    return result


def _call_deepseek_raw(model_name: str, prompt: str) -> dict:
    """Call DeepSeek with a custom prompt (used for deliberation round)."""
    try:
        chat = _deepseek_client().chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an AI trading analyst. Respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3, max_tokens=500,
        )
    except Exception as exc:
        if getattr(exc, "status_code", None) == 402:
            raise RuntimeError("DeepSeek 402") from exc
        raise
    result = parse_json(chat.choices[0].message.content)
    result.setdefault("model", model_name)
    result["_platform"] = "DeepSeek"
    return result


def _build_deliberation_prompt(
    model_name: str,
    all_votes: list[dict],
    user_context: str = "",
    timeframe: str = "",
) -> str:
    """
    Build the Round 2 prompt for one analyst: it sees every peer's
    vote + full reasoning from Round 1, then submits its final position.
    """
    from collections import Counter
    positions = ""
    for v in all_votes:
        name = v.get("model", "?")
        tag  = "YOUR initial vote" if name == model_name else name
        positions += (
            tag + ": " + v.get("vote", "?")
            + " (" + str(v.get("confidence", 0)) + "%) — "
            + v.get("reasoning", "") + "\n"
            "  Full analysis: " + v.get("analysis", "") + "\n"
            "  Key risks    : " + ", ".join(v.get("key_risks", [])) + "\n\n"
        )
    majority = Counter(v.get("vote", "WAIT") for v in all_votes).most_common(1)[0][0]
    own_vote = next((v.get("vote", "?")
                     for v in all_votes if v.get("model") == model_name), "?")
    context_block = _authoritative_context_block(user_context)
    timeframe_line = (
        "Trader-confirmed timeframe: " + timeframe + ". Keep this timeframe in "
        "your revised vote and recommendation.\n\n"
        if timeframe
        else ""
    )
    return (
        "You are " + model_name + ", participating in a structured AI trading debate.\n"
        + context_block
        + timeframe_line
        + "All analysts have submitted their Round 1 positions. "
        "Now read every peer's full reasoning below.\n\n"
        "=== ROUND 1 — ALL ANALYST POSITIONS ===\n"
        + positions
        + "========================================\n"
        "Majority vote so far : " + majority + "\n"
        "Your Round 1 vote    : " + own_vote + "\n\n"
        "Deliberate carefully:\n"
        "• Did you miss any pattern or news that colleagues caught?\n"
        "• Is the majority backed by stronger evidence than your view?\n"
        "• Do you have unique insight that justifies dissent?\n\n"
        "Submit your FINAL vote after deliberation.\n\n"
        "Return ONLY valid JSON — no markdown:\n"
        '{"model": "' + model_name + '", '
        '"vote": "UP"|"DOWN"|"WAIT", '
        '"confidence": 0-100, '
        '"reasoning": "<final stance and whether/why you agree or disagree with peers>", '
        '"changed_mind": true|false}'
    )


def run_deliberation_round(
    analyst_votes: list[dict], chart_data: dict | None = None
) -> list[dict]:
    """
    Round 2: every analyst sees all peers' full reasoning and gives a
    final vote.  Falls back to Round 1 result on any API failure.
    Returns a list in the same format as analyst_votes.
    """
    chart_data = chart_data or {}
    user_context = str(chart_data.get("user_context") or "")
    timeframe = str(chart_data.get("timeframe") or "")
    credentials: dict[str, tuple[str, str]] = {}
    for vote in analyst_votes:
        active_model = vote.get("_active_model")
        if isinstance(active_model, dict):
            provider = active_model.get("provider", vote.get("_platform", ""))
            if provider not in credentials:
                credentials[provider] = (
                    _provider_key(provider),
                    _provider_account_id(provider),
                )

    def deliberate(vote: dict) -> dict:
        model_name = vote.get("model", "?")
        model_id = vote.get("_model_id", "")
        platform = vote.get("_platform", "Groq")
        active_model = vote.get("_active_model")
        if not isinstance(active_model, dict):
            return {
                **vote,
                "_deliberated": False,
                "_round1_vote": vote.get("vote", "?"),
            }
        api_key, account_id = credentials.get(
            active_model.get("provider", platform), ("", "")
        )
        prompt = _build_deliberation_prompt(
            model_name, analyst_votes, user_context, timeframe
        )
        r = _call_activated_text_model(
            active_model, prompt, api_key, account_id
        )
        r["_platform"] = platform
        r["_model_id"] = model_id
        r["_deliberated"] = True
        r["_round1_vote"] = vote.get("vote", "?")
        r.setdefault("key_risks", vote.get("key_risks", []))
        r.setdefault("analysis", vote.get("analysis", ""))
        return r

    # Do not make a slow provider hold every other analyst hostage.
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(analyst_votes)))) as pool:
        futures = {pool.submit(deliberate, vote): vote for vote in analyst_votes}
        revised: list[dict] = []
        for future, vote in futures.items():
            try:
                revised.append(future.result())
            except Exception:
                revised.append({
                    **vote,
                    "_deliberated": False,
                    "_round1_vote": vote.get("vote", "?"),
                })
    return revised


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Synthesis & final verdict
# ─────────────────────────────────────────────────────────────────────────────
def _build_synthesis_prompt(chart_data: dict, analyst_votes: list[dict]) -> str:
    tf         = chart_data.get("timeframe", "unknown")
    tf_minutes = int(chart_data.get("timeframe_minutes", 0) or 0)
    asset      = chart_data.get("asset", "the asset")
    user_ctx   = (chart_data.get("user_context") or "").strip()

    # The vision model is also an active analyst. Do not add a second,
    # duplicate vision position here; every active model must count once.
    positions = ""
    for v in analyst_votes:
        positions += (
            v.get("model", "?") + ": " + v.get("vote", "WAIT")
            + " (" + str(v.get("confidence", 0)) + "%) — "
            + v.get("reasoning", "") + "\n"
            "  Analysis  : " + v.get("analysis", "") + "\n"
            "  Key Risks : " + ", ".join(v.get("key_risks", [])) + "\n"
        )

    online_names = [v.get("model", "?") for v in analyst_votes]

    # Build a concrete hold-duration guide based on the real timeframe
    if tf_minutes > 0:
        min_hold = tf_minutes * 2
        max_hold = tf_minutes * 8
        candle_guide = (
            "Each candle = " + str(tf_minutes) + " min on this chart.\n"
            "   Pick candle_count between 2 and 8 based on signal strength:\n"
            "     STRONG consensus → 5-8 candles (~" + str(tf_minutes * 5) + "-" + str(tf_minutes * 8) + " min hold)\n"
            "     MODERATE consensus → 3-5 candles (~" + str(tf_minutes * 3) + "-" + str(tf_minutes * 5) + " min hold)\n"
            "     WEAK / uncertain → 2-3 candles (~" + str(tf_minutes * 2) + "-" + str(tf_minutes * 3) + " min hold)\n"
            "   total_duration_minutes MUST equal candle_count × " + str(tf_minutes) + " exactly.\n"
            '   display_text format: "' + ("BUY" if True else "") + ' ' + asset + ' on ' + tf
            + ' chart | Hold ~[N] candles ([total_duration_minutes] min) | Direction: [UP/DOWN]"\n'
        )
    else:
        candle_guide = (
            "Timeframe in minutes not detected — estimate hold based on typical "
            + tf + " chart trading: 2-6 candles, realistic duration.\n"
            '   display_text: "Trade ' + asset + ' — [direction] | Hold [N] candles on ' + tf + ' chart"\n'
        )

    user_section = (
        "=== TRADER'S NOTES — AUTHORITATIVE (follow these exactly) ===\n"
        + user_ctx + "\n"
        "The trader's explicit timeframe and question are facts. "
        "Do not replace them with a timeframe inferred from the image. "
        "Answer the trader's direct question explicitly in moderator_note and "
        "recommended_action.display_text.\n"
        "=============================================================\n\n"
    ) if user_ctx else ""

    return (
        "You are the debate moderator for an AI trading analyst panel.\n\n"
        + user_section
        + "Chart: " + asset + " | Timeframe: " + tf + " | Models online ("
        + str(len(online_names)) + "): " + ", ".join(online_names) + "\n\n"
        "=== ALL POST-DELIBERATION ANALYST POSITIONS ===\n"
        + positions
        + "================================================\n\n"
        "Your tasks:\n"
        "0. CONTEXT CHECK — read every quoted note and direct question. Explicitly "
        "answer the trader's question, use the supplied timeframe, and include both "
        "in moderator_note and recommended_action.display_text.\n"
        "1. CROSS-EXAMINE — summarise key agreements and disagreements in 3-4 sentences.\n"
        "2. FINAL_DECISION — choose UP, DOWN, or WAIT based on weight of evidence.\n"
        "3. RECOMMENDED_ACTION — give a concrete, timeframe-specific trade instruction:\n"
        "   Timeframe: " + tf + "\n"
        "   " + candle_guide
        + "   If WAIT: should_trade=false, explain the specific condition preventing a trade.\n\n"
        "Return ONLY valid JSON — no markdown:\n"
        "{\n"
        '  "cross_examination": "<3-4 sentence moderator analysis>",\n'
        '  "vote_tally": {"UP": 0, "DOWN": 0, "WAIT": 0},\n'
        '  "consensus_strength": "<STRONG|MODERATE|DIVIDED>",\n'
        '  "FINAL_DECISION": "<UP|DOWN|WAIT>",\n'
        '  "confidence": <0-100>,\n'
        '  "moderator_note": "<2-3 sentence actionable summary for the trader>",\n'
        '  "recommended_action": {\n'
        '    "should_trade": <true|false>,\n'
        '    "trade_direction": "<UP|DOWN|null>",\n'
        '    "candle_count": <integer 2-8 or null>,\n'
        '    "total_duration_minutes": <candle_count × ' + str(max(tf_minutes, 1)) + ' or null>,\n'
        '    "dont_trade_reason": "<string or null>",\n'
        '    "display_text": "<concrete trade instruction string>"\n'
        "  }\n"
        "}"
    )


def step3_synthesize(chart_data: dict, analyst_votes: list[dict]) -> dict:
    """
    Cross-examines all analyst opinions → FINAL_DECISION + candle recommendation.
    Uses the top available Groq text model.
    """
    prompt  = _build_synthesis_prompt(chart_data, analyst_votes)
    models  = _discover_groq_text_models(n=2)
    last_err = ""
    for model_id, label in models:
        try:
            client = _groq_client()
            chat = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system",
                     "content": "You are a trading debate moderator. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            result = parse_json(chat.choices[0].message.content)
            result["_synth_model"] = label + " (Groq)"
            return _normalise_synthesis(result, chart_data, analyst_votes)
        except Exception as e:
            last_err = str(e)
            continue
    # A provider outage must not erase the committee's usable answer.
    return _normalise_synthesis({}, chart_data, analyst_votes, last_err)


def _normalise_synthesis(
    result: dict,
    chart_data: dict,
    analyst_votes: list[dict],
    error: str = "",
) -> dict:
    """Make the displayed verdict agree with the actual active-model tally."""
    tally = {"UP": 0, "DOWN": 0, "WAIT": 0}
    for vote in analyst_votes:
        value = _normalise_vote_fields(dict(vote), "Vote recorded.")["vote"]
        tally[value] = tally.get(value, 0) + 1

    ordered = sorted(tally.items(), key=lambda item: item[1], reverse=True)
    leader, leader_count = ordered[0] if ordered else ("WAIT", 0)
    second_count = ordered[1][1] if len(ordered) > 1 else 0
    total = sum(tally.values())
    # A strict plurality is enough for a directional committee decision;
    # a tie remains WAIT rather than inventing conviction.
    decision = leader if leader_count > second_count and leader_count else "WAIT"
    confidence = int(round(100 * leader_count / total)) if total else 0
    existing_tally = result.get("vote_tally")
    if not isinstance(existing_tally, dict):
        existing_tally = {}

    result["vote_tally"] = {
        key: int(existing_tally.get(key, tally[key]) or tally[key])
        for key in ("UP", "DOWN", "WAIT")
    }
    # The source-of-truth tally must win over an LLM's malformed or stale count.
    result["vote_tally"] = tally
    result["FINAL_DECISION"] = decision
    result["confidence"] = max(0, min(100, int(result.get("confidence", confidence) or confidence)))
    if not result["confidence"]:
        result["confidence"] = confidence
    result["consensus_strength"] = result.get(
        "consensus_strength"
    ) or (
        "STRONG" if total and leader_count / total >= 0.7
        else ("MODERATE" if decision != "WAIT" else "DIVIDED")
    )
    result["cross_examination"] = result.get(
        "cross_examination"
    ) or (
        f"The committee recorded {tally['UP']} UP, {tally['DOWN']} DOWN, "
        f"and {tally['WAIT']} WAIT positions. "
        f"{decision} leads by the current weight of evidence."
    )
    result["moderator_note"] = result.get(
        "moderator_note"
    ) or (
        f"Final committee signal: {decision}. Review the chart levels, "
        "risk controls, and current market conditions before acting."
        if decision != "WAIT"
        else "The committee is divided or lacks a clear edge. Wait for confirmation "
        "from price action before entering a position."
    )
    user_context = str(chart_data.get("user_context") or "").strip()
    timeframe = str(chart_data.get("timeframe") or "").strip()
    if user_context and timeframe:
        acknowledgement = (
            f"Using the trader-provided {timeframe} timeframe."
        )
        if timeframe.casefold() not in result["moderator_note"].casefold():
            result["moderator_note"] = (
                acknowledgement + " " + result["moderator_note"]
            )
    action = result.get("recommended_action")
    if not isinstance(action, dict):
        action = {}
    action.setdefault("should_trade", decision in {"UP", "DOWN"})
    action.setdefault("trade_direction", decision if decision in {"UP", "DOWN"} else None)
    action.setdefault("candle_count", 3 if decision in {"UP", "DOWN"} else None)
    minutes = int(chart_data.get("timeframe_minutes", 0) or 0)
    action.setdefault(
        "total_duration_minutes",
        (3 * minutes) if decision in {"UP", "DOWN"} and minutes else None,
    )
    action.setdefault(
        "dont_trade_reason",
        None if decision in {"UP", "DOWN"} else "No clear majority signal.",
    )
    action.setdefault(
        "display_text",
        (
            f"{decision} signal — confirm entry with risk controls."
            if decision in {"UP", "DOWN"}
            else "WAIT — no clear majority signal.",
        ),
    )
    if user_context and timeframe:
        display_text = str(action.get("display_text") or "")
        if timeframe.casefold() not in display_text.casefold():
            action["display_text"] = f"{timeframe} chart — {display_text}"
    result["recommended_action"] = action
    if error:
        result.setdefault("_synth_error", error[:240])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# UI constants & CSS
# ─────────────────────────────────────────────────────────────────────────────
VOTE_COLOR = {"UP": "green", "DOWN": "red", "WAIT": "orange"}
VOTE_ICON  = {"UP": "⬆️",   "DOWN": "⬇️",  "WAIT": "⏸️"}

_NEON = {
    "UP":      "#00ff88",
    "DOWN":    "#ff3860",
    "WAIT":    "#ffaa00",
    "Offline": "#6a7f9a",
    "OFFLINE": "#555555",
    "blue":    "#00b4ff",
}

_PREMIUM_CSS = """
<style>
/* ── Google Font ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

/* ── Animated mesh background ───────────────────────────────────────── */
@keyframes meshMove {
    0%   { transform: translateY(0px)   rotate(0deg);   opacity:.025; }
    50%  { transform: translateY(-28px) rotate(6deg);   opacity:.055; }
    100% { transform: translateY(0px)   rotate(0deg);   opacity:.025; }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0);    }
}
@keyframes btnPulse {
    0%,100% { box-shadow: 0 0 0   0   rgba(0,180,255,.55); }
    50%     { box-shadow: 0 0 0  14px rgba(0,180,255,0);   }
}
@keyframes voteGlow {
    0%,100% { filter: brightness(1)   drop-shadow(0 0 4px  currentColor); }
    50%     { filter: brightness(1.3) drop-shadow(0 0 14px currentColor); }
}
@keyframes scanLine {
    0%   { top: -2px; }
    100% { top: 100%; }
}
@keyframes ambientDrift {
    0%,100% { transform: translate3d(0,0,0) scale(1); opacity:.18; }
    50%     { transform: translate3d(2vw,-1.5vh,0) scale(1.04); opacity:.28; }
}
@keyframes headerReveal {
    0%   { opacity:.35; letter-spacing:.02em; }
    50%  { opacity:1; letter-spacing:.035em; }
    100% { opacity:.92; letter-spacing:.02em; }
}

/* ── Base dark theme ────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #05080f !important;
    color: #c8d6e8;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"]::before {
    content:'';
    position:fixed; inset:-12%;
    pointer-events:none; z-index:0;
    background:
        radial-gradient(circle at 18% 16%, #00b4ff18 0 8%, transparent 28%),
        radial-gradient(circle at 84% 72%, #00ff8812 0 7%, transparent 26%),
        linear-gradient(115deg, transparent 0 48%, #1a356012 49%, transparent 50%);
    background-size: 100% 100%, 100% 100%, 34px 34px;
    animation: ambientDrift 14s ease-in-out infinite;
}
[data-testid="stMain"] { position:relative; z-index:1; }
[data-testid="stHeader"],
[data-testid="stToolbar"]    { background: transparent !important; }
[data-testid="stSidebar"]    {
    background: linear-gradient(180deg, #080c1a 0%, #060910 100%) !important;
    border-right: 1px solid #1a2340;
}
[data-testid="stMain"] > div { padding-top: 1rem; }
hr { border-color: #1a2340 !important; }

/* ── Metrics ────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#0d1220,#0a0f1c);
    border: 1px solid #1a2340; border-radius: 10px; padding: 10px 14px;
    transition: border-color .25s;
}
[data-testid="metric-container"]:hover { border-color: #00b4ff44; }

/* ── File uploader ──────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0a0f1c; border: 2px dashed #1a3560;
    border-radius: 14px; padding: 8px; transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover { border-color: #00b4ff; }

/* ── Text inputs & areas ────────────────────────────────────────────── */
textarea, input[type="text"] {
    background: #0a0f1c !important; color: #c8d6e8 !important;
    border: 1px solid #1a2340 !important; border-radius: 10px !important;
    transition: border-color .2s !important;
}
textarea:focus, input[type="text"]:focus { border-color: #00b4ff !important; }

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0a0f1c !important; border: 1px solid #1a2340 !important;
    border-radius: 12px !important;
}

/* ── Primary button ─────────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0044dd, #00b4ff) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 800 !important;
    letter-spacing: .05em !important;
    animation: btnPulse 2.4s ease-in-out infinite;
    transition: transform .15s, box-shadow .15s !important;
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #0060ff, #33ccff) !important;
    box-shadow: 0 0 28px #00b4ffaa !important;
    transform: translateY(-2px) !important;
    animation: none;
}

/* ── Secondary buttons ──────────────────────────────────────────────── */
[data-testid="baseButton-secondary"] {
    background: #101828 !important; color: #a0b8d0 !important;
    border: 1px solid #1e3050 !important; border-radius: 8px !important;
    transition: border-color .2s, color .2s !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #00b4ff !important; color: #00b4ff !important;
}

/* ── AI analyst cards ───────────────────────────────────────────────── */
.ai-card {
    background: linear-gradient(135deg, #0d1220 0%, #0a101e 100%);
    border: 1px solid #1a2340;
    border-radius: 16px; padding: 18px 20px 14px; margin: 10px 0;
    animation: fadeUp .55s ease both;
    box-shadow: 0 4px 24px rgba(0,0,0,.5);
    /* 3D perspective */
    perspective: 900px;
    transition: transform .35s cubic-bezier(.25,.46,.45,.94),
                box-shadow .35s ease,
                border-color .3s ease;
    position: relative; overflow: hidden;
}
/* Subtle scan-line shimmer on hover */
.ai-card::after {
    content:''; position:absolute; left:0; top:-2px;
    width:100%; height:2px;
    background: linear-gradient(90deg,transparent,#00b4ff55,transparent);
    opacity:0; transition:opacity .3s;
}
.ai-card:hover {
    transform: translateY(-6px) rotateX(2.5deg) rotateY(-1deg);
    box-shadow: 0 18px 50px rgba(0,180,255,.18),
                0 6px 16px rgba(0,0,0,.6);
    border-color: #00b4ff44 !important;
}
.ai-card:hover::after { opacity:1; animation: scanLine 1.4s linear infinite; }
.ai-card.offline { border-color: #ff386028; opacity:.72; }

.card-row { display:flex; align-items:center; gap:10px;
            flex-wrap:wrap; margin-bottom:10px; }
.model-lbl  { font-size:.95rem; font-weight:700; color:#e0eaf8; }
.vote-pill  { font-size:.8rem; font-weight:800; padding:3px 11px;
              border-radius:20px; border:1.5px solid; letter-spacing:.05em; }
.conf-lbl   { font-size:.78rem; color:#6a82a0; margin-left:auto; }
.analysis-txt { font-size:.88rem; color:#a8bcd4; line-height:1.6; margin:6px 0; }
.risks-row  { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
.risk-tag   { font-size:.74rem; padding:3px 9px; border-radius:6px;
              background:#0e1526; border:1px solid #1e2f50; color:#7a9abf; }
.reasoning-txt { font-size:.83rem; color:#5e7a9a; font-style:italic; margin-top:6px; }
.err-txt    { font-size:.82rem; color:#ff6080; margin-top:6px; }

/* ── Scoreboard tiles ───────────────────────────────────────────────── */
.sb-tile {
    background: linear-gradient(160deg,#0d1220,#0a101e);
    border: 1.5px solid #1a2340; border-radius: 12px;
    padding: 10px 6px; text-align: center;
    animation: fadeUp .5s ease both;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    cursor: default;
}
.sb-tile:hover {
    transform: translateY(-4px) scale(1.04);
    box-shadow: 0 10px 30px rgba(0,180,255,.15);
    border-color: #00b4ff55;
}
.sb-model-lbl { font-size:.65rem; color:#566880; margin-bottom:4px; }
.sb-vote-icon { font-size:1.4rem; }
.sb-vote-txt  { font-size:.88rem; font-weight:800;
                animation: voteGlow 3s ease-in-out infinite; }
.sb-conf-txt  { font-size:.65rem; color:#566880; margin-top:2px; }

/* ── Final verdict banner ───────────────────────────────────────────── */
.verdict-banner {
    text-align:center; padding:32px 0 14px;
    animation: fadeUp .6s ease both;
}
.verdict-word {
    font-size:5rem; font-weight:900; line-height:1;
    animation: voteGlow 2.5s ease-in-out infinite;
}
.verdict-sub  { font-size:1rem; color:#6a82a0; margin-top:8px; }

/* ── Trade boxes ────────────────────────────────────────────────────── */
.trade-box {
    border-radius: 18px; padding: 34px 28px; margin:10px 0;
    text-align:center; animation: fadeUp .6s ease both;
    position: relative; overflow:hidden;
}
.trade-box::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 50% 0%, rgba(255,255,255,.04), transparent 70%);
    pointer-events:none;
}
.trade-arrow { font-size:2.8rem; }
.trade-txt   { font-size:1.5rem; font-weight:900;
               letter-spacing:.03em; line-height:1.55; margin-top:10px; }

/* ── Step headers ───────────────────────────────────────────────────── */
.step-header {
    display:flex; align-items:center; gap:10px;
    font-size:1.2rem; font-weight:700; color:#c8d6e8;
    border-left: 3px solid;
    border-image: linear-gradient(180deg,#00b4ff,#00ff88) 1;
    padding-left:12px; margin:24px 0 10px;
}

/* ── Platform badges ─────────────────────────────────────────────────── */
.platform-badge {
    font-size:.65rem; font-weight:800; padding:2px 7px;
    border-radius:5px; letter-spacing:.06em; vertical-align:middle;
    display:inline-block;
}
.platform-groq    { background:#0f2008; color:#6ee73a; border:1px solid #3a8810; }
.platform-deepseek{ background:#080f28; color:#60a5fa; border:1px solid #2e5fd4; }
.platform-gemini  { background:#130828; color:#c084fc; border:1px solid #7c22d0; }
.platform-custom  { background:#101018; color:#94a3b8; border:1px solid #3a4060; }

/* ── Sidebar model tags ─────────────────────────────────────────────── */
.custom-model-tag {
    display:inline-flex; align-items:center; gap:6px;
    background:#0d1424; border:1px solid #1e3050; border-radius:8px;
    padding:4px 10px; margin:3px 0; font-size:.8rem; color:#7a9abf;
    width:100%; transition: border-color .2s;
}
.custom-model-tag:hover { border-color: #00b4ff44; }

/* ── Catalog model row ──────────────────────────────────────────────── */
.catalog-row {
    display:flex; align-items:center; justify-content:space-between;
    background:#0a0f1c; border:1px solid #1a2340; border-radius:8px;
    padding:5px 10px; margin:3px 0; font-size:.78rem; color:#7a9abf;
}
</style>
"""


def _model_icon(name: str) -> str:
    n = name.lower()
    if "gemini"        in n: return "✨"
    if "deepseek"      in n: return "🔭"
    if "llama-4"       in n: return "🦙"
    if "llama-4-scout" in n: return "🔬"
    if "mixtral"       in n: return "⚗️"
    if "gemma"         in n: return "💎"
    if "qwen"          in n: return "🧠"
    if "vision"        in n: return "👁️"
    if "llama"         in n: return "🦙"
    return "🤖"


def render_chart_analysis(g: dict, status_msg: str) -> None:
    vision = g.get("vision_model", "Vision Model")

    is_warn = any(w in status_msg.lower()
                  for w in ("fallback", "unavailable", "failed", "offline"))
    if is_warn:
        st.warning(status_msg)
    else:
        st.success(status_msg)

    # ── Vision card ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='step-header' style='font-size:1rem;margin:12px 0 8px;'>"
        "👁️ Vision Analysis</div>",
        unsafe_allow_html=True,
    )
    grv      = g.get("gemini_vision_vote", g.get("groq_vision_vote", "Offline"))
    grc      = g.get("gemini_vision_confidence", g.get("groq_vision_confidence", 0))
    grm      = g.get("vision_model", g.get("groq_vision_model", "Vision Model"))
    provider = g.get("vision_provider", "Vision AI")
    gr_color = _NEON.get(grv, "#555")
    _, vcenter, _ = st.columns([1, 2, 1])
    with vcenter:
        st.markdown(
            "<div class='ai-card' style='border-color:" + gr_color + "44;text-align:center;'>"
            "<div style='margin-bottom:6px;'>" + _platform_badge(provider) + "</div>"
            "<div style='font-size:.85rem;color:#8090a8;margin-bottom:4px;'>" + grm.split("/")[-1] + "</div>"
            "<div style='font-size:2rem;font-weight:900;color:" + gr_color + ";'>"
            + VOTE_ICON.get(grv, "❓") + " " + grv + "</div>"
            "<div style='font-size:.78rem;color:#6a82a0;margin-top:4px;'>" + str(grc) + "% confidence</div>"
            "<div style='font-size:.8rem;color:#a0b4c8;margin-top:8px;font-style:italic;'>"
            + g.get("gemini_vision_reasoning",
                    g.get("groq_vision_reasoning", "")) + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Live news banner ──────────────────────────────────────────────────────
    live_news  = g.get("live_news", [])
    real_news  = g.get("news_searched", False)
    if live_news:
        news_label = (
            "🌐 Live Market News — " + g.get("asset", "Asset")
            + " (real-time web search)"
            if real_news else
            "📰 Market News — " + g.get("asset", "Asset") + " (AI knowledge)"
        )
        st.markdown(
            "<div class='step-header' style='font-size:.95rem;margin:16px 0 8px;'>"
            + news_label + "</div>",
            unsafe_allow_html=True,
        )
        news_html = ""
        for item in live_news[:6]:
            headline = item.get("headline", "")
            source   = item.get("source", "")
            date     = item.get("date", "")
            news_html += (
                "<div style='padding:7px 0;border-bottom:1px solid #0d1a2e;'>"
                "<div style='font-size:.82rem;color:#c0d4e8;font-weight:600;line-height:1.4;'>"
                + headline + "</div>"
                "<div style='font-size:.7rem;color:#566880;margin-top:2px;'>"
                + source + ("  ·  " + date if date else "") + "</div>"
                "</div>"
            )
        st.markdown(
            "<div class='ai-card' style='padding:12px 16px;'>" + news_html + "</div>",
            unsafe_allow_html=True,
        )
    elif real_news is False:
        st.markdown(
            "<p style='font-size:.78rem;color:#2a3a50;font-style:italic;'>"
            "⚠️ News search unavailable — analysts will use chart data only.</p>",
            unsafe_allow_html=True,
        )

    # ── Technical analysis detail ─────────────────────────────────────────────
    with st.expander("📊 " + vision + " — Full Technical Analysis + Market News", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Asset",     g.get("asset",     "—"))
        c2.metric("Timeframe", g.get("timeframe", "—"))
        c3.metric("Trend",     g.get("trend",     "—"))

        st.info("**Technical Summary:** " + g.get("technical_summary", ""))

        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Support**")
            for s in g.get("support", []):
                st.markdown(f"- `{s}`")
        with cb:
            st.markdown("**Resistance**")
            for r in g.get("resistance", []):
                st.markdown(f"- `{r}`")

        inds = g.get("indicators", {})
        if inds:
            st.markdown("**Indicators**")
            for k, val in inds.items():
                st.markdown(f"- **{k}**: {val}")

        if g.get("patterns"):
            st.markdown("**Patterns:** " + " · ".join(f"`{p}`" for p in g["patterns"]))

        st.markdown("---")
        real_news = g.get("news_searched", False)
        st.markdown(
            "**📰 " + ("Live Web News (DuckDuckGo)" if real_news else "Market News (AI knowledge)") + "**"
        )
        for item in g.get("live_news", []):
            headline = item.get("headline", "")
            source   = item.get("source", "")
            date     = item.get("date", "")
            body     = item.get("body", "")
            sent     = item.get("sentiment", "Neutral")
            dot      = "🟢" if sent == "Bullish" else ("🔴" if sent == "Bearish" else "🟡")
            date_str = f" · {date}" if date else ""
            st.markdown(dot + " **" + headline + "**  \n"
                        + f"*{source}{date_str}*"
                        + (f"  \n{body}" if body else ""))
        if not real_news:
            st.markdown("**News Summary:** " + g.get("news_summary", ""))


def _platform_badge(platform: str) -> str:
    p = platform.lower()
    if p == "groq":
        return "<span class='platform-badge platform-groq'>⚡ GROQ</span>"
    if p == "deepseek":
        return "<span class='platform-badge platform-deepseek'>🔭 DEEPSEEK</span>"
    if p == "gemini":
        return "<span class='platform-badge platform-gemini'>✨ GEMINI</span>"
    return "<span class='platform-badge platform-custom'>🤖 " + platform.upper() + "</span>"


def render_vote_card(v: dict, status: str = "online") -> None:
    model    = v.get("model",      "Unknown")
    vote     = v.get("vote",       "WAIT")
    icon     = _model_icon(model)
    conf     = v.get("confidence", 0)
    analysis = v.get("analysis",   "")
    risks    = v.get("key_risks",  [])
    reason   = v.get("reasoning",  "")
    platform = v.get("_platform",  "")
    badge    = _platform_badge(platform) if platform else ""

    if status == "offline":
        err = _safe_error(v.get("error_msg", "This model did not respond."))
        plat = v.get("_platform", "")
        b    = _platform_badge(plat) if plat else ""
        st.markdown(
            "<div class='ai-card offline'>"
            "<div class='card-row'>"
            "<span class='model-lbl'>🔴 " + icon + " " + model + "</span>"
            + b +
            "<span style='font-size:.8rem;color:#ff6080;margin-left:4px;'>— OFFLINE / SKIPPED</span>"
            "</div>"
            "<p class='err-txt'>" + err + "</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    color     = _NEON.get(vote, "#888")
    vote_icon = VOTE_ICON.get(vote, "❓")
    risks_html = "".join("<span class='risk-tag'>" + r + "</span>" for r in risks)

    st.markdown(
        "<div class='ai-card'>"
        "<div class='card-row'>"
        "<span class='model-lbl'>" + icon + " " + model + "</span>"
        + badge +
        "<span class='vote-pill' style='color:" + color + ";border-color:" + color + ";'>"
        + vote_icon + " " + vote + "</span>"
        "<span class='conf-lbl'>" + str(conf) + "% confidence</span>"
        "</div>"
        "<p class='analysis-txt'>" + analysis + "</p>"
        "<div class='risks-row'>" + risks_html + "</div>"
        "<p class='reasoning-txt'>💬 " + reason + "</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_scoreboard(chart_data: dict, analyst_votes: list[dict],
                      offline: list[str]) -> None:
    st.markdown("<div class='step-header'>🗳️ All AI Votes at a Glance</div>",
                unsafe_allow_html=True)

    # Each active model appears exactly once; the vision model is included
    # through analyst_votes after its primary chart pass.
    entries = [{
        "model":    v.get("model", "?"),
        "vote":     v.get("vote",  "WAIT"),
        "conf":     v.get("confidence", 0),
        "icon":     _model_icon(v.get("model", "")),
        "platform": v.get("_platform", ""),
    } for v in analyst_votes]

    cols = st.columns(max(len(entries), 1))
    for col, entry in zip(cols, entries):
        vote     = entry["vote"]
        color    = _NEON.get(vote, "#555")
        icon     = VOTE_ICON.get(vote, "❌") if vote not in ("OFFLINE", "Offline") else "❌"
        conf_txt = "offline" if vote in ("OFFLINE", "Offline") else str(entry["conf"]) + "% conf"
        plat_html = _platform_badge(entry["platform"]) if entry["platform"] else ""
        col.markdown(
            "<div class='sb-tile' style='border-color:" + color + "22;'>"
            "<div class='sb-model-lbl'>" + entry["icon"] + " " + entry["model"] + "</div>"
            "<div style='margin:4px 0;'>" + plat_html + "</div>"
            "<div class='sb-vote-icon'>" + icon + "</div>"
            "<div class='sb-vote-txt' style='color:" + color + ";'>" + vote + "</div>"
            "<div class='sb-conf-txt'>" + conf_txt + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_final_decision(synth: dict) -> None:
    decision = synth.get("FINAL_DECISION", "WAIT")
    color    = _NEON.get(decision, "#888")
    arrow    = VOTE_ICON.get(decision, "❓")
    strength = synth.get("consensus_strength", "")
    tally    = synth.get("vote_tally", {})
    conf     = synth.get("confidence", 0)
    s_model  = synth.get("_synth_model", "Gemini")

    st.markdown("---")
    st.markdown(
        "<div class='step-header'>🏆 Final Consensus"
        "<span style='font-size:.8rem;font-weight:400;color:#566880;margin-left:8px;'>"
        "moderated by " + s_model + "</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='verdict-banner'>"
        "<div class='verdict-word' style='color:" + color + ";'>" + arrow + " " + decision + "</div>"
        "<div class='verdict-sub'>"
        "Consensus: <strong style='color:#c8d6e8;'>" + strength + "</strong>"
        " &nbsp;·&nbsp; Confidence: <strong style='color:" + color + ";'>"
        + str(conf) + "%</strong>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("⬆️ UP",   tally.get("UP",   0))
    c2.metric("⬇️ DOWN", tally.get("DOWN", 0))
    c3.metric("⏸️ WAIT", tally.get("WAIT", 0))

    st.markdown("**Cross-Examination:** " + synth.get("cross_examination", ""))
    st.info("📋 **Moderator Note:** " + synth.get("moderator_note", ""))


def render_deliberation_round(round1: list[dict], round2: list[dict]) -> None:
    """
    Deliberation panel — shows what each AI decided after reading all peers'
    reasoning, including who changed their mind and why.
    """
    from collections import Counter

    st.markdown("---")
    st.markdown(
        "<div class='step-header'>🔄 Deliberation Round — AIs Share Logic & Debate</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:.8rem;color:#566880;margin-bottom:12px;'>"
        "Each AI read all peers' full reasoning and submitted a final vote. "
        "These revised positions are what the moderator uses for the verdict.</p>",
        unsafe_allow_html=True,
    )

    # ── Tally shift summary ───────────────────────────────────────────────────
    r1_tally = Counter(v.get("vote", "?") for v in round1)
    r2_tally = Counter(v.get("vote", "?") for v in round2)
    changed  = sum(1 for r in round2
                   if r.get("_deliberated") and r.get("_round1_vote","?") != r.get("vote","?"))

    def _tally_html(tally: Counter) -> str:
        return "  ·  ".join(
            "<strong style='color:" + _NEON.get(v, "#aaa") + ";'>" + v + ": " + str(c) + "</strong>"
            for v, c in tally.most_common()
        )

    shift_note = (
        "<div style='font-size:.75rem;color:#ffaa00;margin-top:8px;'>🔄 "
        + str(changed) + " analyst(s) changed position after deliberation.</div>"
        if changed else
        "<div style='font-size:.75rem;color:#00ff88;margin-top:8px;'>"
        "✅ All analysts maintained their Round 1 positions.</div>"
    )
    st.markdown(
        "<div class='ai-card' style='padding:14px 18px;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:.88rem;'>"
        "<tr><td style='color:#566880;font-size:.72rem;width:100px;padding-bottom:6px;'>"
        "Round 1</td><td>" + _tally_html(r1_tally) + "</td></tr>"
        "<tr><td style='color:#566880;font-size:.72rem;padding-bottom:4px;'>"
        "After Debate</td><td>" + _tally_html(r2_tally) + "</td></tr>"
        "</table>" + shift_note + "</div>",
        unsafe_allow_html=True,
    )

    # ── Individual deliberation cards ────────────────────────────────────────
    n        = len(round2)
    ncols    = min(n, 3)
    cols     = st.columns(ncols) if n > 1 else [st.container()]

    for idx, r2 in enumerate(round2):
        model_name  = r2.get("model", "?")
        platform    = r2.get("_platform", "Groq")
        r1_vote     = r2.get("_round1_vote", r2.get("vote", "?"))
        r2_vote     = r2.get("vote", "?")
        deliberated = r2.get("_deliberated", False)
        flipped     = deliberated and r1_vote != r2_vote
        conf        = r2.get("confidence", 0)
        reasoning   = r2.get("reasoning", "")
        color       = _NEON.get(r2_vote, "#6a82a0")

        badge_html = (
            "<span style='font-size:.65rem;color:#ffaa00;font-weight:800;'>🔄 REVISED</span>"
            if flipped else
            "<span style='font-size:.65rem;color:#00ff88;font-weight:800;'>✅ MAINTAINED</span>"
        ) if deliberated else (
            "<span style='font-size:.62rem;color:#566880;'>⚠️ Round 1 kept</span>"
        )

        with cols[idx % ncols]:
            st.markdown(
                "<div class='ai-card' style='border-color:" + color + "33;text-align:center;'>"
                "<div style='display:flex;justify-content:space-between;align-items:center;"
                "margin-bottom:8px;'>"
                + _platform_badge(platform) + badge_html + "</div>"
                "<div style='font-size:.78rem;color:#8090a8;margin-bottom:4px;'>"
                + model_name + "</div>"
                + ("<div style='font-size:.72rem;color:#566880;'>"
                   "was: <span style='text-decoration:line-through;color:#ff6080;'>"
                   + VOTE_ICON.get(r1_vote, "") + " " + r1_vote + "</span></div>"
                   if flipped else "")
                + "<div style='font-size:1.8rem;font-weight:900;color:" + color + ";margin:6px 0;'>"
                + VOTE_ICON.get(r2_vote, "❓") + " " + r2_vote + "</div>"
                "<div style='font-size:.72rem;color:#6a82a0;'>" + str(conf) + "% confidence</div>"
                "<div style='font-size:.78rem;color:#5e7a9a;font-style:italic;"
                "margin-top:8px;line-height:1.5;text-align:left;'>"
                + reasoning + "</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def render_trade_recommendation(synth: dict) -> None:
    ra = synth.get("recommended_action", {})
    if not ra:
        return

    should_trade = ra.get("should_trade", False)
    display_text = str(ra.get("display_text") or "")
    direction    = str(ra.get("trade_direction") or "").upper()
    candles      = ra.get("candle_count")
    total_mins   = ra.get("total_duration_minutes")
    no_trade_rsn = ra.get("dont_trade_reason") or ""

    st.markdown("---")
    st.markdown("<div class='step-header'>🎯 Recommended Action</div>",
                unsafe_allow_html=True)

    if should_trade and direction in ("UP", "DOWN"):
        bg     = str("#041a0a" if direction == "UP" else "#1a0408")
        txt    = str(_NEON["UP"]  if direction == "UP" else _NEON["DOWN"])
        border = str("#00c853"    if direction == "UP" else "#d50000")
        arrow  = str("⬆️"        if direction == "UP" else "⬇️")
        st.markdown(
            "<div class='trade-box' style='background:" + bg + ";border:2.5px solid " + border + ";'>"
            "<div class='trade-arrow'>" + arrow + "</div>"
            "<div class='trade-txt' style='color:" + txt + ";'>" + display_text + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if candles and total_mins:
            m1, m2, m3 = st.columns(3)
            m1.metric("Direction",       arrow + " " + direction)
            m2.metric("Candles to Hold", str(candles))
            m3.metric("Est. Duration",   str(total_mins) + " min")
    else:
        st.markdown(
            "<div class='trade-box' style='background:#1a1200;border:2.5px solid #ff8f00;'>"
            "<div class='trade-arrow'>🚫</div>"
            "<div class='trade-txt' style='color:#ffd740;'>" + display_text + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if no_trade_rsn and no_trade_rsn.lower() not in ("null", "none", ""):
            st.warning("⚠️ " + no_trade_rsn)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE CANDLESTICK ENGINE — Exness / Quotex market-data mode
# ─────────────────────────────────────────────────────────────────────────────
_LIVE_MARKETS = {
    "EUR/USD": {"yahoo": "EURUSD=X", "mt5": "EURUSD", "ccxt": None},
    "GBP/USD": {"yahoo": "GBPUSD=X", "mt5": "GBPUSD", "ccxt": None},
    "USD/JPY": {"yahoo": "JPY=X", "mt5": "USDJPY", "ccxt": None},
    "AUD/USD": {"yahoo": "AUDUSD=X", "mt5": "AUDUSD", "ccxt": None},
    "USD/CAD": {"yahoo": "CAD=X", "mt5": "USDCAD", "ccxt": None},
    "Gold (XAU/USD)": {"yahoo": "GC=F", "mt5": "XAUUSD", "ccxt": None},
    "BTC/USDT": {"yahoo": "BTC-USD", "mt5": "BTCUSD", "ccxt": "BTC/USDT"},
}

_LIVE_TIMEFRAMES = {
    "1m": {"yahoo": "1m", "minutes": 1, "period": "5d", "ccxt": "1m"},
    "5m": {"yahoo": "5m", "minutes": 5, "period": "60d", "ccxt": "5m"},
    "15m": {"yahoo": "15m", "minutes": 15, "period": "60d", "ccxt": "15m"},
    "1h": {"yahoo": "1h", "minutes": 60, "period": "730d", "ccxt": "1h"},
}


def _live_interval_config(timeframe: str) -> dict:
    return _LIVE_TIMEFRAMES.get(timeframe, _LIVE_TIMEFRAMES["5m"])


def _normalise_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance/ccxt output into a consistent OHLCV dataframe."""
    if frame is None or frame.empty:
        raise RuntimeError("The market connector returned no candle data.")
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(column[0]) for column in frame.columns]
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    aliases = {
        "adj close": "close",
        "datetime": "timestamp",
        "date": "timestamp",
    }
    frame = frame.rename(columns=aliases)
    if "timestamp" in frame.columns:
        frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True)
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[["open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"]
    )
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if frame.empty:
        raise RuntimeError("The market connector returned invalid OHLC values.")
    return frame.tail(240)


def _fetch_mt5_market_data(
    symbol: str, timeframe: str, login: str, password: str, server: str
) -> tuple[pd.DataFrame | None, str]:
    """Try the optional MT5 bridge without making Streamlit Cloud depend on it."""
    if not login.strip() or not password.strip() or not server.strip():
        return None, ""
    try:
        import MetaTrader5 as mt5
    except Exception:
        return None, (
            "MT5 bridge is not available in this Streamlit Cloud runtime; "
            "public market fallback was used."
        )
    timeframe_map = {
        "1m": getattr(mt5, "TIMEFRAME_M1", None),
        "5m": getattr(mt5, "TIMEFRAME_M5", None),
        "15m": getattr(mt5, "TIMEFRAME_M15", None),
        "1h": getattr(mt5, "TIMEFRAME_H1", None),
    }
    try:
        login_id = int(login.strip())
    except ValueError:
        return None, "MT5 Account Login ID must be numeric; public market fallback was used."
    try:
        if not mt5.initialize(
            login=login_id, password=password, server=server.strip()
        ):
            return None, "MT5 connection failed; public market fallback was used."
        rates = mt5.copy_rates_from_pos(
            symbol, timeframe_map.get(timeframe), 0, 240
        )
        if rates is None or len(rates) == 0:
            return None, "MT5 returned no candles; public market fallback was used."
        frame = pd.DataFrame(rates).rename(columns={"time": "timestamp"})
        return _normalise_market_frame(frame), "Exness MT5"
    except Exception as exc:
        return None, "MT5 request failed; public market fallback was used."
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


def _fetch_public_market_data(pair: str, timeframe: str) -> tuple[pd.DataFrame, str]:
    """Fetch public candles through yfinance first, then ccxt for supported pairs."""
    market = _LIVE_MARKETS[pair]
    interval = _live_interval_config(timeframe)
    errors: list[str] = []
    try:
        frame = yf.download(
            market["yahoo"],
            period=interval["period"],
            interval=interval["yahoo"],
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return _normalise_market_frame(frame), "Yahoo Finance public feed"
    except Exception as exc:
        errors.append("yfinance: " + _safe_error(exc))

    if ccxt is not None and market.get("ccxt"):
        try:
            exchange = ccxt.binance({"enableRateLimit": True})
            rows = exchange.fetch_ohlcv(
                market["ccxt"], timeframe=interval["ccxt"], limit=240
            )
            frame = pd.DataFrame(
                rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            return _normalise_market_frame(frame), "Binance public feed via ccxt"
        except Exception as exc:
            errors.append("ccxt: " + _safe_error(exc))
    raise RuntimeError(
        "No public candle connector returned data. " + " | ".join(errors)[:360]
    )


def _add_market_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    close = frame["close"]
    frame["sma_20"] = close.rolling(20, min_periods=1).mean()
    frame["ema_9"] = close.ewm(span=9, adjust=False, min_periods=1).mean()
    frame["ema_21"] = close.ewm(span=21, adjust=False, min_periods=1).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=1).mean()
    frame["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    frame["rsi_14"] = frame["rsi_14"].fillna(50.0).clip(0, 100)
    frame["macd"] = close.ewm(span=12, adjust=False, min_periods=1).mean() - close.ewm(
        span=26, adjust=False, min_periods=1
    ).mean()
    frame["macd_signal"] = frame["macd"].ewm(
        span=9, adjust=False, min_periods=1
    ).mean()
    frame["macd_histogram"] = frame["macd"] - frame["macd_signal"]
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14, min_periods=1).mean()
    return frame.replace([np.inf, -np.inf], np.nan).ffill().bfill()


def _load_live_market(
    pair: str,
    timeframe: str,
    mt5_login: str = "",
    mt5_password: str = "",
    mt5_server: str = "",
) -> dict:
    frame, source = _fetch_mt5_market_data(
        _LIVE_MARKETS[pair].get("mt5", pair),
        timeframe,
        mt5_login,
        mt5_password,
        mt5_server,
    )
    notice = ""
    if frame is None:
        frame, source = _fetch_public_market_data(pair, timeframe)
        notice = source if "fallback" in source.lower() else ""
    frame = _add_market_indicators(frame)
    latest = frame.iloc[-1]
    indicator_metrics = {
        "RSI (14)": round(float(latest["rsi_14"]), 2),
        "SMA (20)": round(float(latest["sma_20"]), 6),
        "EMA (9)": round(float(latest["ema_9"]), 6),
        "EMA (21)": round(float(latest["ema_21"]), 6),
        "MACD": round(float(latest["macd"]), 6),
        "MACD signal": round(float(latest["macd_signal"]), 6),
        "ATR (14)": round(float(latest["atr_14"]), 6),
    }
    return {
        "pair": pair,
        "timeframe": timeframe,
        "timeframe_minutes": _live_interval_config(timeframe)["minutes"],
        "source": source,
        "notice": notice,
        "frame": frame,
        "latest": {
            key: float(latest[key])
            for key in ("open", "high", "low", "close", "volume")
        },
        "indicator_metrics": indicator_metrics,
    }


def _live_serialisable_data(market: dict) -> dict:
    frame = market["frame"].tail(80)
    candles = []
    for timestamp, row in frame.iterrows():
        candles.append(
            {
                "timestamp": timestamp.isoformat(),
                "open": round(float(row["open"]), 8),
                "high": round(float(row["high"]), 8),
                "low": round(float(row["low"]), 8),
                "close": round(float(row["close"]), 8),
                "volume": round(float(row["volume"]), 4),
                "rsi_14": round(float(row["rsi_14"]), 4),
                "ema_9": round(float(row["ema_9"]), 8),
                "ema_21": round(float(row["ema_21"]), 8),
                "macd": round(float(row["macd"]), 8),
                "macd_signal": round(float(row["macd_signal"]), 8),
            }
        )
    return {
        "pair": market["pair"],
        "timeframe": market["timeframe"],
        "timeframe_minutes": market["timeframe_minutes"],
        "source": market["source"],
        "latest": market["latest"],
        "indicator_metrics": market["indicator_metrics"],
        "recent_candles": candles,
    }


def _build_live_prediction_prompt(market: dict) -> str:
    payload = json.dumps(_live_serialisable_data(market), ensure_ascii=False)
    return (
        "You are an active member of a multi-AI market prediction committee for "
        "Exness and Quotex traders. Analyze the supplied OHLCV candles and indicator "
        "metrics; do not claim to see an image. The market and timeframe in the data "
        "are authoritative. Return a directional vote and a numerical forecast for "
        "the next 4 candles. Use realistic OHLC relationships: high must be at least "
        "max(open, close), and low must be at most min(open, close). "
        "Do not invent a different pair or timeframe.\n\n"
        "MARKET DATA:\n"
        + payload
        + "\n\nReturn ONLY valid JSON:\n"
        "{"
        '"vote":"UP|DOWN|WAIT",'
        '"confidence":0-100,'
        '"analysis":"concise evidence from price and indicators",'
        '"key_risks":["specific risk"],'
        '"prediction_candles":['
        '{"candle_number":1,"open":0,"high":0,"low":0,"close":0},'
        '{"candle_number":2,"open":0,"high":0,"low":0,"close":0},'
        '{"candle_number":3,"open":0,"high":0,"low":0,"close":0},'
        '{"candle_number":4,"open":0,"high":0,"low":0,"close":0}'
        "]}"
    )


def _live_fallback_candles(market: dict, vote: str = "WAIT") -> list[dict]:
    latest = market["latest"]
    close = float(latest["close"])
    atr = max(float(market["indicator_metrics"].get("ATR (14)", 0) or 0), close * 0.0002)
    direction = 1 if vote == "UP" else (-1 if vote == "DOWN" else 0)
    candles = []
    previous = close
    for number in range(1, 5):
        opening = previous
        movement = direction * atr * 0.35
        closing = max(0.00000001, opening + movement)
        high = max(opening, closing) + atr * 0.18
        low = max(0.00000001, min(opening, closing) - atr * 0.18)
        candles.append(
            {
                "candle_number": number,
                "open": opening,
                "high": high,
                "low": low,
                "close": closing,
            }
        )
        previous = closing
    return candles


def _normalise_live_prediction(result: dict, market: dict, model: dict) -> dict:
    result = dict(result or {})
    result.setdefault("model", model.get("name", model.get("id", "AI model")))
    result.setdefault("_active_model", model)
    vote = _normalise_vote_fields(result, "Market-data analysis completed.")["vote"]
    raw_candles = (
        result.get("prediction_candles")
        or result.get("predicted_candles")
        or result.get("forecast_candles")
        or result.get("forecast")
        or []
    )
    if isinstance(raw_candles, dict):
        raw_candles = raw_candles.get("candles", [])
    candles = []
    if isinstance(raw_candles, list):
        for index, raw in enumerate(raw_candles[:4], start=1):
            if not isinstance(raw, dict):
                continue
            values = {}
            for key in ("open", "high", "low", "close"):
                try:
                    values[key] = float(raw.get(key))
                except (TypeError, ValueError):
                    values = {}
                    break
            if not values or min(values.values()) <= 0:
                continue
            values["candle_number"] = index
            values["high"] = max(values["high"], values["open"], values["close"])
            values["low"] = min(values["low"], values["open"], values["close"])
            candles.append(values)
    if len(candles) < 2:
        candles = _live_fallback_candles(market, vote)
        result["forecast_source"] = "Deterministic volatility fallback"
    else:
        result["forecast_source"] = "AI forecast"
    result["vote"] = vote
    result["prediction_candles"] = candles
    result["confidence"] = max(0, min(100, int(result.get("confidence", 50) or 50)))
    return result


def _run_live_prediction_round(market: dict, models: list[dict]) -> list[dict]:
    prompt = _build_live_prediction_prompt(market)
    credentials = {
        provider: (_provider_key(provider), _provider_account_id(provider))
        for provider in {model.get("provider", "") for model in models}
    }

    def run_model(model: dict) -> dict:
        provider_key, account_id = credentials.get(model.get("provider", ""), ("", ""))
        result = _call_activated_text_model(
            model, prompt, provider_key, account_id
        )
        return _normalise_live_prediction(result, market, model)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(models)))) as pool:
        futures = {pool.submit(run_model, model): model for model in models}
        for future, model in futures.items():
            try:
                results.append(future.result())
            except Exception:
                _retire_model(model)
    return results


def _build_live_deliberation_prompt(market: dict, votes: list[dict]) -> str:
    positions = []
    for vote in votes:
        positions.append(
            {
                "model": vote.get("model", "?"),
                "vote": vote.get("vote", "WAIT"),
                "confidence": vote.get("confidence", 0),
                "analysis": vote.get("analysis", ""),
                "key_risks": vote.get("key_risks", []),
                "prediction_candles": vote.get("prediction_candles", []),
            }
        )
    return (
        "You are revising your forecast after cross-examining every peer in a "
        "multi-AI trading debate. The pair and timeframe are authoritative. Compare "
        "the evidence and numerical forecasts, correct unrealistic OHLC values, then "
        "return your final vote and 2 to 4 predicted candles. Keep the prediction "
        "grounded in the latest OHLCV data and indicators.\n\n"
        "MARKET DATA:\n"
        + json.dumps(_live_serialisable_data(market), ensure_ascii=False)
        + "\n\nPEER POSITIONS:\n"
        + json.dumps(positions, ensure_ascii=False)
        + "\n\nReturn ONLY valid JSON with vote, confidence, analysis, key_risks, "
        "and prediction_candles."
    )


def _run_live_deliberation_round(market: dict, votes: list[dict]) -> list[dict]:
    prompt = _build_live_deliberation_prompt(market, votes)
    credentials = {
        provider: (_provider_key(provider), _provider_account_id(provider))
        for provider in {model.get("provider", "") for model in votes
                         for model in [model.get("_active_model") or {}]}
    }

    def deliberate(vote: dict) -> dict:
        model = vote.get("_active_model")
        if not isinstance(model, dict):
            return vote
        provider_key, account_id = credentials.get(model.get("provider", ""), ("", ""))
        result = _call_activated_text_model(
            model, prompt, provider_key, account_id
        )
        return _normalise_live_prediction(result, market, model)

    revised: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(votes)))) as pool:
        futures = {pool.submit(deliberate, vote): vote for vote in votes}
        for future, vote in futures.items():
            try:
                revised.append(future.result())
            except Exception:
                revised.append(vote)
    return revised


def _build_live_consensus(market: dict, votes: list[dict]) -> dict:
    tally = {"UP": 0, "DOWN": 0, "WAIT": 0}
    for vote in votes:
        direction = str(vote.get("vote", "WAIT")).upper()
        tally[direction if direction in tally else "WAIT"] += 1
    ordered = sorted(tally.items(), key=lambda item: item[1], reverse=True)
    decision = ordered[0][0] if ordered and (
        len(ordered) == 1 or ordered[0][1] > ordered[1][1]
    ) else "WAIT"
    total = sum(tally.values())
    confidence = int(round(100 * tally.get(decision, 0) / total)) if total else 0
    lengths = [
        len(vote.get("prediction_candles", []))
        for vote in votes
        if vote.get("prediction_candles")
    ]
    count = min(max(lengths or [4]), 4)
    if count < 2:
        count = 2
    consensus = []
    fallback = _live_fallback_candles(market, decision)
    for index in range(count):
        values = []
        for vote in votes:
            candles = vote.get("prediction_candles", [])
            if index < len(candles):
                values.append(candles[index])
        source = values or [fallback[index]]
        candle = {
            "candle_number": index + 1,
            "open": float(np.median([item["open"] for item in source])),
            "high": float(np.median([item["high"] for item in source])),
            "low": float(np.median([item["low"] for item in source])),
            "close": float(np.median([item["close"] for item in source])),
        }
        candle["high"] = max(candle["high"], candle["open"], candle["close"])
        candle["low"] = min(candle["low"], candle["open"], candle["close"])
        consensus.append(candle)
    return {
        "decision": decision,
        "confidence": confidence,
        "vote_tally": tally,
        "prediction_candles": consensus,
        "models_count": len(votes),
    }


def _live_prediction_frame(market: dict, consensus: list[dict]) -> pd.DataFrame:
    frame = market["frame"]
    if not consensus:
        return pd.DataFrame()
    step = timedelta(minutes=market["timeframe_minutes"])
    start = frame.index[-1] + step
    index = pd.date_range(start=start, periods=len(consensus), freq=step)
    return pd.DataFrame(consensus, index=index)


def _build_live_figure(market: dict, consensus: list[dict]) -> go.Figure:
    from plotly.subplots import make_subplots

    frame = market["frame"].tail(160)
    predicted = _live_prediction_frame(market, consensus)
    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.53, 0.13, 0.17, 0.17],
        subplot_titles=("OHLCV + Moving Averages", "Volume", "RSI (14)", "MACD"),
    )
    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Live candles",
            increasing_line_color="#00d084",
            decreasing_line_color="#ff4d6d",
        ),
        row=1,
        col=1,
    )
    for column, color in (
        ("sma_20", "#ffd166"),
        ("ema_9", "#00b4ff"),
        ("ema_21", "#c77dff"),
    ):
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[column],
                mode="lines",
                name=column.upper().replace("_", " "),
                line={"color": color, "width": 1.4},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Bar(x=frame.index, y=frame["volume"], name="Volume", marker_color="#4c6fff"),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["rsi_14"],
            name="RSI",
            line={"color": "#ff9f1c", "width": 1.5},
        ),
        row=3,
        col=1,
    )
    figure.add_hline(y=70, line_dash="dot", line_color="#ff4d6d", row=3, col=1)
    figure.add_hline(y=30, line_dash="dot", line_color="#00d084", row=3, col=1)
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["macd"],
            name="MACD",
            line={"color": "#00b4ff", "width": 1.5},
        ),
        row=4,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["macd_signal"],
            name="MACD signal",
            line={"color": "#ff9f1c", "width": 1.2},
        ),
        row=4,
        col=1,
    )
    if not predicted.empty:
        figure.add_trace(
            go.Candlestick(
                x=predicted.index,
                open=predicted["open"],
                high=predicted["high"],
                low=predicted["low"],
                close=predicted["close"],
                name="AI predicted candles",
                increasing_line_color="#ffe066",
                decreasing_line_color="#ff8fab",
                increasing_fillcolor="#6c5ce7",
                decreasing_fillcolor="#6c5ce7",
                opacity=0.9,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=predicted.index,
                y=predicted["close"],
                name="Consensus close path",
                mode="lines+markers",
                line={"color": "#ffe066", "width": 2, "dash": "dash"},
                marker={"size": 7, "symbol": "diamond"},
            ),
            row=1,
            col=1,
        )
    figure.update_layout(
        height=820,
        template="plotly_dark",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    figure.update_yaxes(showgrid=True, gridcolor="rgba(130,150,180,.16)")
    return figure


def _render_live_engine(result: dict) -> None:
    market = result["market"]
    consensus = result.get("consensus", {})
    st.markdown("---")
    st.markdown(
        "<div class='step-header'>📈 Live Candlestick Charting & Multi-AI "
        "Prediction Engine</div>",
        unsafe_allow_html=True,
    )
    source = escape(str(market.get("source", "public feed")))
    st.caption(
        f"{market['pair']} · {market['timeframe']} candles · Data source: {source} · "
        "Forecast candles are model estimates, not broker execution instructions."
    )
    if market.get("notice"):
        st.info(market["notice"])
    metrics = market["indicator_metrics"]
    metric_columns = st.columns(5)
    for column, label in zip(
        metric_columns,
        ("RSI (14)", "EMA (9)", "EMA (21)", "MACD", "ATR (14)"),
    ):
        column.metric(label, str(metrics[label]))
    st.plotly_chart(
        _build_live_figure(market, consensus.get("prediction_candles", [])),
        use_container_width=True,
        key="live_market_chart",
    )
    votes = result.get("revised_votes", [])
    if not votes:
        st.warning(
            "Live candles are available. Connect and activate AI models in the "
            "sidebar to generate the multi-AI forecast overlay."
        )
        return
    decision = consensus.get("decision", "WAIT")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Consensus", decision)
    summary_columns[1].metric("Confidence", str(consensus.get("confidence", 0)) + "%")
    summary_columns[2].metric("Models", str(consensus.get("models_count", 0)))
    summary_columns[3].metric(
        "Forecast horizon",
        str(len(consensus.get("prediction_candles", []))) + " candles",
    )
    with st.expander("AI debate breakdown", expanded=True):
        st.write(
            "Round 1 forecasts are cross-examined in Round 2. The overlay uses the "
            "median OHLC forecast from the post-debate model positions."
        )
        for vote in votes:
            label = vote.get("model", "AI model")
            platform = vote.get("_platform", "")
            st.markdown(
                f"**{escape(str(label))} · {escape(str(platform))}** — "
                f"**{vote.get('vote', 'WAIT')}** · {vote.get('confidence', 0)}%"
            )
            st.caption(str(vote.get("analysis") or vote.get("reasoning") or "No explanation returned."))
            st.dataframe(
                pd.DataFrame(vote.get("prediction_candles", [])),
                use_container_width=True,
                hide_index=True,
            )
    st.markdown("**Consensus predicted candles**")
    st.dataframe(
        pd.DataFrame(consensus.get("prediction_candles", [])),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Educational market analysis only. Forecasts can be wrong; verify price, "
        "spread, liquidity, and broker conditions before making any decision."
    )


def _run_live_engine(request: dict, models: list[dict]) -> dict:
    market = _load_live_market(
        request["pair"],
        request["timeframe"],
        request.get("mt5_login", ""),
        request.get("mt5_password", ""),
        request.get("mt5_server", ""),
    )
    round_one = _run_live_prediction_round(market, models) if models else []
    revised = _run_live_deliberation_round(market, round_one) if round_one else []
    consensus = _build_live_consensus(market, revised)
    return {
        "market": market,
        "round_one": round_one,
        "revised_votes": revised,
        "consensus": consensus,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — secure provider/model manager
# ─────────────────────────────────────────────────────────────────────────────
if "provider_keys" not in st.session_state:
    st.session_state["provider_keys"] = {}
if "active_models" not in st.session_state:
    st.session_state["active_models"] = []
if "available_models" not in st.session_state:
    st.session_state["available_models"] = []
if "provider_errors" not in st.session_state:
    st.session_state["provider_errors"] = {}
if "custom_models" not in st.session_state:
    st.session_state["custom_models"] = []
if not st.session_state["available_models"] and st.session_state["active_models"]:
    st.session_state["available_models"] = list(st.session_state["active_models"])
# A prior session may contain catalog entries that the current capability
# filter no longer considers callable through chat completions.
st.session_state["available_models"] = [
    model for model in st.session_state["available_models"]
    if _model_is_chat_capable(model.get("id", ""), model)
]
st.session_state["active_models"] = [
    model for model in st.session_state["active_models"]
    if _model_is_chat_capable(model.get("id", ""), model)
]

# Keep the existing server-side Groq setup useful on first load.
if not st.session_state["active_models"] and get_secret("GROQ_API_KEY"):
    try:
        st.session_state["provider_keys"]["GROQ_API_KEY"] = get_secret("GROQ_API_KEY")
        discovered = _discover_provider_models("Groq", get_secret("GROQ_API_KEY"))
        st.session_state["available_models"] = discovered
        st.session_state["active_models"] = discovered
    except Exception as exc:
        st.session_state["provider_errors"]["Groq"] = _safe_error(exc)

with st.sidebar:
    st.markdown(
        "<h2 style='font-size:1.1rem;font-weight:800;color:#c8d6e8;"
        "border-bottom:1px solid #1a2340;padding-bottom:8px;margin-bottom:8px;'>"
        "🧩 AI Model Configuration</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Keys stay in this server session. Connect a provider to discover and "
        "activate every usable chat model available to that key."
    )
    provider = st.selectbox(
        "AI provider / aggregator",
        list(_PROVIDER_CONFIG),
        format_func=_provider_label,
        key="manager_provider",
    )
    with st.form("provider_connection_form", clear_on_submit=True):
        api_key_input = st.text_input(
            "API key",
            type="password",
            placeholder="Paste a key to discover models",
            help="The key is used only by the server to call this provider's models endpoint.",
        )
        connect = st.form_submit_button(
            "🔐 Connect & discover models", type="primary", use_container_width=True
        )
    if connect:
        if not api_key_input.strip():
            st.error("Paste an API key first.")
        else:
            with st.spinner("Validating key and discovering usable chat models…"):
                try:
                    discovered = _discover_provider_models(provider, api_key_input.strip())
                    st.session_state["provider_keys"][
                        _PROVIDER_CONFIG[provider]["key"]
                    ] = api_key_input.strip()
                    st.session_state["available_models"] = [
                        model for model in st.session_state["available_models"]
                        if model["provider"] != provider
                    ] + discovered
                    st.session_state["active_models"] = [
                        model for model in st.session_state["active_models"]
                        if model["provider"] != provider
                    ] + discovered
                    st.session_state["provider_errors"].pop(provider, None)
                    st.success(
                        f"Discovered and activated {len(discovered)} {provider} model(s)."
                    )
                    st.rerun()
                except Exception as exc:
                    message = _safe_error(exc)
                    st.session_state["provider_errors"][provider] = message[:240]
                    st.error("Could not connect: " + message[:240])

    connected = sorted({model["provider"] for model in st.session_state["available_models"]})
    if connected:
        st.markdown("**Active model committee**")
        st.caption(
            f"{len(_active_models())} models active · "
            + ", ".join(connected)
        )
        with st.expander("Manage active models", expanded=True):
            active_ids = {model["provider"] + ":" + model["id"] for model in _active_models()}
            for model in st.session_state["available_models"]:
                model_key = model["provider"] + ":" + model["id"]
                checked = st.checkbox(
                    _model_label(model) + (" · vision" if model.get("vision") else " · text"),
                    value=model_key in active_ids,
                    key="activate_" + re.sub(r"[^a-zA-Z0-9_]", "_", model_key),
                )
                if checked and model_key not in active_ids:
                    st.session_state["active_models"].append(model)
                elif not checked and model_key in active_ids:
                    st.session_state["active_models"] = [
                        item for item in st.session_state["active_models"]
                        if item["provider"] + ":" + item["id"] != model_key
                    ]
    else:
        st.info("No models active yet. Connect OpenRouter or another provider above.")
    st.caption(
        "Vision models receive the picture directly. Text-only models receive the "
        "structured chart record extracted from the picture."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(_PREMIUM_CSS, unsafe_allow_html=True)

st.markdown(
    "<h1 style='font-size:2.1rem;font-weight:900;letter-spacing:.01em;"
    "background:linear-gradient(90deg,#00b4ff,#00ff88);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "margin-bottom:0;'>📊 AI Trading Committee</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#566880;font-size:.88rem;margin-top:4px;'>"
    "Crash-proof &nbsp;·&nbsp; live AI debate &nbsp;·&nbsp; "
    "auto-fallback on any outage &nbsp;·&nbsp; candle-based trade target</p>",
    unsafe_allow_html=True,
)

st.markdown("<div class='step-header'>1️⃣ Upload Chart</div>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload chart screenshot",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
    help="Screenshot of any trading chart",
)
if uploaded:
    st.image(Image.open(uploaded), caption="Uploaded chart", width="stretch")
    uploaded.seek(0)

with st.form("debate_inputs_form", clear_on_submit=False):
    st.markdown(
        "<div class='step-header'>2️⃣ Extra Context "
        "<span style='font-weight:400;font-size:.8rem;color:#566880;'>(optional)</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    extra_ctx = st.text_area(
        "context",
        label_visibility="collapsed",
        placeholder="e.g. BTC/USDT 10-min chart, NY session open…",
        height=70,
    )
    st.markdown("<div class='step-header'>3️⃣ Start the Debate</div>", unsafe_allow_html=True)
    run = st.form_submit_button(
        "🚀  START AI DEBATE",
        disabled=(uploaded is None),
        use_container_width=True,
        type="primary",
    )

cached_debate = st.session_state.get("last_debate")
if not run and not isinstance(cached_debate, dict):
    st.stop()

# ── Read image bytes ──────────────────────────────────────────────────────────
if run:
    uploaded.seek(0)
    image_bytes = uploaded.read()
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")
else:
    image_bytes = b""
    mime_type = cached_debate.get("mime_type", "image/png")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Chart vision
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='step-header'>Step 1 — Chart Analysis & Market News</div>",
            unsafe_allow_html=True)
if run:
    with st.spinner("🔍 Reading chart… Groq vision model…"):
        chart_data, vision_status = step1_analyze_chart(image_bytes, mime_type, extra_ctx)

    asset_name = chart_data.get("asset", "asset")
    # Store notes and make explicit trader-supplied timeframe authoritative.
    chart_data = _apply_user_context(chart_data, extra_ctx)

    with st.spinner("🌐 Searching latest market news for " + asset_name + "…"):
        chart_data = _inject_real_news(chart_data)
else:
    chart_data = cached_debate["chart_data"]
    vision_status = cached_debate["vision_status"]

render_chart_analysis(chart_data, vision_status)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Independent analyst votes
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div class='step-header'>Step 2 — Round 1: Independent Analyst Votes</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:.78rem;color:#566880;margin:-6px 0 8px;'>"
    "Each AI votes independently with no knowledge of other opinions. "
    "A deliberation round follows where they share logic and can revise.</p>",
    unsafe_allow_html=True,
)

active_models = _active_models()
_model_labels = ", ".join(_model_label(model) for model in active_models)
if not _model_labels:
    _model_labels = "No active models"
st.markdown(
    "<p style='font-size:.8rem;color:#566880;margin-bottom:12px;'>"
    "🤖 Active roster: <strong style='color:#a0b8d0;'>" + escape(_model_labels) + "</strong></p>",
    unsafe_allow_html=True,
)

if run:
    analyst_votes: list[dict] = []
    runtime_credentials = {
        provider: (_provider_key(provider), _provider_account_id(provider))
        for provider in {model["provider"] for model in active_models}
    }

    def _run_first_round_model(model: dict) -> dict:
        """Run one independent analyst without touching Streamlit from a worker."""
        provider_key, account_id = runtime_credentials.get(model["provider"], ("", ""))
        if model.get("vision") and model.get("id") == chart_data.get("_vision_model_id"):
            # The primary vision pass is already this model's answer. Reuse it
            # instead of spending a second request and triggering provider rate limits.
            result = {
                **chart_data,
                "model": model["name"],
                "_platform": model["provider"],
                "_provider": model["provider"],
                "_model_id": model["id"],
                "_active_model": model,
            }
            return _normalise_vote_fields(
                result, "The uploaded chart was read by " + model["name"] + "."
            )
        if model.get("vision"):
            result = _call_activated_model(
                model,
                image_bytes,
                mime_type,
                _activated_chart_prompt(extra_ctx),
                provider_key,
                account_id,
            )
            return _normalise_vote_fields(
                _apply_user_context(result, extra_ctx),
                "The uploaded chart was read by " + model["name"] + ".",
            )
        result = _call_activated_text_model(
            model,
            _activated_vote_prompt(chart_data, extra_ctx),
            provider_key,
            account_id,
        )
        return _normalise_vote_fields(
            _apply_user_context(result, extra_ctx),
            "The structured chart record was reviewed by " + model["name"] + ".",
        )

    # Parallel requests keep a slow Gemini model from delaying fast Groq models.
    with st.spinner("⚡ Running independent model votes in parallel…"):
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(active_models)))) as pool:
            futures = {
                pool.submit(_run_first_round_model, model): model
                for model in active_models
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    outcome = future.result()
                    analyst_votes.append(outcome)
                    # Fast providers appear immediately; slower Gemini calls
                    # continue in the background without blocking their cards.
                    render_vote_card(outcome, status="online")
                except Exception as exc:
                    _retire_model(model)
else:
    analyst_votes = cached_debate["analyst_votes"]

if not analyst_votes:
    st.info("No currently available model returned a result. Reconnect a provider and try again.")
    st.stop()

render_scoreboard(chart_data, analyst_votes, [])

# ══════════════════════════════════════════════════════════════════════════════
# DELIBERATION — AIs share full reasoning and submit revised final votes
# ══════════════════════════════════════════════════════════════════════════════
if run:
    with st.spinner("🔄 Running deliberation round — AIs reading each other's reasoning…"):
        revised_votes = run_deliberation_round(analyst_votes, chart_data)
else:
    revised_votes = cached_debate["revised_votes"]

render_deliberation_round(analyst_votes, revised_votes)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cross-examination & final verdict (uses post-deliberation votes)
# ══════════════════════════════════════════════════════════════════════════════
n_online = len(revised_votes)
n_total  = len(analyst_votes)
st.markdown("---")
st.markdown(
    "<div class='step-header'>Step 3 — Cross-Examination & Final Verdict"
    "<span style='font-size:.8rem;font-weight:400;color:#566880;margin-left:8px;'>"
    "based on post-deliberation positions · "
    + str(n_online) + "/" + str(n_total) + " models online</span></div>",
    unsafe_allow_html=True,
)
if run:
    with st.spinner("🧠 Cross-examining all deliberated positions and computing final verdict…"):
        try:
            synthesis = step3_synthesize(chart_data, revised_votes)
        except Exception as exc:
            st.error("**Final synthesis failed:** " + _safe_error(exc))
            st.stop()
    st.session_state["last_debate"] = {
        "chart_data": chart_data,
        "vision_status": vision_status,
        "analyst_votes": analyst_votes,
        "revised_votes": revised_votes,
        "synthesis": synthesis,
        "mime_type": mime_type,
    }
else:
    synthesis = cached_debate["synthesis"]

render_final_decision(synthesis)
render_trade_recommendation(synthesis)
if run:
    st.balloons()
