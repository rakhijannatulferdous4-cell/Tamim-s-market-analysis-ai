"""
AI Trading Debate — clean, future-proof, crash-proof.

Pipeline
────────
Step 1  Chart Vision  : Gemini 1.5 Flash (single shot, immediate fallback)
                        → llama-3.2-11b-vision-preview (Groq, hardcoded first)
                        → dynamic Groq vision discovery (any model with 'vision')
                        → text-fallback if all vision APIs fail
Step 2  Analyst Votes : Top-3 Groq text models (auto-selected live)
                          preference: llama-3.3-70b-versatile, qwen-2.5-32b
                        + DeepSeek-Chat (skipped gracefully on 402)
                        <think>…</think> tokens stripped from Qwen/DeepSeek before parsing
Step 3  Final Verdict : Gemini 1.5 Flash synthesis → top Groq model fallback
                        → FINAL_DECISION + candle recommendation box
"""

import base64
import json
import os
import re
import time

import requests
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Debate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Secret helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_secret(key: str) -> str:
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
            "Open the Replit **Secrets** panel (🔒 icon) and add it there."
        )
        st.stop()
    return val


# ─────────────────────────────────────────────────────────────────────────────
# JSON parser — handles <think> tokens, markdown fences, and bare JSON
# ─────────────────────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict:
    raw = text or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start < 0:
        raise ValueError(f"No JSON object found in response:\n{text[:400]}")
    return json.loads(raw[start:end])


# ─────────────────────────────────────────────────────────────────────────────
# Groq model pools
# ─────────────────────────────────────────────────────────────────────────────
_GROQ_TEXT_EXCLUDE = frozenset(
    ["vision", "whisper", "guard", "embed", "tts", "distil"]
)

_GROQ_TEXT_FALLBACK = [
    ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
    ("qwen-2.5-32b",            "Qwen 2.5 32B"),
    ("llama-3.1-8b-instant",    "Llama 3.1 8B"),
]


def _discover_groq_vision_models() -> list[tuple[str, str]]:
    """
    Return all vision-capable model IDs from Groq's live catalog.
    Filters for any model whose ID contains 'vision'.
    Returns an empty list if the API call fails — never raises.
    """
    try:
        from groq import Groq

        client = Groq(api_key=require_secret("GROQ_API_KEY"))
        listing = client.models.list()
        return [(m.id, m.id) for m in listing.data if "vision" in m.id.lower()]
    except Exception:
        return []


def _discover_groq_text_models(n: int = 3) -> list[tuple[str, str]]:
    """
    Return up to `n` text-only model IDs from Groq's live catalog,
    ranked by a quality heuristic (family + parameter count).

    Excludes vision, audio, guard, and embed models.
    Falls back to _GROQ_TEXT_FALLBACK if discovery fails or yields nothing.
    """
    try:
        from groq import Groq

        client = Groq(api_key=require_secret("GROQ_API_KEY"))
        listing = client.models.list()

        # Explicit top-priority models get a guaranteed high score
        _PREFERRED = {
            "llama-3.3-70b-versatile": 200,
            "qwen-2.5-32b":            190,
        }

        candidates: list[tuple[int, str]] = []
        for m in listing.data:
            mid = m.id.lower()
            if any(x in mid for x in _GROQ_TEXT_EXCLUDE):
                continue
            # Check explicit preferences first
            if m.id in _PREFERRED:
                candidates.append((_PREFERRED[m.id], m.id))
                continue
            # Score by family (higher = newer / more capable)
            score = 0
            if "llama-3.3" in mid:
                score += 100
            elif "qwen" in mid:
                score += 90  # Qwen 2.5 / Qwen3 are top-tier
            elif "llama-3.2" in mid:
                score += 80
            elif "llama-3.1" in mid:
                score += 70
            elif "llama-3" in mid:
                score += 60
            elif "mixtral" in mid:
                score += 55
            elif "gemma" in mid:
                score += 50
            else:
                score += 10
            # Bonus for larger parameter counts
            if any(x in mid for x in ["70b", "72b", "32b"]):
                score += 30
            elif any(x in mid for x in ["34b", "13b"]):
                score += 20
            elif "8b" in mid:
                score += 15
            elif "7b" in mid:
                score += 12
            candidates.append((score, m.id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        result = [(mid, mid) for _, mid in candidates[:n]]
        return result if result else _GROQ_TEXT_FALLBACK[:n]
    except Exception:
        return _GROQ_TEXT_FALLBACK[:n]


def _short_label(model_id: str) -> str:
    """Human-readable short label for any Groq model ID."""
    mid = model_id.lower()
    for size in ["72b", "70b", "34b", "32b", "13b", "9b", "8b", "7b"]:
        if size in mid:
            if "llama-3.3" in mid:
                return f"Llama 3.3 {size.upper()}"
            if "llama-3.2" in mid:
                return f"Llama 3.2 {size.upper()}"
            if "llama-3.1" in mid:
                return f"Llama 3.1 {size.upper()}"
            if "llama-3" in mid:
                return f"Llama 3 {size.upper()}"
            if "mixtral" in mid:
                return f"Mixtral {size.upper()}"
            if "gemma" in mid:
                return f"Gemma {size.upper()}"
            if "qwen" in mid:
                return f"Qwen {size.upper()}"
    return model_id.split("/")[-1].replace("-", " ").title()


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
  "vision_model": "<which model performed this analysis>",
  "gemini_vote": "<UP|DOWN|WAIT>",
  "gemini_confidence": <0-100>,
  "gemini_reasoning": "<one concise sentence>"
}"""

CHART_PROMPT_PREFIX = (
    "You are a senior technical analyst and financial journalist.\n\n"
    "Analyse the trading chart image in full detail. "
    "Include support/resistance levels, trend direction, indicators, and chart patterns. "
    "Also provide the latest relevant market news for the asset shown "
    "(use search if available, otherwise use your best knowledge).\n\n"
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
    "gemini_vote": "WAIT",
    "gemini_confidence": 0,
    "gemini_reasoning": "Vision analysis unavailable — apply strict risk management.",
}


def _gemini_vision(image_bytes: bytes, mime: str, extra: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_secret("GEMINI_API_KEY"))
    extra_line = (f"\n\nExtra context: {extra.strip()}") if extra.strip() else ""
    prompt = CHART_PROMPT_PREFIX + extra_line + "\n" + CHART_ANALYSIS_JSON_SPEC

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(inline_data=types.Blob(mime_type=mime, data=image_bytes)),
                types.Part(text=prompt),
            ],
        )
    ]
    # Try with Google Search grounding first; fall back to plain if that errors
    try:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )
    except Exception:
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents,
        )
    result = parse_json(resp.text)
    result["vision_model"] = "Gemini 1.5 Flash"
    return result


def _groq_vision(
    image_bytes: bytes, mime: str, extra: str, model_id: str, model_label: str
) -> dict:
    from groq import Groq

    client = Groq(api_key=require_secret("GROQ_API_KEY"))
    b64_image = base64.b64encode(image_bytes).decode()
    extra_line = (f"\n\nExtra context: {extra.strip()}") if extra.strip() else ""
    prompt = (
        CHART_PROMPT_PREFIX
        + extra_line
        + "\n"
        + CHART_ANALYSIS_JSON_SPEC
        + "\n\nNote: You may not have live search. Fill live_news with your best "
        "knowledge of recent market events for the asset shown."
    )
    chat = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.3,
        max_tokens=1800,
    )
    result = parse_json(chat.choices[0].message.content)
    result["vision_model"] = model_label
    result.setdefault("gemini_vote", result.get("vote", "WAIT"))
    result.setdefault("gemini_confidence", result.get("confidence", 50))
    result.setdefault(
        "gemini_reasoning", result.get("reasoning", "Fallback vision model.")
    )
    return result


def step1_analyze_chart(image_bytes: bytes, mime: str, extra: str) -> tuple[dict, str]:
    """
    Returns (chart_data, status_message). Never raises.

    Strategy
    ────────
    1. Gemini 1.5 Flash — single attempt; on ANY error immediately fall through.
    2. Groq vision — hardcoded llama-3.2-11b-vision-preview tried first (fastest),
       then any additional vision models discovered live from Groq's catalog.
       Image bytes are base64-encoded as a data:{mime};base64,… URI — correct format.
    3. Text fallback — debate still runs with high-volatility context if all fail.
    """
    # ── 1. Gemini — single shot, immediate fallback on any error ──────────────
    gemini_error = ""
    try:
        data = _gemini_vision(image_bytes, mime, extra)
        return data, "✨ Gemini 1.5 Flash — chart analysis complete."
    except Exception as exc:
        gemini_error = f"{type(exc).__name__}: {str(exc)[:200]}"

    # ── 2. Groq vision — hardcoded first, then dynamic discovery ─────────────
    # Build ordered list: hardcoded active model, then any others from catalog.
    HARDCODED = "llama-3.2-11b-vision-preview"
    groq_vision_queue: list[str] = [HARDCODED]
    for mid, _ in _discover_groq_vision_models():
        if mid not in groq_vision_queue:
            groq_vision_queue.append(mid)

    groq_errors: list[str] = []
    for vision_model in groq_vision_queue:
        try:
            data = _groq_vision(image_bytes, mime, extra, vision_model, vision_model)
            # Keep Gemini slot in the committee scoreboard — show as Offline
            data.setdefault("gemini_vote",       "Offline")
            data.setdefault("gemini_confidence", 0)
            data.setdefault("gemini_reasoning",
                            f"Gemini temporarily offline — {gemini_error[:120]}")
            return data, (
                f"⚠️ Gemini offline ({gemini_error[:140]}). "
                f"Fell back to **{vision_model}** (Groq vision) — analysis complete."
            )
        except Exception as exc:
            groq_errors.append(
                f"{vision_model}: {type(exc).__name__}: {str(exc)[:100]}"
            )

    # ── 3. Safe text fallback — debate continues regardless ──────────────────
    all_errors = (
        "Gemini: " + gemini_error + " || Groq: " + " | ".join(groq_errors)
    )
    status = (
        "🚨 **All vision APIs failed** — running in text-fallback mode.\n\n"
        f"Errors: {all_errors[:300]}\n\n"
        "The AI committee will debate using a **highly volatile / high-risk** "
        "context. Chart-specific signals will not be available."
    )
    fallback = dict(_TEXT_FALLBACK_DATA)
    if extra.strip():
        fallback["technical_summary"] += f"  User context: {extra.strip()}"
    return fallback, status


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Independent analyst votes
# ─────────────────────────────────────────────────────────────────────────────
def build_analyst_prompt(model_name: str, g: dict) -> str:
    news_lines = (
        "\n".join(
            f"  • [{n.get('sentiment', '?')}] {n.get('headline', '')} — {n.get('source', '')}"
            for n in g.get("live_news", [])
        )
        or "  No live news available."
    )

    indicators = (
        "; ".join(f"{k}: {v}" for k, v in g.get("indicators", {}).items()) or "N/A"
    )

    return (
        f"You are {model_name}, an expert AI trading analyst.\n\n"
        "Study the following chart analysis and market news, then give your own "
        "independent trading opinion.\n\n"
        "=== CHART ANALYSIS ===\n"
        f"Vision Model : {g.get('vision_model', '?')}\n"
        f"Asset        : {g.get('asset', '?')}\n"
        f"Timeframe    : {g.get('timeframe', '?')}\n"
        f"Price        : {g.get('current_price', '?')}\n"
        f"Trend        : {g.get('trend', '?')}\n"
        f"Support      : {', '.join(g.get('support', []))}\n"
        f"Resistance   : {', '.join(g.get('resistance', []))}\n"
        f"Indicators   : {indicators}\n"
        f"Patterns     : {', '.join(g.get('patterns', []))}\n"
        f"Summary      : {g.get('technical_summary', '')}\n"
        f"Vision Vote  : {g.get('gemini_vote', '?')} ({g.get('gemini_confidence', 0)}%)\n"
        f"Vision Says  : {g.get('gemini_reasoning', '')}\n\n"
        "=== MARKET NEWS ===\n"
        f"{news_lines}\n"
        f"Summary: {g.get('news_summary', '')}\n"
        "===================\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        f'  "model": "{model_name}",\n'
        '  "analysis": "<2-3 sentence technical + fundamental view>",\n'
        '  "key_risks": ["<risk 1>", "<risk 2>"],\n'
        '  "vote": "<UP|DOWN|WAIT>",\n'
        '  "confidence": <0-100>,\n'
        '  "reasoning": "<one concise sentence>"\n'
        "}"
    )


def _call_groq(model_id: str, model_name: str, chart_data: dict) -> dict:
    from groq import Groq

    client = Groq(api_key=require_secret("GROQ_API_KEY"))
    prompt = build_analyst_prompt(model_name, chart_data)
    chat = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "You are an expert AI trading analyst. Respond with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=600,
    )
    result = parse_json(chat.choices[0].message.content)
    result.setdefault("model", model_name)
    return result


def _call_deepseek(chart_data: dict) -> dict:
    api_key = require_secret("DEEPSEEK_API_KEY")
    prompt = build_analyst_prompt("DeepSeek-Chat", chart_data)
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert AI trading analyst. Respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 600,
        },
        timeout=40,
    )
    # 402 = account has no balance — skip gracefully, don't crash the debate
    if resp.status_code == 402:
        raise RuntimeError(
            "💳 DeepSeek is waiting for a top-up (402 Payment Required). "
            "Skipping — remaining analysts will carry the debate."
        )
    resp.raise_for_status()
    result = parse_json(resp.json()["choices"][0]["message"]["content"])
    result.setdefault("model", "DeepSeek-Chat")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Synthesis & final verdict
# ─────────────────────────────────────────────────────────────────────────────
def step3_synthesize(chart_data: dict, analyst_votes: list[dict]) -> dict:
    """
    Cross-examines all analyst opinions and returns FINAL_DECISION +
    candle-based trade recommendation.
    Primary: Gemini 2.5 Flash.  Fallback: top available Groq text model.
    """
    tf = chart_data.get("timeframe", "unknown")
    tf_minutes = chart_data.get("timeframe_minutes", 0)

    positions = f"Vision Model ({chart_data.get('vision_model', '?')}): "
    positions += f"{chart_data.get('gemini_vote', 'WAIT')} "
    positions += f"({chart_data.get('gemini_confidence', 0)}%) — "
    positions += chart_data.get("gemini_reasoning", "") + "\n"
    for v in analyst_votes:
        positions += (
            f"{v.get('model', '?')}: {v.get('vote', 'WAIT')} "
            f"({v.get('confidence', 0)}%) — {v.get('reasoning', '')}\n"
            f"  Analysis  : {v.get('analysis', '')}\n"
            f"  Key Risks : {', '.join(v.get('key_risks', []))}\n"
        )

    online_names = [chart_data.get("vision_model", "Vision")] + [
        v.get("model", "?") for v in analyst_votes
    ]

    prompt = (
        "You are the debate moderator for an AI trading analyst panel.\n\n"
        f"Models that responded ({len(online_names)} online): {', '.join(online_names)}\n\n"
        "=== ALL ANALYST POSITIONS ===\n"
        + positions
        + "==============================\n\n"
        "Tasks:\n"
        "1. Cross-examine — identify key agreements and disagreements.\n"
        "2. Decide FINAL_DECISION (UP / DOWN / WAIT) based on evidence weight.\n"
        "3. Compute RECOMMENDED_ACTION using the chart timeframe:\n"
        f"   Timeframe: {tf} ({tf_minutes} minutes per candle)\n"
        "   • UP or DOWN → estimate candle_count (1-5), calculate "
        "total_duration_minutes = candle_count × timeframe_minutes.\n"
        '     display_text: "TRADE DIRECTION: X | TARGET: Next N candles will go X '
        '(Duration: Y minutes on a Z-min chart)"\n'
        "   • WAIT → should_trade = false, explain specific reason.\n"
        '     display_text: "DON\'T TRADE: <reason with asset and timeframe>"\n\n'
        "Return ONLY valid JSON — no markdown:\n"
        "{\n"
        '  "cross_examination": "<3-4 sentence moderator analysis>",\n'
        '  "vote_tally": {"UP": 0, "DOWN": 0, "WAIT": 0},\n'
        '  "consensus_strength": "<STRONG|MODERATE|DIVIDED>",\n'
        '  "FINAL_DECISION": "<UP|DOWN|WAIT>",\n'
        '  "confidence": <0-100>,\n'
        '  "moderator_note": "<2-3 sentence summary for the trader>",\n'
        '  "recommended_action": {\n'
        '    "should_trade": <true|false>,\n'
        '    "trade_direction": "<UP|DOWN|null>",\n'
        '    "candle_count": <integer or null>,\n'
        '    "total_duration_minutes": <integer or null>,\n'
        '    "dont_trade_reason": "<string or null>",\n'
        '    "display_text": "<formatted string per rules above>"\n'
        "  }\n"
        "}"
    )

    # Primary: Gemini 1.5 Flash (consistent with vision step)
    synth_model = "Gemini 1.5 Flash"
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=require_secret("GEMINI_API_KEY"))
        resp = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        )
        result = parse_json(resp.text)
    except Exception as gemini_err:
        # Fallback: top available Groq text model (dynamically discovered)
        try:
            from groq import Groq

            fallback_models = _discover_groq_text_models(n=1)
            fallback_id = fallback_models[0][0]
            client_g = Groq(api_key=require_secret("GROQ_API_KEY"))
            chat = client_g.chat.completions.create(
                model=fallback_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading debate moderator. Respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            result = parse_json(chat.choices[0].message.content)
            synth_model = _short_label(fallback_id) + " (Groq fallback)"
        except Exception as groq_err:
            raise RuntimeError(
                f"Both synthesis models failed.\nGemini: {gemini_err}\nGroq: {groq_err}"
            )

    result["_synth_model"] = synth_model
    return result


# ─────────────────────────────────────────────────────────────────────────────
# UI constants
# ─────────────────────────────────────────────────────────────────────────────
VOTE_COLOR = {"UP": "green", "DOWN": "red", "WAIT": "orange"}
VOTE_ICON  = {"UP": "⬆️",   "DOWN": "⬇️", "WAIT": "⏸️"}

_NEON = {
    "UP":      "#00ff88",
    "DOWN":    "#ff3860",
    "WAIT":    "#ffaa00",
    "OFFLINE": "#555",
    "blue":    "#00b4ff",
    "border":  "#1a2340",
    "card":    "#0d1220",
    "bg":      "#080c17",
}

_PREMIUM_CSS = """
<style>
/* ── Base dark theme ──────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080c17 !important;
    color: #c8d6e8;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"],
[data-testid="stToolbar"]        { background: transparent !important; }
[data-testid="stSidebar"]        { background: #0d1220 !important; }
[data-testid="stMain"] > div     { padding-top: 1.2rem; }

/* ── Section dividers ─────────────────────────────────────────────── */
hr { border-color: #1a2340 !important; }

/* ── Metrics ──────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #0d1220;
    border: 1px solid #1a2340;
    border-radius: 10px;
    padding: 10px 14px;
}

/* ── File uploader ────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0d1220;
    border: 2px dashed #1a3560;
    border-radius: 12px;
    padding: 8px;
}
[data-testid="stFileUploader"]:hover { border-color: #00b4ff; }

/* ── Text area ────────────────────────────────────────────────────── */
textarea {
    background: #0d1220 !important;
    color: #c8d6e8 !important;
    border: 1px solid #1a2340 !important;
    border-radius: 10px !important;
}
textarea:focus { border-color: #00b4ff !important; }

/* ── Expanders ────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0d1220 !important;
    border: 1px solid #1a2340 !important;
    border-radius: 12px !important;
}

/* ── START button — electric-blue pulse ───────────────────────────── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0057ff, #00b4ff) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    animation: btnPulse 2.4s ease-in-out infinite;
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #0070ff, #33c6ff) !important;
    box-shadow: 0 0 22px #00b4ff88 !important;
    animation: none;
}
@keyframes btnPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,180,255,.55); }
    50%       { box-shadow: 0 0 0 14px rgba(0,180,255,0); }
}

/* ── Analyst / info cards — fade-in ──────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ai-card {
    background: #0d1220;
    border: 1px solid #1a2340;
    border-radius: 14px;
    padding: 18px 20px 14px;
    margin: 10px 0;
    animation: fadeUp .55s ease both;
    box-shadow: 0 4px 24px rgba(0,0,0,.45);
}
.ai-card:hover { border-color: #00b4ff55; box-shadow: 0 4px 28px rgba(0,180,255,.15); }
.ai-card.offline { border-color: #ff386040; opacity: .75; }

.card-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
.model-lbl { font-size:.95rem; font-weight:700; color:#e0eaf8; }
.vote-pill {
    font-size:.8rem; font-weight:800; padding:3px 11px;
    border-radius:20px; border: 1.5px solid; letter-spacing:.05em;
}
.conf-lbl  { font-size:.78rem; color:#6a82a0; margin-left:auto; }

.analysis-txt { font-size:.88rem; color:#a8bcd4; line-height:1.6; margin:6px 0; }
.risks-row    { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
.risk-tag {
    font-size:.74rem; padding:3px 9px; border-radius:6px;
    background:#12192e; border:1px solid #1e2f50; color:#7a9abf;
}
.reasoning-txt { font-size:.83rem; color:#5e7a9a; font-style:italic; margin-top:6px; }
.err-txt  { font-size:.82rem; color:#ff6080; margin-top:6px; }

/* ── Scoreboard tiles ─────────────────────────────────────────────── */
.sb-tile {
    background: #0d1220;
    border: 1.5px solid #1a2340;
    border-radius: 10px;
    padding: 10px 6px;
    text-align: center;
    animation: fadeUp .5s ease both;
}
.sb-tile:hover { border-color: #00b4ff55; }
.sb-model-lbl { font-size:.65rem; color:#566880; margin-bottom:4px; }
.sb-vote-icon { font-size:1.4rem; }
.sb-vote-txt  { font-size:.88rem; font-weight:800; }
.sb-conf-txt  { font-size:.65rem; color:#566880; margin-top:2px; }

/* ── Final verdict banner ─────────────────────────────────────────── */
.verdict-banner {
    text-align: center;
    padding: 28px 0 12px;
    animation: fadeUp .6s ease both;
}
.verdict-word { font-size: 4.8rem; font-weight: 900; line-height: 1; }
.verdict-sub  { font-size: 1rem; color: #6a82a0; margin-top: 6px; }

/* ── Trade recommendation boxes ───────────────────────────────────── */
.trade-box {
    border-radius: 16px;
    padding: 32px 28px;
    margin: 10px 0;
    text-align: center;
    animation: fadeUp .6s ease both;
}
.trade-arrow { font-size: 2.6rem; }
.trade-txt {
    font-size: 1.45rem; font-weight: 900;
    letter-spacing: .025em; line-height: 1.55; margin-top: 10px;
}

/* ── Step headers ─────────────────────────────────────────────────── */
.step-header {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.25rem; font-weight: 700; color: #c8d6e8;
    border-left: 3px solid #00b4ff;
    padding-left: 12px; margin: 28px 0 12px;
}
</style>
"""


def _model_icon(model_name: str) -> str:
    n = model_name.lower()
    if "gemini"   in n: return "✨"
    if "deepseek" in n: return "🔭"
    if "mixtral"  in n: return "⚗️"
    if "gemma"    in n: return "💎"
    if "qwen"     in n: return "🧠"
    if "vision"   in n: return "👁️"
    if "llama"    in n: return "🦙"
    return "🤖"


def badge(vote: str) -> str:
    c = VOTE_COLOR.get(vote, "gray")
    i = VOTE_ICON.get(vote, "❓")
    return f":{c}[**{i} {vote}**]"


def _vote_pill_html(vote: str, confidence: int = 0) -> str:
    color = _NEON.get(vote, "#888")
    icon  = VOTE_ICON.get(vote, "❓")
    conf  = f"<span class='conf-lbl'>{confidence}% conf</span>" if confidence else ""
    return (
        f"<span class='vote-pill' style='color:{color};border-color:{color};'>"
        f"{icon} {vote}</span>{conf}"
    )


def render_chart_analysis(g: dict, status_msg: str):
    vision = g.get("vision_model", "Vision Model")
    icon   = _model_icon(vision)

    if any(w in status_msg.lower() for w in ("fallback", "unavailable", "failed", "offline")):
        st.warning(status_msg)
    else:
        st.success(status_msg)

    with st.expander(f"{icon} {vision} — Chart Analysis + Market News", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset",     g.get("asset", "—"))
        c2.metric("Timeframe", g.get("timeframe", "—"))
        c3.metric("Trend",     g.get("trend", "—"))
        v = g.get("gemini_vote", "WAIT")
        c4.metric("Vision Vote",
                  f"{VOTE_ICON.get(v,'')} {v}",
                  f"{g.get('gemini_confidence', 0)}% confidence")

        st.info(f"**Technical Summary:** {g.get('technical_summary','')}")

        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Support**")
            for s in g.get("support", []): st.markdown(f"- `{s}`")
        with cb:
            st.markdown("**Resistance**")
            for r in g.get("resistance", []): st.markdown(f"- `{r}`")

        inds = g.get("indicators", {})
        if inds:
            st.markdown("**Indicators**")
            for k, val in inds.items(): st.markdown(f"- **{k}**: {val}")

        if g.get("patterns"):
            st.markdown("**Patterns:** " + " · ".join(f"`{p}`" for p in g["patterns"]))

        st.markdown("---")
        st.markdown("**📰 Market News**")
        for item in g.get("live_news", []):
            sent = item.get("sentiment", "Neutral")
            dot  = "🟢" if sent == "Bullish" else ("🔴" if sent == "Bearish" else "🟡")
            st.markdown(f"{dot} **{item.get('headline','')}**  \n*{item.get('source','')}*")
        st.markdown(f"**News Summary:** {g.get('news_summary','')}")
        st.markdown(f"💬 *{g.get('gemini_reasoning','')}*")


def render_vote_card(v: dict, status: str = "online"):
    model    = v.get("model", "Unknown")
    vote     = v.get("vote", "WAIT")
    icon     = _model_icon(model)
    conf     = v.get("confidence", 0)
    analysis = v.get("analysis", "")
    risks    = v.get("key_risks", [])
    reason   = v.get("reasoning", "")

    if status == "offline":
        err = str(v.get("error_msg", "This model did not respond."))[:220]
        st.markdown(
            f"<div class='ai-card offline'>"
            f"<div class='card-row'>"
            f"<span class='model-lbl'>🔴 {icon} {model} — OFFLINE / SKIPPED</span>"
            f"</div>"
            f"<p class='err-txt'>{err}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    color      = _NEON.get(vote, "#888")
    vote_icon  = VOTE_ICON.get(vote, "❓")
    risks_html = "".join(f"<span class='risk-tag'>{r}</span>" for r in risks)

    st.markdown(
        f"<div class='ai-card'>"
        f"<div class='card-row'>"
        f"<span class='model-lbl'>{icon} {model}</span>"
        f"<span class='vote-pill' style='color:{color};border-color:{color};'>"
        f"{vote_icon} {vote}</span>"
        f"<span class='conf-lbl'>{conf}% confidence</span>"
        f"</div>"
        f"<p class='analysis-txt'>{analysis}</p>"
        f"<div class='risks-row'>{risks_html}</div>"
        f"<p class='reasoning-txt'>💬 {reason}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_scoreboard(chart_data: dict, analyst_votes: list[dict], offline: list[str]):
    st.markdown("<div class='step-header'>🗳️ All AI Votes at a Glance</div>",
                unsafe_allow_html=True)

    vision_name   = chart_data.get("vision_model", "Vision")
    online_entries = [{
        "model": vision_name,
        "vote":  chart_data.get("gemini_vote", "WAIT"),
        "conf":  chart_data.get("gemini_confidence", 0),
        "icon":  _model_icon(vision_name),
    }] + [{
        "model": v.get("model", "?"),
        "vote":  v.get("vote", "WAIT"),
        "conf":  v.get("confidence", 0),
        "icon":  _model_icon(v.get("model", "")),
    } for v in analyst_votes]

    all_entries = online_entries + [
        {"model": name, "vote": "OFFLINE", "conf": 0, "icon": "🔴"}
        for name in offline
    ]

    cols = st.columns(max(len(all_entries), 1))
    for col, entry in zip(cols, all_entries):
        vote  = entry["vote"]
        color = _NEON.get(vote, "#555")
        icon  = VOTE_ICON.get(vote, "❌") if vote != "OFFLINE" else "❌"
        conf_txt = "offline" if vote == "OFFLINE" else f"{entry['conf']}% conf"
        col.markdown(
            f"<div class='sb-tile' style='border-color:{color}22;'>"
            f"<div class='sb-model-lbl'>{entry['icon']} {entry['model']}</div>"
            f"<div class='sb-vote-icon'>{icon}</div>"
            f"<div class='sb-vote-txt' style='color:{color};'>{vote}</div>"
            f"<div class='sb-conf-txt'>{conf_txt}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_final_decision(synth: dict):
    decision = synth.get("FINAL_DECISION", "WAIT")
    color    = _NEON.get(decision, "#888")
    arrow    = VOTE_ICON.get(decision, "❓")
    strength = synth.get("consensus_strength", "")
    tally    = synth.get("vote_tally", {})
    conf     = synth.get("confidence", 0)
    s_model  = synth.get("_synth_model", "Gemini")

    st.markdown("---")
    st.markdown(
        f"<div class='step-header'>🏆 Final Consensus"
        f"<span style='font-size:.8rem;font-weight:400;color:#566880;margin-left:8px;'>"
        f"moderated by {s_model}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='verdict-banner'>"
        f"<div class='verdict-word' style='color:{color};'>{arrow} {decision}</div>"
        f"<div class='verdict-sub'>"
        f"Consensus: <strong style='color:#c8d6e8;'>{strength}</strong>"
        f" &nbsp;·&nbsp; Confidence: <strong style='color:{color};'>{conf}%</strong>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("⬆️ UP",   tally.get("UP",   0))
    c2.metric("⬇️ DOWN", tally.get("DOWN", 0))
    c3.metric("⏸️ WAIT", tally.get("WAIT", 0))

    st.markdown(f"**Cross-Examination:** {synth.get('cross_examination','')}")
    st.info(f"📋 **Moderator Note:** {synth.get('moderator_note','')}")


def render_trade_recommendation(synth: dict):
    ra = synth.get("recommended_action", {})
    if not ra:
        return

    should_trade = ra.get("should_trade", False)
    display_text = ra.get("display_text", "")
    direction    = ra.get("trade_direction") or ""
    candles      = ra.get("candle_count")
    total_mins   = ra.get("total_duration_minutes")
    no_trade_rsn = ra.get("dont_trade_reason") or ""

    st.markdown("---")
    st.markdown("<div class='step-header'>🎯 Recommended Action</div>",
                unsafe_allow_html=True)

    if should_trade and direction in ("UP", "DOWN"):
        bg     = "#041a0a" if direction == "UP" else "#1a0408"
        txt    = _NEON["UP"]   if direction == "UP" else _NEON["DOWN"]
        border = "#00c853"     if direction == "UP" else "#d50000"
        arrow  = "⬆️"         if direction == "UP" else "⬇️"
        st.markdown(
            f"<div class='trade-box' style='background:{bg};border:2.5px solid {border};'>"
            f"<div class='trade-arrow'>{arrow}</div>"
            f"<div class='trade-txt' style='color:{txt};'>{display_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if candles and total_mins:
            m1, m2, m3 = st.columns(3)
            m1.metric("Direction",       f"{arrow} {direction}")
            m2.metric("Candles to Hold", str(candles))
            m3.metric("Est. Duration",   f"{total_mins} min")
    else:
        st.markdown(
            f"<div class='trade-box' style='background:#1a1200;border:2.5px solid #ff8f00;'>"
            f"<div class='trade-arrow'>🚫</div>"
            f"<div class='trade-txt' style='color:#ffd740;'>{display_text}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if no_trade_rsn and no_trade_rsn.lower() not in ("null", "none", ""):
            st.warning(f"⚠️ {no_trade_rsn}")


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

# Inject global CSS first
st.markdown(_PREMIUM_CSS, unsafe_allow_html=True)

# Hero header
st.markdown(
    "<h1 style='font-size:2.1rem;font-weight:900;letter-spacing:.01em;"
    "background:linear-gradient(90deg,#00b4ff,#00ff88);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;margin-bottom:0;'>📊 AI Trading Debate</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#566880;font-size:.88rem;margin-top:4px;'>"
    "Crash-proof &nbsp;·&nbsp; 5 live AI models &nbsp;·&nbsp; "
    "auto-fallback on any outage &nbsp;·&nbsp; candle-based trade target</p>",
    unsafe_allow_html=True,
)

st.markdown("<div class='step-header'>1️⃣ Upload Chart</div>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
    help="Screenshot of any trading chart",
)
if uploaded:
    st.image(Image.open(uploaded), caption="Uploaded chart", use_container_width=True)

st.markdown("<div class='step-header'>2️⃣ Extra Context <span style='font-weight:400;font-size:.8rem;color:#566880;'>(optional)</span></div>",
            unsafe_allow_html=True)
extra_ctx = st.text_area(
    "context",
    label_visibility="collapsed",
    placeholder="e.g. BTC/USDT 10-min chart, NY session open…",
    height=70,
)

st.markdown("<div class='step-header'>3️⃣ Start the Debate</div>", unsafe_allow_html=True)
run = st.button(
    "🚀  START AI DEBATE",
    disabled=(uploaded is None),
    use_container_width=True,
    type="primary",
)

if not run:
    st.stop()

# Read uploaded image bytes
uploaded.seek(0)
image_bytes = uploaded.read()
ext       = uploaded.name.rsplit(".", 1)[-1].lower()
mime_map  = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
             "png": "image/png",  "webp": "image/webp"}
mime_type = mime_map.get(ext, "image/jpeg")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Chart vision
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='step-header'>Step 1 — Chart Analysis & Market News</div>",
            unsafe_allow_html=True)
with st.spinner("🔍 Reading chart… Gemini primary, Groq auto-fallback…"):
    chart_data, vision_status = step1_analyze_chart(image_bytes, mime_type, extra_ctx)

render_chart_analysis(chart_data, vision_status)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Independent analyst votes
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div class='step-header'>Step 2 — Independent Analyst Votes</div>",
            unsafe_allow_html=True)

with st.spinner("⚡ Discovering active Groq text models…"):
    _groq_text_models = _discover_groq_text_models(n=3)

_groq_labels = ", ".join(_short_label(mid) for mid, _ in _groq_text_models)
st.markdown(
    f"<p style='font-size:.8rem;color:#566880;margin-bottom:12px;'>"
    f"🤖 Auto-selected: <strong style='color:#a0b8d0;'>{_groq_labels}</strong></p>",
    unsafe_allow_html=True,
)

ANALYST_MODELS: list[tuple[str, str | None, str]] = [
    ("groq", mid, _short_label(mid)) for mid, _ in _groq_text_models
] + [("deepseek", None, "DeepSeek-Chat")]

analyst_votes:  list[dict] = []
offline_models: list[str]  = []

for backend, model_id, model_name in ANALYST_MODELS:
    with st.spinner(f"{_model_icon(model_name)} {model_name} is analysing…"):
        try:
            result = (_call_groq(model_id, model_name, chart_data)
                      if backend == "groq" else _call_deepseek(chart_data))
            analyst_votes.append(result)
            render_vote_card(result, status="online")
        except Exception as exc:
            err_msg = str(exc)
            offline_models.append(model_name)
            render_vote_card({"model": model_name, "error_msg": err_msg}, status="offline")

if len(analyst_votes) == 0:
    st.error("All analyst models are currently offline. Please try again in a few minutes.")
    st.stop()

render_scoreboard(chart_data, analyst_votes, offline_models)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cross-examination & final verdict
# ══════════════════════════════════════════════════════════════════════════════
n_online = 1 + len(analyst_votes)
n_total  = 1 + len(ANALYST_MODELS)
st.markdown("---")
st.markdown(
    f"<div class='step-header'>Step 3 — Cross-Examination & Final Verdict"
    f"<span style='font-size:.8rem;font-weight:400;color:#566880;margin-left:8px;'>"
    f"{n_online}/{n_total} models online</span></div>",
    unsafe_allow_html=True,
)
if offline_models:
    st.markdown(
        f"<p style='font-size:.78rem;color:#566880;'>🔴 Offline: "
        f"{', '.join(offline_models)}</p>",
        unsafe_allow_html=True,
    )

with st.spinner("🧠 Cross-examining all opinions and computing final verdict…"):
    try:
        synthesis = step3_synthesize(chart_data, analyst_votes)
    except Exception as exc:
        st.error(f"**Final synthesis failed:** {exc}")
        st.stop()

render_final_decision(synthesis)
render_trade_recommendation(synthesis)
st.balloons()
