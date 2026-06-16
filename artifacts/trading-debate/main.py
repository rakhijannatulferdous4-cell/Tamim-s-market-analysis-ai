"""
AI Trading Debate — crash-proof, future-proof.

Pipeline
────────
Step 1  Chart Vision  : Gemini 2.0 Flash  (google-generativeai stable SDK)
                        → live Groq vision catalog  (seeded with known 2026 models)
                        → text-fallback if all vision APIs fail
Step 2  Analyst Votes : Top-3 Groq text models  (live auto-discovery)
                          explicit preference: llama-3.3-70b-versatile, qwen-2.5-32b
                        + DeepSeek-Chat (skipped gracefully on 402)
                        + any custom models added by the user in the sidebar
                        <think>…</think> tokens stripped before JSON parsing
Step 3  Final Verdict : Gemini 2.0 Flash synthesis → top Groq text model fallback
                        → FINAL_DECISION + candle recommendation box
"""

import base64
import io
import json
import os
import re

import requests
import streamlit as st
from PIL import Image

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
# JSON parser — strips <think> tokens, markdown fences, and bare objects
# ─────────────────────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict:
    raw = text or ""
    # Strip Qwen / DeepSeek-R1 chain-of-thought blocks
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
# Groq model discovery helpers
# ─────────────────────────────────────────────────────────────────────────────
_GROQ_TEXT_EXCLUDE = frozenset(
    ["vision", "whisper", "guard", "embed", "tts", "distil"]
)

# Hard fallback list used when the live API call fails
_GROQ_TEXT_FALLBACK = [
    ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
    ("qwen-2.5-32b",            "Qwen 2.5 32B"),
    ("llama-3.1-8b-instant",    "Llama 3.1 8B"),
]

# Seeded candidates for vision — newest first.
# Dynamic discovery appends anything else in the live catalog.
_GROQ_VISION_SEEDS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-90b-vision-preview",
    "llama-3.2-11b-vision-preview",
]


def _discover_groq_vision_models() -> list[str]:
    """
    Return an ordered list of vision-capable Groq model IDs.
    Seeds come first; the live catalog appends any additional ones.
    Never raises — returns seed list on any error.
    """
    try:
        from groq import Groq
        client = Groq(api_key=require_secret("GROQ_API_KEY"))
        listing = client.models.list()
        live_ids = {m.id for m in listing.data}
        # Start with seeds that exist in the live catalog
        queue = [s for s in _GROQ_VISION_SEEDS if s in live_ids]
        # Append any additional vision models from catalog not already queued
        for m in listing.data:
            if "vision" in m.id.lower() and m.id not in queue:
                queue.append(m.id)
        return queue if queue else _GROQ_VISION_SEEDS
    except Exception:
        return _GROQ_VISION_SEEDS


def _discover_groq_text_models(n: int = 3) -> list[tuple[str, str]]:
    """
    Return up to `n` text-only model IDs from Groq's live catalog,
    ranked by a quality heuristic. Explicit preferences always rank #1 and #2.
    Falls back to _GROQ_TEXT_FALLBACK if discovery fails or yields nothing.
    """
    _PREFERRED = {
        "llama-3.3-70b-versatile": 200,
        "qwen-2.5-32b":            190,
    }
    try:
        from groq import Groq
        client = Groq(api_key=require_secret("GROQ_API_KEY"))
        listing = client.models.list()

        candidates: list[tuple[int, str]] = []
        for m in listing.data:
            mid_lower = m.id.lower()
            if any(x in mid_lower for x in _GROQ_TEXT_EXCLUDE):
                continue
            if m.id in _PREFERRED:
                candidates.append((_PREFERRED[m.id], m.id))
                continue
            score = 0
            if "llama-3.3" in mid_lower:   score += 100
            elif "qwen"    in mid_lower:   score += 90
            elif "llama-3.2" in mid_lower: score += 80
            elif "llama-3.1" in mid_lower: score += 70
            elif "llama-3"  in mid_lower:  score += 60
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
        return result if result else _GROQ_TEXT_FALLBACK[:n]
    except Exception:
        return _GROQ_TEXT_FALLBACK[:n]


def _short_label(model_id: str) -> str:
    """Human-readable short label for any Groq model ID."""
    mid = model_id.lower()
    # Handle meta-llama/ prefix
    base = mid.split("/")[-1]
    for size in ["72b", "70b", "34b", "32b", "13b", "9b", "8b", "7b"]:
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
    "gemini_vote": "WAIT",
    "gemini_confidence": 0,
    "gemini_reasoning": "Vision analysis unavailable — apply strict risk management.",
}


def _gemini_vision(image_bytes: bytes, extra: str) -> dict:
    """
    Call Gemini 2.0 Flash via the stable google-generativeai SDK.
    Passes a PIL Image directly — no blob/Blob, no v1beta path issues.
    """
    import google.generativeai as genai

    genai.configure(api_key=require_secret("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    extra_line = (f"\n\nExtra context: {extra.strip()}") if extra.strip() else ""
    prompt = CHART_PROMPT_TEXT + extra_line + "\n" + CHART_ANALYSIS_JSON_SPEC

    img = Image.open(io.BytesIO(image_bytes))
    response = model.generate_content([img, prompt])
    result = parse_json(response.text)
    result["vision_model"] = "Gemini 2.0 Flash"
    return result


def _groq_vision(image_bytes: bytes, mime: str, extra: str,
                 model_id: str, model_label: str) -> dict:
    """
    Call a Groq vision model.
    Image is base64-encoded and sent as a data-URI inside the OpenAI-compatible
    image_url content block — the only format Groq accepts.
    """
    from groq import Groq

    client = Groq(api_key=require_secret("GROQ_API_KEY"))
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
    result.setdefault("gemini_vote",       result.get("vote", "WAIT"))
    result.setdefault("gemini_confidence", result.get("confidence", 50))
    result.setdefault("gemini_reasoning",  result.get("reasoning", "Groq vision fallback."))
    return result


def step1_analyze_chart(image_bytes: bytes, mime: str, extra: str) -> tuple[dict, str]:
    """
    Returns (chart_data, status_message). Never raises.

    Chain:
    1. Gemini 2.0 Flash (stable SDK) — single shot, instant fallback on ANY error.
    2. Groq vision — tries seeded candidates + live catalog discovery in order.
    3. Text fallback — debate still runs with high-volatility context.
    """
    # ── 1. Gemini — single attempt ────────────────────────────────────────────
    gemini_error = ""
    try:
        data = _gemini_vision(image_bytes, extra)
        return data, "✨ Gemini 2.0 Flash — chart analysis complete."
    except Exception as exc:
        gemini_error = f"{type(exc).__name__}: {str(exc)[:220]}"

    # ── 2. Groq vision — ordered queue (seeds + live discovery) ──────────────
    vision_queue = _discover_groq_vision_models()
    groq_errors: list[str] = []

    for model_id in vision_queue:
        try:
            data = _groq_vision(image_bytes, mime, extra, model_id, model_id)
            # Stamp Gemini's committee slot as Offline
            data.setdefault("gemini_vote",       "Offline")
            data.setdefault("gemini_confidence", 0)
            data.setdefault("gemini_reasoning",
                            f"Gemini temporarily offline — {gemini_error[:100]}")
            return data, (
                f"⚠️ Gemini offline ({gemini_error[:160]}). "
                f"Fell back to **{model_id}** (Groq vision) — analysis complete."
            )
        except Exception as exc:
            groq_errors.append(f"{model_id}: {type(exc).__name__}: {str(exc)[:80]}")

    # ── 3. Text fallback — debate continues regardless ────────────────────────
    all_errs = f"Gemini: {gemini_error} || Groq: {' | '.join(groq_errors)}"
    status = (
        "🚨 **All vision APIs failed** — running in text-fallback mode.\n\n"
        f"Errors: {all_errs[:350]}\n\n"
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
            "  • [" + n.get("sentiment", "?") + "] "
            + n.get("headline", "") + " — " + n.get("source", "")
            for n in g.get("live_news", [])
        )
        or "  No live news available."
    )
    indicators = (
        "; ".join(k + ": " + str(v) for k, v in g.get("indicators", {}).items())
        or "N/A"
    )
    return (
        "You are " + model_name + ", an expert AI trading analyst.\n\n"
        "Study the following chart analysis and market news, then give your own "
        "independent trading opinion.\n\n"
        "=== CHART ANALYSIS ===\n"
        "Vision Model : " + g.get("vision_model", "?") + "\n"
        "Asset        : " + g.get("asset", "?") + "\n"
        "Timeframe    : " + g.get("timeframe", "?") + "\n"
        "Price        : " + g.get("current_price", "?") + "\n"
        "Trend        : " + g.get("trend", "?") + "\n"
        "Support      : " + ", ".join(g.get("support", [])) + "\n"
        "Resistance   : " + ", ".join(g.get("resistance", [])) + "\n"
        "Indicators   : " + indicators + "\n"
        "Patterns     : " + ", ".join(g.get("patterns", [])) + "\n"
        "Summary      : " + g.get("technical_summary", "") + "\n"
        "Vision Vote  : " + str(g.get("gemini_vote", "?"))
        + " (" + str(g.get("gemini_confidence", 0)) + "%)\n"
        "Vision Says  : " + g.get("gemini_reasoning", "") + "\n\n"
        "=== MARKET NEWS ===\n"
        + news_lines + "\n"
        "Summary: " + g.get("news_summary", "") + "\n"
        "===================\n\n"
        "Return ONLY valid JSON — no markdown:\n"
        '{\n'
        '  "model": "' + model_name + '",\n'
        '  "analysis": "<2-3 sentence technical + fundamental view>",\n'
        '  "key_risks": ["<risk 1>", "<risk 2>"],\n'
        '  "vote": "<UP|DOWN|WAIT>",\n'
        '  "confidence": <0-100>,\n'
        '  "reasoning": "<one concise sentence>"\n'
        '}'
    )


def _call_groq(model_id: str, model_name: str, chart_data: dict) -> dict:
    from groq import Groq
    client = Groq(api_key=require_secret("GROQ_API_KEY"))
    prompt = build_analyst_prompt(model_name, chart_data)
    chat = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system",
             "content": "You are an expert AI trading analyst. Respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=700,
    )
    result = parse_json(chat.choices[0].message.content)
    result.setdefault("model", model_name)
    result["_platform"] = "Groq"
    return result


def _call_deepseek(chart_data: dict) -> dict:
    api_key = require_secret("DEEPSEEK_API_KEY")
    prompt  = build_analyst_prompt("DeepSeek-Chat", chart_data)
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system",
                 "content": "You are an expert AI trading analyst. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 700,
        },
        timeout=40,
    )
    if resp.status_code == 402:
        raise RuntimeError(
            "💳 DeepSeek is waiting for a top-up (402 Payment Required). "
            "Skipping — remaining analysts will carry the debate."
        )
    resp.raise_for_status()
    result = parse_json(resp.json()["choices"][0]["message"]["content"])
    result.setdefault("model", "DeepSeek-Chat")
    result["_platform"] = "DeepSeek"
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Synthesis & final verdict
# ─────────────────────────────────────────────────────────────────────────────
def _build_synthesis_prompt(chart_data: dict, analyst_votes: list[dict]) -> str:
    tf         = chart_data.get("timeframe", "unknown")
    tf_minutes = chart_data.get("timeframe_minutes", 0)

    positions = (
        "Vision Model (" + chart_data.get("vision_model", "?") + "): "
        + chart_data.get("gemini_vote", "WAIT")
        + " (" + str(chart_data.get("gemini_confidence", 0)) + "%) — "
        + chart_data.get("gemini_reasoning", "") + "\n"
    )
    for v in analyst_votes:
        positions += (
            v.get("model", "?") + ": " + v.get("vote", "WAIT")
            + " (" + str(v.get("confidence", 0)) + "%) — "
            + v.get("reasoning", "") + "\n"
            + "  Analysis  : " + v.get("analysis", "") + "\n"
            + "  Key Risks : " + ", ".join(v.get("key_risks", [])) + "\n"
        )

    online_names = [chart_data.get("vision_model", "Vision")] + [
        v.get("model", "?") for v in analyst_votes
    ]

    return (
        "You are the debate moderator for an AI trading analyst panel.\n\n"
        "Models online (" + str(len(online_names)) + "): "
        + ", ".join(online_names) + "\n\n"
        "=== ALL ANALYST POSITIONS ===\n"
        + positions
        + "==============================\n\n"
        "Tasks:\n"
        "1. Cross-examine — identify key agreements and disagreements.\n"
        "2. Decide FINAL_DECISION (UP / DOWN / WAIT) based on evidence weight.\n"
        "3. Compute RECOMMENDED_ACTION using the chart timeframe:\n"
        "   Timeframe: " + tf + " (" + str(tf_minutes) + " minutes per candle)\n"
        "   UP or DOWN: estimate candle_count (1-5), "
        "total_duration_minutes = candle_count x timeframe_minutes.\n"
        '     display_text: "TRADE DIRECTION: X | TARGET: Next N candles will go X '
        '(Duration: Y minutes on a Z-min chart)"\n'
        "   WAIT: should_trade = false, explain specific reason.\n"
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


def step3_synthesize(chart_data: dict, analyst_votes: list[dict]) -> dict:
    """
    Cross-examines all analyst opinions → FINAL_DECISION + candle recommendation.
    Primary: Gemini 2.0 Flash (stable SDK).  Fallback: top Groq text model.
    """
    import google.generativeai as genai

    prompt = _build_synthesis_prompt(chart_data, analyst_votes)

    synth_model = "Gemini 2.0 Flash"
    try:
        genai.configure(api_key=require_secret("GEMINI_API_KEY"))
        model    = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        result   = parse_json(response.text)
    except Exception as gemini_err:
        try:
            from groq import Groq
            fallback_models = _discover_groq_text_models(n=1)
            fallback_id     = fallback_models[0][0]
            client_g = Groq(api_key=require_secret("GROQ_API_KEY"))
            chat = client_g.chat.completions.create(
                model=fallback_id,
                messages=[
                    {"role": "system",
                     "content": "You are a trading debate moderator. Respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            result      = parse_json(chat.choices[0].message.content)
            synth_model = _short_label(fallback_id) + " (Groq fallback)"
        except Exception as groq_err:
            raise RuntimeError(
                "Both synthesis models failed.\n"
                "Gemini: " + str(gemini_err) + "\n"
                "Groq: "   + str(groq_err)
            )

    result["_synth_model"] = synth_model
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
/* ── Base dark theme ────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #080c17 !important;
    color: #c8d6e8;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"],
[data-testid="stToolbar"]    { background: transparent !important; }
[data-testid="stSidebar"]    { background: #0a0f1e !important;
                                border-right: 1px solid #1a2340; }
[data-testid="stMain"] > div { padding-top: 1rem; }

hr { border-color: #1a2340 !important; }

/* ── Metrics ────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #0d1220; border: 1px solid #1a2340;
    border-radius: 10px; padding: 10px 14px;
}

/* ── File uploader ──────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #0d1220; border: 2px dashed #1a3560;
    border-radius: 12px; padding: 8px;
}
[data-testid="stFileUploader"]:hover { border-color: #00b4ff; }

/* ── Text inputs & areas ────────────────────────────────────────────── */
textarea, input[type="text"] {
    background: #0d1220 !important; color: #c8d6e8 !important;
    border: 1px solid #1a2340 !important; border-radius: 10px !important;
}
textarea:focus, input[type="text"]:focus { border-color: #00b4ff !important; }

/* ── Expanders ──────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0d1220 !important;
    border: 1px solid #1a2340 !important;
    border-radius: 12px !important;
}

/* ── Primary button — electric-blue pulse ───────────────────────────── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0057ff, #00b4ff) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    letter-spacing: .04em !important;
    animation: btnPulse 2.4s ease-in-out infinite;
}
[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #0070ff, #33c6ff) !important;
    box-shadow: 0 0 22px #00b4ff88 !important; animation: none;
}
@keyframes btnPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,180,255,.55); }
    50%       { box-shadow: 0 0 0 14px rgba(0,180,255,0); }
}

/* ── Secondary buttons (sidebar) ────────────────────────────────────── */
[data-testid="baseButton-secondary"] {
    background: #12192e !important; color: #a0b8d0 !important;
    border: 1px solid #1e3050 !important; border-radius: 8px !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #00b4ff !important; color: #00b4ff !important;
}

/* ── Analyst cards — fade-in ────────────────────────────────────────── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.ai-card {
    background: #0d1220; border: 1px solid #1a2340;
    border-radius: 14px; padding: 18px 20px 14px; margin: 10px 0;
    animation: fadeUp .55s ease both;
    box-shadow: 0 4px 24px rgba(0,0,0,.45);
}
.ai-card:hover { border-color: #00b4ff44;
                  box-shadow: 0 4px 28px rgba(0,180,255,.12); }
.ai-card.offline { border-color: #ff386035; opacity: .75; }

.card-row { display:flex; align-items:center; gap:10px;
            flex-wrap:wrap; margin-bottom:10px; }
.model-lbl  { font-size:.95rem; font-weight:700; color:#e0eaf8; }
.vote-pill  { font-size:.8rem; font-weight:800; padding:3px 11px;
              border-radius:20px; border:1.5px solid; letter-spacing:.05em; }
.conf-lbl   { font-size:.78rem; color:#6a82a0; margin-left:auto; }
.analysis-txt { font-size:.88rem; color:#a8bcd4; line-height:1.6; margin:6px 0; }
.risks-row  { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
.risk-tag   { font-size:.74rem; padding:3px 9px; border-radius:6px;
              background:#12192e; border:1px solid #1e2f50; color:#7a9abf; }
.reasoning-txt { font-size:.83rem; color:#5e7a9a; font-style:italic; margin-top:6px; }
.err-txt    { font-size:.82rem; color:#ff6080; margin-top:6px; }

/* ── Scoreboard tiles ───────────────────────────────────────────────── */
.sb-tile {
    background: #0d1220; border: 1.5px solid #1a2340;
    border-radius: 10px; padding: 10px 6px; text-align: center;
    animation: fadeUp .5s ease both;
}
.sb-tile:hover { border-color: #00b4ff44; }
.sb-model-lbl { font-size:.65rem; color:#566880; margin-bottom:4px; }
.sb-vote-icon { font-size:1.4rem; }
.sb-vote-txt  { font-size:.88rem; font-weight:800; }
.sb-conf-txt  { font-size:.65rem; color:#566880; margin-top:2px; }

/* ── Final verdict banner ───────────────────────────────────────────── */
.verdict-banner { text-align:center; padding:28px 0 12px;
                   animation: fadeUp .6s ease both; }
.verdict-word { font-size:4.8rem; font-weight:900; line-height:1; }
.verdict-sub  { font-size:1rem; color:#6a82a0; margin-top:6px; }

/* ── Trade boxes ────────────────────────────────────────────────────── */
.trade-box { border-radius:16px; padding:32px 28px; margin:10px 0;
             text-align:center; animation: fadeUp .6s ease both; }
.trade-arrow { font-size:2.6rem; }
.trade-txt   { font-size:1.45rem; font-weight:900;
               letter-spacing:.025em; line-height:1.55; margin-top:10px; }

/* ── Step headers ───────────────────────────────────────────────────── */
.step-header {
    display:flex; align-items:center; gap:10px;
    font-size:1.2rem; font-weight:700; color:#c8d6e8;
    border-left:3px solid #00b4ff; padding-left:12px; margin:24px 0 10px;
}

/* ── Platform badges (Groq / DeepSeek / Gemini) ─────────────────────── */
.platform-badge {
    font-size:.65rem; font-weight:800; padding:2px 7px;
    border-radius:5px; letter-spacing:.06em; vertical-align:middle;
}
.platform-groq    { background:#1a2e10; color:#6ee73a; border:1px solid #4aaa1a; }
.platform-deepseek{ background:#0d1f3c; color:#60a5fa; border:1px solid #3b82f6; }
.platform-gemini  { background:#1e0d38; color:#c084fc; border:1px solid #9333ea; }
.platform-custom  { background:#1a1a2e; color:#94a3b8; border:1px solid #475569; }

/* ── Sidebar model tags ─────────────────────────────────────────────── */
.custom-model-tag {
    display:inline-flex; align-items:center; gap:6px;
    background:#12192e; border:1px solid #1e3050; border-radius:8px;
    padding:4px 10px; margin:4px 0; font-size:.8rem; color:#7a9abf;
    width:100%;
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
    icon   = _model_icon(vision)

    is_warn = any(w in status_msg.lower()
                  for w in ("fallback", "unavailable", "failed", "offline"))
    if is_warn:
        st.warning(status_msg)
    else:
        st.success(status_msg)

    with st.expander(f"{icon} {vision} — Chart Analysis + Market News", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset",     g.get("asset",     "—"))
        c2.metric("Timeframe", g.get("timeframe", "—"))
        c3.metric("Trend",     g.get("trend",     "—"))
        v = g.get("gemini_vote", "WAIT")
        c4.metric("Vision Vote",
                  (VOTE_ICON.get(v, "") + " " + v).strip(),
                  str(g.get("gemini_confidence", 0)) + "% confidence")

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
        st.markdown("**📰 Market News**")
        for item in g.get("live_news", []):
            sent = item.get("sentiment", "Neutral")
            dot  = "🟢" if sent == "Bullish" else ("🔴" if sent == "Bearish" else "🟡")
            st.markdown(dot + " **" + item.get("headline", "") + "**  \n*" + item.get("source", "") + "*")
        st.markdown("**News Summary:** " + g.get("news_summary", ""))
        st.markdown("💬 *" + g.get("gemini_reasoning", "") + "*")


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
        err = str(v.get("error_msg", "This model did not respond."))[:240]
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

    vision_name = chart_data.get("vision_model", "Vision")
    vision_platform = "Gemini" if "gemini" in vision_name.lower() else "Groq"

    entries = [{
        "model":    vision_name,
        "vote":     chart_data.get("gemini_vote", "WAIT"),
        "conf":     chart_data.get("gemini_confidence", 0),
        "icon":     _model_icon(vision_name),
        "platform": vision_platform,
    }] + [{
        "model":    v.get("model", "?"),
        "vote":     v.get("vote",  "WAIT"),
        "conf":     v.get("confidence", 0),
        "icon":     _model_icon(v.get("model", "")),
        "platform": v.get("_platform", ""),
    } for v in analyst_votes] + [
        {"model": nm, "vote": "OFFLINE", "conf": 0, "icon": "🔴", "platform": ""}
        for nm in offline
    ]

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


def render_trade_recommendation(synth: dict) -> None:
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
        txt    = _NEON["UP"]  if direction == "UP" else _NEON["DOWN"]
        border = "#00c853"    if direction == "UP" else "#d50000"
        arrow  = "⬆️"        if direction == "UP" else "⬇️"
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
# Sidebar — Custom AI Model Manager
# ─────────────────────────────────────────────────────────────────────────────
if "custom_models" not in st.session_state:
    st.session_state["custom_models"] = []

with st.sidebar:
    st.markdown(
        "<h2 style='font-size:1.1rem;font-weight:800;color:#c8d6e8;"
        "border-bottom:1px solid #1a2340;padding-bottom:8px;margin-bottom:12px;'>"
        "🧩 Custom Model Manager</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:.8rem;color:#566880;margin-bottom:10px;'>"
        "Add any live Groq model ID to include it in the voting committee.</p>",
        unsafe_allow_html=True,
    )

    custom_input = st.text_input(
        "Groq Model ID",
        key="custom_model_input",
        placeholder="e.g. qwen-qwq-32b",
        label_visibility="collapsed",
    )
    if st.button("➕ Add to Committee", use_container_width=True):
        cid = (custom_input or "").strip()
        if cid and cid not in st.session_state["custom_models"]:
            st.session_state["custom_models"].append(cid)
            st.rerun()

    if st.session_state["custom_models"]:
        st.markdown(
            "<p style='font-size:.75rem;color:#566880;margin:10px 0 4px;'>"
            "Active custom analysts:</p>",
            unsafe_allow_html=True,
        )
        for i, mid in enumerate(list(st.session_state["custom_models"])):
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                "<div class='custom-model-tag'>🤖 " + mid + "</div>",
                unsafe_allow_html=True,
            )
            if c2.button("✕", key="rm_" + str(i), help="Remove"):
                st.session_state["custom_models"].pop(i)
                st.rerun()
    else:
        st.markdown(
            "<p style='font-size:.78rem;color:#2a3a50;font-style:italic;margin-top:8px;'>"
            "No custom models yet.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:.72rem;color:#2a3a50;'>"
        "Tip: paste an exact model ID from Groq's console.<br>"
        "Examples: <code>qwen-qwq-32b</code>, "
        "<code>meta-llama/llama-4-scout-17b-16e-instruct</code></p>",
        unsafe_allow_html=True,
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
run = st.button(
    "🚀  START AI DEBATE",
    disabled=(uploaded is None),
    use_container_width=True,
    type="primary",
)

if not run:
    st.stop()

# ── Read image bytes ──────────────────────────────────────────────────────────
uploaded.seek(0)
image_bytes = uploaded.read()
ext      = uploaded.name.rsplit(".", 1)[-1].lower()
mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp"}
mime_type = mime_map.get(ext, "image/jpeg")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Chart vision
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='step-header'>Step 1 — Chart Analysis & Market News</div>",
            unsafe_allow_html=True)
with st.spinner("🔍 Reading chart…  Gemini primary → Groq vision fallback…"):
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
_custom_count = len(st.session_state["custom_models"])
_extra_note   = (f" + {_custom_count} custom" if _custom_count else "")
st.markdown(
    "<p style='font-size:.8rem;color:#566880;margin-bottom:12px;'>"
    "🤖 Auto-selected: <strong style='color:#a0b8d0;'>" + _groq_labels + "</strong>"
    + _extra_note + "</p>",
    unsafe_allow_html=True,
)

# Build the analyst roster: auto-discovered + DeepSeek + user-added custom models
ANALYST_MODELS: list[tuple[str, str, str]] = (
    [("groq", mid, _short_label(mid)) for mid, _ in _groq_text_models]
    + [("deepseek", "", "DeepSeek-Chat")]
    + [("groq", mid, mid) for mid in st.session_state["custom_models"]]
)

analyst_votes:  list[dict] = []
offline_models: list[str]  = []

for backend, model_id, model_name in ANALYST_MODELS:
    with st.spinner(_model_icon(model_name) + " " + model_name + " is analysing…"):
        try:
            if backend == "groq":
                result = _call_groq(model_id, model_name, chart_data)
            else:
                result = _call_deepseek(chart_data)
            analyst_votes.append(result)
            render_vote_card(result, status="online")
        except Exception as exc:
            err_msg = str(exc)
            offline_models.append(model_name)
            render_vote_card({"model": model_name, "error_msg": err_msg}, status="offline")

if not analyst_votes:
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
    "<div class='step-header'>Step 3 — Cross-Examination & Final Verdict"
    "<span style='font-size:.8rem;font-weight:400;color:#566880;margin-left:8px;'>"
    + str(n_online) + "/" + str(n_total) + " models online</span></div>",
    unsafe_allow_html=True,
)
if offline_models:
    st.markdown(
        "<p style='font-size:.78rem;color:#566880;'>🔴 Offline this run: "
        + ", ".join(offline_models) + "</p>",
        unsafe_allow_html=True,
    )

with st.spinner("🧠 Cross-examining all opinions and computing final verdict…"):
    try:
        synthesis = step3_synthesize(chart_data, analyst_votes)
    except Exception as exc:
        st.error("**Final synthesis failed:** " + str(exc))
        st.stop()

render_final_decision(synthesis)
render_trade_recommendation(synthesis)
st.balloons()
