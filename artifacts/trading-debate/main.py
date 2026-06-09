"""
AI Trading Debate — bulletproof, crash-proof architecture.

Pipeline
────────
Step 1  Chart Vision  : Gemini 2.5 Flash (primary)
                        → fallback: Groq llama-3.2-11b-vision-preview
Step 2  Analyst Votes : Llama 3 70B / Mixtral 8x7B / Gemma 2 9B (Groq)
                        + DeepSeek-Chat — each fully independent
Step 3  Final Verdict : Gemini (or Groq fallback) synthesises all online models
                        → FINAL_DECISION + candle recommendation
"""

import base64
import json
import os
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
# Secret loader — st.secrets first, then Replit env var
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
# JSON extractor — tolerant of markdown wrappers
# ─────────────────────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict:
    raw = text or ""
    # Strip reasoning tokens emitted by thinking models (Qwen3, DeepSeek-R1, etc.)
    # Everything inside <think>…</think> is internal monologue, not output JSON.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Strip markdown code fences
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start < 0:
        raise ValueError(f"No JSON object in response:\n{text[:400]}")
    return json.loads(raw[start:end])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Chart vision: Gemini primary → Groq vision fallback
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
    "(use search if available, otherwise use your knowledge).\n\n"
    "Return ONLY valid JSON — no markdown, no extra text:\n"
)


def _gemini_analyze(image_bytes: bytes, mime: str, extra: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_secret("GEMINI_API_KEY"))

    extra_line = (f"\n\nExtra context: {extra.strip()}") if extra.strip() else ""
    prompt     = CHART_PROMPT_PREFIX + extra_line + "\n" + CHART_ANALYSIS_JSON_SPEC

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(inline_data=types.Blob(mime_type=mime, data=image_bytes)),
                types.Part(text=prompt),
            ],
        )
    ]
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )
    except Exception:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
    result = parse_json(resp.text)
    result["vision_model"] = "Gemini 2.5 Flash"
    return result


# ── Dynamic Groq vision model discovery ──────────────────────────────────────
# No hardcoded model names. At runtime we ask Groq for its live model list and
# pick the first model whose ID contains "vision". When Groq renames or adds
# models the code automatically picks up the new name — no code changes needed.

def _discover_groq_vision_models() -> list[tuple[str, str]]:
    """
    Query Groq's live /models endpoint and return all vision-capable model IDs.
    Falls back to an empty list if the API call fails.
    """
    try:
        from groq import Groq
        client  = Groq(api_key=require_secret("GROQ_API_KEY"))
        listing = client.models.list()
        found   = [
            (m.id, m.id)
            for m in listing.data
            if "vision" in m.id.lower()
        ]
        return found
    except Exception:
        return []


# Keywords that disqualify a model from the text-analyst pool
_GROQ_TEXT_EXCLUDE = frozenset(
    ["vision", "whisper", "guard", "embed", "tts", "preview", "distil"]
)

# Hardcoded safe fallback if the live API call fails
_GROQ_TEXT_FALLBACK = [
    ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
    ("mixtral-8x7b-32768",      "Mixtral 8x7B"),
    ("llama-3.1-8b-instant",    "Llama 3.1 8B"),
]


def _discover_groq_text_models(n: int = 3) -> list[tuple[str, str]]:
    """
    Query Groq's live /models endpoint and return up to `n` text-only models
    ranked by quality heuristic (family + parameter count).

    Excludes vision, audio, guard, and embed models so only pure text
    chat models are returned.  Falls back to _GROQ_TEXT_FALLBACK if the
    API call fails or returns no usable models.
    """
    try:
        from groq import Groq
        client  = Groq(api_key=require_secret("GROQ_API_KEY"))
        listing = client.models.list()

        candidates: list[tuple[int, str]] = []
        for m in listing.data:
            mid = m.id.lower()
            if any(x in mid for x in _GROQ_TEXT_EXCLUDE):
                continue
            # Score by model family (higher = more capable / newer)
            score = 0
            if "llama-3.3"     in mid: score += 100
            elif "llama-3.2"   in mid: score += 80
            elif "llama-3.1"   in mid: score += 70
            elif "llama-3"     in mid: score += 60
            elif "mixtral"     in mid: score += 55
            elif "gemma"       in mid: score += 50
            elif "qwen"        in mid: score += 45
            else:                      score += 10
            # Bonus for larger parameter counts
            if   any(x in mid for x in ["70b", "72b"]): score += 30
            elif any(x in mid for x in ["34b", "13b"]): score += 20
            elif "8b"  in mid:                           score += 15
            elif "7b"  in mid:                           score += 12
            candidates.append((score, m.id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        result = [(mid, mid) for _, mid in candidates[:n]]
        return result if result else _GROQ_TEXT_FALLBACK[:n]
    except Exception:
        return _GROQ_TEXT_FALLBACK[:n]


def _short_label(model_id: str) -> str:
    """Return a human-readable short label for any Groq model ID."""
    mid = model_id.lower()
    # Extract parameter size if present
    for size in ["70b", "72b", "34b", "13b", "8b", "7b", "9b"]:
        if size in mid:
            # Find family
            if "llama-3.3" in mid: return f"Llama 3.3 {size.upper()}"
            if "llama-3.2" in mid: return f"Llama 3.2 {size.upper()}"
            if "llama-3.1" in mid: return f"Llama 3.1 {size.upper()}"
            if "llama-3"   in mid: return f"Llama 3 {size.upper()}"
            if "mixtral"   in mid: return f"Mixtral {size.upper()}"
            if "gemma"     in mid: return f"Gemma {size.upper()}"
            if "qwen"      in mid: return f"Qwen {size.upper()}"
    # Fallback: capitalise the raw ID
    return model_id.split("/")[-1].replace("-", " ").title()


def _groq_vision_analyze(image_bytes: bytes, mime: str, extra: str,
                          model_id: str, model_label: str) -> dict:
    from groq import Groq

    client     = Groq(api_key=require_secret("GROQ_API_KEY"))
    b64_image  = base64.b64encode(image_bytes).decode()
    extra_line = (f"\n\nExtra context: {extra.strip()}") if extra.strip() else ""
    prompt     = (
        CHART_PROMPT_PREFIX + extra_line + "\n" + CHART_ANALYSIS_JSON_SPEC
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
    result.setdefault("gemini_vote",       result.get("vote", "WAIT"))
    result.setdefault("gemini_confidence", result.get("confidence", 50))
    result.setdefault("gemini_reasoning",  result.get("reasoning", "Fallback vision model."))
    return result


# ── Safe text fallback used when ALL vision APIs are unavailable ──────────────
_TEXT_FALLBACK_DATA: dict = {
    "asset":            "Unknown (vision APIs unavailable)",
    "timeframe":        "Unknown",
    "timeframe_minutes": 5,
    "current_price":    "Unknown",
    "trend":            "Unknown — treat as highly volatile",
    "support":          [],
    "resistance":       [],
    "indicators":       {},
    "patterns":         [],
    "technical_summary": (
        "Chart image could not be analysed — all vision APIs are currently "
        "unavailable. The AI committee will debate using strict risk-management "
        "assumptions: highly volatile market, no confirmed trend."
    ),
    "live_news": [],
    "news_summary": "No live news available. Assume high uncertainty.",
    "vision_model":       "Text Fallback (no vision API available)",
    "gemini_vote":        "WAIT",
    "gemini_confidence":  0,
    "gemini_reasoning":   "Vision analysis unavailable — all analysts should apply strict risk management.",
}


def step1_analyze_chart(image_bytes: bytes, mime: str, extra: str) -> tuple[dict, str]:
    """
    Returns (chart_data, status_message).  Never raises — always returns data.

    Strategy
    ────────
    1. Try Gemini 2.5 Flash up to 3 times (3-second pause between attempts).
       503 high-demand spikes usually clear within one retry.
    2. If all Gemini attempts fail, call Groq's live /models API to discover
       every currently active vision model dynamically, then try each one in
       turn until one succeeds.  No hardcoded model names — future-proof.
    3. If every vision option fails, return a safe text-only fallback so the
       debate (Steps 2-3) can still run with full risk-management context.
    """
    GEMINI_RETRIES = 3
    GEMINI_DELAY   = 3  # seconds between retries

    # ── 1. Gemini with retries ────────────────────────────────────────────────
    gemini_errors: list[str] = []
    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            data   = _gemini_analyze(image_bytes, mime, extra)
            suffix = f" (attempt {attempt}/{GEMINI_RETRIES})" if attempt > 1 else ""
            return data, f"✨ Gemini 2.5 Flash — chart analysis complete{suffix}."
        except Exception as exc:
            gemini_errors.append(
                f"Attempt {attempt}: {type(exc).__name__}: {str(exc)[:140]}"
            )
            if attempt < GEMINI_RETRIES:
                time.sleep(GEMINI_DELAY)

    # ── 2. Dynamic Groq vision fallback ──────────────────────────────────────
    groq_vision_models = _discover_groq_vision_models()
    groq_errors: list[str] = []

    if groq_vision_models:
        for model_id, model_label in groq_vision_models:
            try:
                data = _groq_vision_analyze(
                    image_bytes, mime, extra, model_id, model_label
                )
                gemini_summary = " | ".join(gemini_errors)
                return data, (
                    f"⚠️ Gemini failed after {GEMINI_RETRIES} attempts "
                    f"({gemini_summary[:180]}). "
                    f"Auto-detected and used **{model_label}** (Groq) — "
                    "analysis complete."
                )
            except Exception as exc:
                groq_errors.append(
                    f"{model_label}: {type(exc).__name__}: {str(exc)[:140]}"
                )
    else:
        groq_errors.append("No vision-capable models found in Groq model list.")

    # ── 3. Text fallback — never stop execution ───────────────────────────────
    all_errors = (
        "Gemini: " + " | ".join(gemini_errors)
        + " || Groq: " + " | ".join(groq_errors)
    )
    status = (
        "🚨 **All vision APIs failed** — running in text-fallback mode.\n\n"
        f"Errors: {all_errors[:300]}\n\n"
        "The AI committee will debate using a **highly volatile / high-risk** "
        "context. Chart-specific signals will not be available."
    )
    fallback = dict(_TEXT_FALLBACK_DATA)
    if extra.strip():
        fallback["technical_summary"] += f" User context: {extra.strip()}"
    return fallback, status


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Independent analyst votes
# ─────────────────────────────────────────────────────────────────────────────
def build_analyst_prompt(model_name: str, g: dict) -> str:
    news_lines = "\n".join(
        f"  • [{n.get('sentiment','?')}] {n.get('headline','')} — {n.get('source','')}"
        for n in g.get("live_news", [])
    ) or "  No live news available."

    indicators = "; ".join(
        f"{k}: {v}" for k, v in g.get("indicators", {}).items()
    ) or "N/A"

    return (
        f"You are {model_name}, an expert AI trading analyst.\n\n"
        "Study the following chart analysis and market news, then give your own "
        "independent trading opinion.\n\n"
        "=== CHART ANALYSIS ===\n"
        f"Vision Model : {g.get('vision_model','?')}\n"
        f"Asset        : {g.get('asset','?')}\n"
        f"Timeframe    : {g.get('timeframe','?')}\n"
        f"Price        : {g.get('current_price','?')}\n"
        f"Trend        : {g.get('trend','?')}\n"
        f"Support      : {', '.join(g.get('support', []))}\n"
        f"Resistance   : {', '.join(g.get('resistance', []))}\n"
        f"Indicators   : {indicators}\n"
        f"Patterns     : {', '.join(g.get('patterns', []))}\n"
        f"Summary      : {g.get('technical_summary','')}\n"
        f"Vision Vote  : {g.get('gemini_vote','?')} ({g.get('gemini_confidence',0)}%)\n"
        f"Vision Says  : {g.get('gemini_reasoning','')}\n\n"
        "=== MARKET NEWS ===\n"
        f"{news_lines}\n"
        f"Summary: {g.get('news_summary','')}\n"
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


def _call_groq(model_id: str, model_name: str, gemini_data: dict) -> dict:
    from groq import Groq
    client = Groq(api_key=require_secret("GROQ_API_KEY"))
    prompt = build_analyst_prompt(model_name, gemini_data)
    chat   = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert AI trading analyst. Respond with valid JSON only."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=600,
    )
    result = parse_json(chat.choices[0].message.content)
    result.setdefault("model", model_name)
    return result


def _call_deepseek(gemini_data: dict) -> dict:
    api_key = require_secret("DEEPSEEK_API_KEY")
    prompt  = build_analyst_prompt("DeepSeek-Chat", gemini_data)
    resp    = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an expert AI trading analyst. Respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 600,
        },
        timeout=40,
    )
    if resp.status_code == 402:
        raise RuntimeError(
            "💳 DeepSeek is waiting for a top-up (402 Payment Required). "
            "Skipping — the remaining analysts will carry the debate."
        )
    resp.raise_for_status()
    result = parse_json(resp.json()["choices"][0]["message"]["content"])
    result.setdefault("model", "DeepSeek-Chat")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Dynamic synthesis (works with any ≥1 online model)
# ─────────────────────────────────────────────────────────────────────────────
def step3_synthesize(chart_data: dict, analyst_votes: list[dict]) -> dict:
    """
    Sends all successful analyst opinions to Gemini (or Groq as fallback)
    for cross-examination and final verdict.
    """
    tf         = chart_data.get("timeframe", "unknown")
    tf_minutes = chart_data.get("timeframe_minutes", 0)

    # Build positions text without .format() to avoid { } injection issues
    positions  = f"Vision Model ({chart_data.get('vision_model','?')}): "
    positions += f"{chart_data.get('gemini_vote','WAIT')} "
    positions += f"({chart_data.get('gemini_confidence',0)}%) — "
    positions += chart_data.get("gemini_reasoning", "") + "\n"
    for v in analyst_votes:
        positions += (
            f"{v.get('model','?')}: {v.get('vote','WAIT')} "
            f"({v.get('confidence',0)}%) — {v.get('reasoning','')}\n"
            f"  Analysis  : {v.get('analysis','')}\n"
            f"  Key Risks : {', '.join(v.get('key_risks', []))}\n"
        )

    online_names = (
        [chart_data.get("vision_model", "Vision")] +
        [v.get("model", "?") for v in analyst_votes]
    )
    n_online = len(online_names)

    prompt = (
        "You are the debate moderator for an AI trading analyst panel.\n\n"
        f"Models that responded ({n_online} online): {', '.join(online_names)}\n\n"
        "=== ALL ANALYST POSITIONS ===\n"
        + positions +
        "==============================\n\n"
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

    # Try Gemini first, fall back to Groq llama-3 70b
    def _try_gemini(p: str) -> dict:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=require_secret("GEMINI_API_KEY"))
        resp   = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Content(role="user", parts=[types.Part(text=p)])],
        )
        return parse_json(resp.text)

    def _try_groq_text(p: str) -> dict:
        from groq import Groq
        client = Groq(api_key=require_secret("GROQ_API_KEY"))
        chat   = client.chat.completions.create(
            model=_GROQ_TEXT_FALLBACK[0][0],
            messages=[
                {"role": "system", "content": "You are a trading debate moderator. Respond with valid JSON only."},
                {"role": "user",   "content": p},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return parse_json(chat.choices[0].message.content)

    synth_model = "Gemini 2.5 Flash"
    try:
        result = _try_gemini(prompt)
    except Exception as e1:
        try:
            result = _try_groq_text(prompt)
            synth_model = "Llama 3 70B (Groq fallback)"
        except Exception as e2:
            raise RuntimeError(
                f"Both synthesis models failed.\nGemini: {e1}\nGroq: {e2}"
            )
    result["_synth_model"] = synth_model
    return result


# ─────────────────────────────────────────────────────────────────────────────
# UI constants & helpers
# ─────────────────────────────────────────────────────────────────────────────
VOTE_COLOR = {"UP": "green", "DOWN": "red", "WAIT": "orange"}
VOTE_ICON  = {"UP": "⬆️",   "DOWN": "⬇️", "WAIT": "⏸️"}
MODEL_ICON = {
    "Gemini 2.5 Flash":              "✨",
    "Llama 3.2 11B Vision (Groq)":   "👁️",
    "Llama 3 70B":                   "🦙",
    "Mixtral 8x7B":                  "⚗️",
    "Gemma 2 9B":                    "💎",
    "DeepSeek-Chat":                 "🔭",
}


def badge(vote: str) -> str:
    c = VOTE_COLOR.get(vote, "gray")
    i = VOTE_ICON.get(vote, "❓")
    return f":{c}[**{i} {vote}**]"


def render_chart_analysis(g: dict, status_msg: str):
    vision = g.get("vision_model", "Vision Model")
    icon   = MODEL_ICON.get(vision, "🔍")
    if "fallback" in status_msg.lower() or "unavailable" in status_msg.lower():
        st.warning(status_msg)
    else:
        st.success(status_msg)

    with st.expander(f"{icon} {vision} — Chart Analysis + Market News", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset",     g.get("asset", "—"))
        c2.metric("Timeframe", g.get("timeframe", "—"))
        c3.metric("Trend",     g.get("trend", "—"))
        v = g.get("gemini_vote", "WAIT")
        c4.metric("Vision Vote", f"{VOTE_ICON.get(v,'')} {v}",
                  f"{g.get('gemini_confidence', 0)}% confidence")

        st.info(f"**Technical Summary:** {g.get('technical_summary','')}")

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
            st.markdown(
                f"{dot} **{item.get('headline','')}**  \n*{item.get('source','')}*"
            )
        st.markdown(f"**News Summary:** {g.get('news_summary','')}")
        st.markdown(f"💬 *{g.get('gemini_reasoning','')}*")


def render_vote_card(v: dict, status: str = "online"):
    model = v.get("model", "Unknown")
    vote  = v.get("vote", "WAIT")
    icon  = MODEL_ICON.get(model, "🤖")

    if status == "offline":
        with st.expander(f"🔴 **{model}** — OFFLINE / ERROR", expanded=False):
            st.error(v.get("error_msg", "This model did not respond."))
        return

    with st.expander(
        f"{icon} **{model}** — {badge(vote)} — {v.get('confidence', 0)}% confidence",
        expanded=True,
    ):
        st.markdown(f"**Analysis:** {v.get('analysis','')}")
        risks = v.get("key_risks", [])
        if risks:
            st.markdown("**Key Risks:** " + " · ".join(f"`{r}`" for r in risks))
        st.markdown(f"💬 *{v.get('reasoning','')}*")


def render_scoreboard(chart_data: dict, analyst_votes: list[dict], offline: list[str]):
    st.markdown("### 🗳️ All AI Votes at a Glance")

    online_entries = [
        {
            "model": chart_data.get("vision_model", "Vision"),
            "vote":  chart_data.get("gemini_vote", "WAIT"),
            "conf":  chart_data.get("gemini_confidence", 0),
            "icon":  MODEL_ICON.get(chart_data.get("vision_model",""), "🔍"),
        }
    ] + [
        {
            "model": v.get("model","?"),
            "vote":  v.get("vote","WAIT"),
            "conf":  v.get("confidence",0),
            "icon":  MODEL_ICON.get(v.get("model",""), "🤖"),
        }
        for v in analyst_votes
    ]

    all_entries = online_entries + [
        {"model": name, "vote": "OFFLINE", "conf": 0, "icon": "🔴"}
        for name in offline
    ]

    cols = st.columns(max(len(all_entries), 1))
    for col, entry in zip(cols, all_entries):
        vote  = entry["vote"]
        color = VOTE_COLOR.get(vote, "#888") if vote != "OFFLINE" else "#888"
        icon  = VOTE_ICON.get(vote, "❌") if vote != "OFFLINE" else "❌"
        col.markdown(
            f"<div style='text-align:center;padding:10px 4px;"
            f"border:1px solid #444;border-radius:8px;'>"
            f"<div style='font-size:0.7rem;color:#aaa;margin-bottom:2px;'>"
            f"{entry['icon']} {entry['model']}</div>"
            f"<div style='font-size:1.5rem;'>{icon}</div>"
            f"<div style='font-size:.95rem;font-weight:800;color:{color};'>{vote}</div>"
            f"<div style='font-size:0.7rem;color:#aaa;'>"
            f"{'offline' if vote=='OFFLINE' else str(entry['conf'])+'% conf'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_final_decision(synth: dict):
    decision = synth.get("FINAL_DECISION", "WAIT")
    color    = VOTE_COLOR.get(decision, "gray")
    arrow    = VOTE_ICON.get(decision, "❓")
    strength = synth.get("consensus_strength", "")
    tally    = synth.get("vote_tally", {})
    conf     = synth.get("confidence", 0)
    s_model  = synth.get("_synth_model", "Gemini")

    st.markdown("---")
    st.markdown(f"## 🏆 Final Consensus  *(moderated by {s_model})*")

    st.markdown(
        f"<div style='text-align:center;padding:18px 0 4px;'>"
        f"<span style='font-size:4.5rem;font-weight:900;color:{color};'>"
        f"{arrow} {decision}</span></div>"
        f"<p style='text-align:center;font-size:1.05rem;margin-top:0;'>"
        f"Consensus: <strong>{strength}</strong> &nbsp;|&nbsp; "
        f"Confidence: <strong>{conf}%</strong></p>",
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
    st.markdown("## 🎯 Recommended Action")

    if should_trade and direction in ("UP", "DOWN"):
        bg     = "#062e0f" if direction == "UP" else "#2e0606"
        txt    = "#00e676" if direction == "UP" else "#ff5252"
        border = "#00c853" if direction == "UP" else "#d50000"
        arrow  = "⬆️"     if direction == "UP" else "⬇️"

        st.markdown(
            f"<div style='background:{bg};border:3px solid {border};"
            f"border-radius:14px;padding:30px 24px;margin:10px 0;text-align:center;'>"
            f"<div style='font-size:2.4rem;'>{arrow}</div>"
            f"<div style='color:{txt};font-size:1.5rem;font-weight:900;"
            f"letter-spacing:.02em;line-height:1.55;margin-top:8px;'>"
            f"{display_text}</div></div>",
            unsafe_allow_html=True,
        )
        if candles and total_mins:
            m1, m2, m3 = st.columns(3)
            m1.metric("Direction",       f"{arrow} {direction}")
            m2.metric("Candles to Hold", str(candles))
            m3.metric("Est. Duration",   f"{total_mins} min")
    else:
        st.markdown(
            f"<div style='background:#2b1e00;border:3px solid #ff8f00;"
            f"border-radius:14px;padding:30px 24px;margin:10px 0;text-align:center;'>"
            f"<div style='font-size:2.4rem;'>🚫</div>"
            f"<div style='color:#ffd740;font-size:1.5rem;font-weight:900;"
            f"letter-spacing:.02em;line-height:1.55;margin-top:8px;'>"
            f"{display_text}</div></div>",
            unsafe_allow_html=True,
        )
        if no_trade_rsn and no_trade_rsn.lower() not in ("null", "none", ""):
            st.warning(f"⚠️ {no_trade_rsn}")


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 AI Trading Debate")
st.caption(
    "Crash-proof · 5 live AI models · auto-fallback on any outage · "
    "candle-based trade target"
)

st.markdown("### 1️⃣  Upload Chart")
uploaded = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "webp"],
    help="Screenshot of any trading chart",
)
if uploaded:
    st.image(Image.open(uploaded), caption="Uploaded chart", use_container_width=True)

st.markdown("### 2️⃣  Extra Context *(optional)*")
extra_ctx = st.text_area(
    "Any context for the AIs",
    placeholder="e.g. BTC/USDT 10-min chart, NY session open…",
    height=70,
)

st.markdown("### 3️⃣  Start")
run = st.button(
    "🚀 START AI DEBATE",
    disabled=(uploaded is None),
    use_container_width=True,
    type="primary",
)

if not run:
    st.stop()

uploaded.seek(0)
image_bytes = uploaded.read()
ext       = uploaded.name.rsplit(".", 1)[-1].lower()
mime_map  = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
             "png": "image/png",  "webp": "image/webp"}
mime_type = mime_map.get(ext, "image/jpeg")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Chart vision (Gemini → Groq vision fallback)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Step 1 — Chart Analysis & Market News")
with st.spinner("Reading chart… (trying Gemini, auto-fallback to Groq if needed)"):
    try:
        chart_data, vision_status = step1_analyze_chart(image_bytes, mime_type, extra_ctx)
    except Exception as exc:
        st.error(f"**Both vision models failed.** Cannot continue.\n\n{exc}")
        st.stop()

render_chart_analysis(chart_data, vision_status)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Independent votes (each fully isolated)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Step 2 — Independent Analyst Votes")

# ── Dynamic analyst roster ────────────────────────────────────────────────────
# Ask Groq's live API which text models are currently active, rank them by
# quality, and pick the top 3.  DeepSeek is always appended as analyst 4
# (handled gracefully if payment fails).
with st.spinner("🔍 Discovering active Groq text models…"):
    _groq_text_models = _discover_groq_text_models(n=3)

ANALYST_MODELS: list[tuple[str, str | None, str]] = [
    ("groq", mid, _short_label(mid))
    for mid, _ in _groq_text_models
] + [
    ("deepseek", None, "DeepSeek-Chat"),
]

# Show which models were auto-selected
_groq_labels = ", ".join(_short_label(mid) for mid, _ in _groq_text_models)
st.caption(f"🤖 Auto-selected Groq analysts: **{_groq_labels}**")

analyst_votes: list[dict] = []
offline_models: list[str] = []

for backend, model_id, model_name in ANALYST_MODELS:
    icon = MODEL_ICON.get(model_name, "🤖")
    with st.spinner(f"{icon} {model_name} is analysing…"):
        try:
            if backend == "groq":
                result = _call_groq(model_id, model_name, chart_data)
            else:
                result = _call_deepseek(chart_data)
            analyst_votes.append(result)
            render_vote_card(result, status="online")
        except Exception as exc:
            err_msg = str(exc)
            st.warning(f"🔴 **{model_name} is offline** — {err_msg[:160]}")
            offline_models.append(model_name)
            render_vote_card(
                {"model": model_name, "error_msg": err_msg},
                status="offline",
            )

# Need at least 1 analyst + vision model = 2 total opinions
if len(analyst_votes) == 0:
    st.error(
        "All analyst models are currently offline. "
        "Only the vision analysis is available. Please try again in a few minutes."
    )
    st.stop()

render_scoreboard(chart_data, analyst_votes, offline_models)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Gemini cross-examination + final verdict (Groq fallback)
# ══════════════════════════════════════════════════════════════════════════════
n_online = 1 + len(analyst_votes)
st.markdown("---")
st.markdown(
    f"## Step 3 — Cross-Examination & Final Verdict  "
    f"*({n_online}/{ 1 + len(ANALYST_MODELS)} models online)*"
)

if offline_models:
    st.caption(f"🔴 Offline this run: {', '.join(offline_models)} — verdict based on available models.")

with st.spinner("Cross-examining all opinions and computing final verdict…"):
    try:
        synthesis = step3_synthesize(chart_data, analyst_votes)
    except Exception as exc:
        st.error(f"**Final synthesis failed:** {exc}")
        st.stop()

render_final_decision(synthesis)
render_trade_recommendation(synthesis)
st.balloons()
