"""
AI Trading Debate — crash-proof, future-proof.

Pipeline
────────
Step 1  Chart Vision  : Gemini 2.5 Flash (primary)
                         → live Groq vision model fallback
                         → text-fallback if vision APIs fail
Step 2  Analyst Votes : Gemini 2.5 Flash + top Groq text models
                         (live auto-discovery)
                          explicit preference: llama-3.3-70b-versatile, qwen-2.5-32b
                        + DeepSeek-Chat (skipped gracefully on 402)
                        + any custom models added by the user in the sidebar
                        <think>…</think> tokens stripped before JSON parsing
Step 3  Final Verdict : Top Groq text model synthesis
                        → FINAL_DECISION + candle recommendation box
"""

import base64
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
    #    instead of JSON. Pull vote / confidence / reasoning from keywords.
    txt_up = text.upper()
    vote   = "WAIT"
    if re.search(r'\bBULLISH\b|\b"UP"\b|\bVOTE[:\s]+UP\b|DIRECTION[:\s]+UP', txt_up):
        vote = "UP"
    elif re.search(r'\bBEARISH\b|\b"DOWN"\b|\bVOTE[:\s]+DOWN\b|DIRECTION[:\s]+DOWN', txt_up):
        vote = "DOWN"
    conf_m = re.search(r'(\d{1,3})\s*%', text)
    conf   = min(int(conf_m.group(1)), 100) if conf_m else 50
    lines  = [l.strip() for l in text.split("\n") if len(l.strip()) > 25]
    rsn    = lines[0][:140] if lines else "Model output could not be parsed as JSON."
    return {"vote": vote, "confidence": conf, "reasoning": rsn,
            "_text_extraction": True}


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
    from google import genai
    from google.genai import types

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)
    extra_line = (f"\n\nExtra context: {extra.strip()}") if extra.strip() else ""
    prompt = (
        CHART_PROMPT_TEXT
        + extra_line
        + "\n"
        + CHART_ANALYSIS_JSON_SPEC
        + "\n\nNote: Fill live_news with your best knowledge of recent market events."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
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
    result["vision_model"] = "Gemini 2.5 Flash"
    result["vision_provider"] = "Gemini"
    result["gemini_vision_model"] = "Gemini 2.5 Flash"
    result["gemini_vision_vote"] = result.get("vote") or "WAIT"
    result["gemini_vision_confidence"] = result.get("confidence") or 50
    result["gemini_vision_reasoning"] = (
        result.get("reasoning") or "Gemini chart analysis."
    )
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
    result["vision_provider"] = "Groq"
    # Normalise — model may return "vote" or "gemini_vote"; expose both consistently
    vote = result.get("vote") or result.get("gemini_vote") or "WAIT"
    conf = result.get("confidence") or result.get("gemini_confidence") or 50
    rsn  = result.get("reasoning") or result.get("gemini_reasoning") or "Groq vision analysis."
    result["vote"]       = vote
    result["confidence"] = conf
    result["reasoning"]  = rsn
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
    Runs Gemini first, then Groq vision as a fallback.
    Returns (chart_data, status_message). Never raises.
    """
    gemini_error = ""
    try:
        chart_data = _gemini_vision(image_bytes, mime, extra)
        chart_data["groq_vision_model"] = "Not used"
        chart_data["groq_vision_vote"] = "—"
        chart_data["groq_vision_confidence"] = 0
        chart_data["groq_vision_reasoning"] = "Gemini completed the primary vision pass."
        return chart_data, "✅ **Gemini 2.5 Flash** analyzed the chart."
    except Exception as exc:
        gemini_error = str(exc)[:160]

    groq_data       = None
    groq_error      = ""
    groq_model_used = ""
    for model_id in _discover_groq_vision_models():
        try:
            groq_data       = _groq_vision(image_bytes, mime, extra, model_id, model_id)
            groq_model_used = model_id
            break
        except Exception as exc:
            groq_error += f"{model_id}: {str(exc)[:60]}; "

    if groq_data:
        chart_data = groq_data
        chart_data["groq_vision_model"]      = groq_model_used
        chart_data["groq_vision_vote"]       = groq_data.get("vote", "WAIT")
        chart_data["groq_vision_confidence"] = groq_data.get("confidence", 50)
        chart_data["groq_vision_reasoning"]  = groq_data.get("reasoning", "")
        status = "✅ **" + groq_model_used.split("/")[-1] + "** ⚡ analyzed the chart."
    else:
        chart_data = dict(_TEXT_FALLBACK_DATA)
        if extra.strip():
            chart_data["technical_summary"] += "  User context: " + extra.strip()
        chart_data["groq_vision_model"]      = "Groq Vision"
        chart_data["groq_vision_vote"]       = "Offline"
        chart_data["groq_vision_confidence"] = 0
        chart_data["groq_vision_reasoning"]  = "Vision API unavailable."
        chart_data["vision_provider"]        = "Text fallback"
        status = (
            "🚨 **Gemini and Groq vision unavailable** — text-fallback mode.\n\n"
            "Gemini: " + gemini_error + "\nGroq: " + groq_error[:220]
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
        "  ⚠️  TRADER'S NOTES — READ AND FOLLOW THESE CAREFULLY\n"
        "══════════════════════════════════════════════════════\n"
        + user_ctx + "\n\n"
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
        "Work through each point; summarise findings in the 'analysis' field:\n\n"
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
    from groq import Groq
    client  = Groq(api_key=require_secret("GROQ_API_KEY"))
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
    from google import genai
    from google.genai import types

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    prompt = build_analyst_prompt("Gemini 2.5 Flash", chart_data)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )
    result = parse_json(response.text or "")
    result.setdefault("model", "Gemini 2.5 Flash")
    result["_platform"] = "Gemini"
    result["_model_id"] = "gemini-2.5-flash"
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
            "max_tokens": 1200,
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
    result["_model_id"]  = "deepseek-chat"   # stored for Round 2 deliberation
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DELIBERATION ROUND — AIs share all reasoning and submit final votes
# ─────────────────────────────────────────────────────────────────────────────
def _call_groq_raw(model_id: str, model_name: str, prompt: str) -> dict:
    """Call Groq with a custom prompt (used for deliberation round)."""
    from groq import Groq
    client   = Groq(api_key=require_secret("GROQ_API_KEY"))
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
    from google import genai
    from google.genai import types

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
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
    result["_model_id"] = "gemini-2.5-flash"
    return result


def _call_deepseek_raw(model_name: str, prompt: str) -> dict:
    """Call DeepSeek with a custom prompt (used for deliberation round)."""
    api_key = require_secret("DEEPSEEK_API_KEY")
    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an AI trading analyst. Respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.3, "max_tokens": 500,
        },
        timeout=40,
    )
    if resp.status_code == 402:
        raise RuntimeError("DeepSeek 402")
    resp.raise_for_status()
    result = parse_json(resp.json()["choices"][0]["message"]["content"])
    result.setdefault("model", model_name)
    result["_platform"] = "DeepSeek"
    return result


def _build_deliberation_prompt(model_name: str, all_votes: list[dict]) -> str:
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
    return (
        "You are " + model_name + ", participating in a structured AI trading debate.\n"
        "All analysts have submitted their Round 1 positions. "
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


def run_deliberation_round(analyst_votes: list[dict]) -> list[dict]:
    """
    Round 2: every analyst sees all peers' full reasoning and gives a
    final vote.  Falls back to Round 1 result on any API failure.
    Returns a list in the same format as analyst_votes.
    """
    revised: list[dict] = []
    for v in analyst_votes:
        model_name = v.get("model", "?")
        model_id   = v.get("_model_id", "")
        platform   = v.get("_platform", "Groq")
        prompt     = _build_deliberation_prompt(model_name, analyst_votes)
        try:
            if platform == "Groq" and model_id:
                r = _call_groq_raw(model_id, model_name, prompt)
            elif platform == "Gemini":
                r = _call_gemini_raw(model_name, prompt)
            elif platform == "DeepSeek":
                r = _call_deepseek_raw(model_name, prompt)
            else:
                revised.append({**v, "_deliberated": False, "_round1_vote": v.get("vote", "?")})
                continue
            r["_platform"]    = platform
            r["_model_id"]    = model_id
            r["_deliberated"] = True
            r["_round1_vote"] = v.get("vote", "?")
            r.setdefault("key_risks", v.get("key_risks", []))
            r.setdefault("analysis",  v.get("analysis", ""))
            revised.append(r)
        except Exception:
            revised.append({**v, "_deliberated": False, "_round1_vote": v.get("vote", "?")})
    return revised


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Synthesis & final verdict
# ─────────────────────────────────────────────────────────────────────────────
def _build_synthesis_prompt(chart_data: dict, analyst_votes: list[dict]) -> str:
    tf         = chart_data.get("timeframe", "unknown")
    tf_minutes = int(chart_data.get("timeframe_minutes", 0) or 0)
    asset      = chart_data.get("asset", "the asset")
    user_ctx   = (chart_data.get("user_context") or "").strip()

    positions = (
        chart_data.get("vision_model", "Vision Model").split("/")[-1] + " (Vision): "
        + chart_data.get("gemini_vision_vote",
                          chart_data.get("groq_vision_vote", "WAIT"))
        + " (" + str(chart_data.get("gemini_vision_confidence",
                                    chart_data.get("groq_vision_confidence", 0))) + "%) — "
        + chart_data.get("gemini_vision_reasoning",
                         chart_data.get("groq_vision_reasoning", "")) + "\n"
    )
    for v in analyst_votes:
        positions += (
            v.get("model", "?") + ": " + v.get("vote", "WAIT")
            + " (" + str(v.get("confidence", 0)) + "%) — "
            + v.get("reasoning", "") + "\n"
            "  Analysis  : " + v.get("analysis", "") + "\n"
            "  Key Risks : " + ", ".join(v.get("key_risks", [])) + "\n"
        )

    online_names = (
        [chart_data.get("vision_model", "Vision Model").split("/")[-1]]
        + [v.get("model", "?") for v in analyst_votes]
    )

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
        "=== TRADER'S NOTES (follow these — they override defaults) ===\n"
        + user_ctx + "\n"
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
    from groq import Groq

    prompt  = _build_synthesis_prompt(chart_data, analyst_votes)
    models  = _discover_groq_text_models(n=2)
    last_err = ""
    for model_id, label in models:
        try:
            client = Groq(api_key=require_secret("GROQ_API_KEY"))
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
            return result
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError("Synthesis failed — " + last_err)


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

/* ── Base dark theme ────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #05080f !important;
    color: #c8d6e8;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
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

    # Vision model always gets its own scoreboard tile
    vision_model = chart_data.get("vision_model", "Vision Model")
    vision_provider = chart_data.get("vision_provider", "Vision AI")
    entries = [
        {
            "model":    vision_model.split("/")[-1],
            "vote":     chart_data.get(
                "gemini_vision_vote",
                chart_data.get("groq_vision_vote", "Offline"),
            ),
            "conf":     chart_data.get(
                "gemini_vision_confidence",
                chart_data.get("groq_vision_confidence", 0),
            ),
            "icon":     "✨" if vision_provider == "Gemini" else "⚡",
            "platform": vision_provider,
        },
    ] + [{
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
        "🧩 AI Model Manager</h2>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Live catalog browser ───────────────────────────────
    st.markdown(
        "<p style='font-size:.75rem;font-weight:700;color:#8090a8;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>"
        "⚡ Live Groq Catalog</p>",
        unsafe_allow_html=True,
    )

    @st.cache_data(ttl=120, show_spinner=False)
    def _get_full_catalog() -> list[tuple[str, str]]:
        return _discover_groq_text_models(n=12)

    catalog   = _get_full_catalog()
    catalog_ids  = [mid for mid, _ in catalog]
    auto_ids     = catalog_ids[:3]
    already_custom = st.session_state.get("custom_models", [])
    addable      = [mid for mid in catalog_ids
                    if mid not in auto_ids and mid not in already_custom]

    for mid in auto_ids:
        st.markdown(
            "<div class='catalog-row'>"
            "<span style='color:#00ff88;font-size:.8rem;'>✅</span>"
            "<span style='flex:1;margin:0 6px;font-size:.78rem;color:#a0b8d0;'>"
            + _short_label(mid) + "</span>"
            "<span style='font-size:.6rem;color:#2a5040;background:#0a1e14;"
            "padding:1px 5px;border-radius:4px;'>AUTO</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p style='font-size:.68rem;color:#2a3a50;margin:4px 0 0;'>"
        "🔄 Auto-refreshed every 2 min from Groq's live API</p>",
        unsafe_allow_html=True,
    )

    # Quick-add from remaining discovered models
    if addable:
        st.markdown(
            "<p style='font-size:.75rem;font-weight:700;color:#8090a8;"
            "text-transform:uppercase;letter-spacing:.06em;margin:12px 0 4px;'>"
            "➕ Quick-Add from Catalog</p>",
            unsafe_allow_html=True,
        )
        quick_pick = st.selectbox(
            "pick_model",
            ["— choose a model —"] + [_short_label(m) + "  |  " + m for m in addable],
            label_visibility="collapsed",
        )
        if st.button("Add Selected ➕", use_container_width=True):
            if quick_pick and "—" not in quick_pick:
                picked_id = quick_pick.split("  |  ")[-1].strip()
                if picked_id not in st.session_state["custom_models"]:
                    st.session_state["custom_models"].append(picked_id)
                    st.rerun()

    # ── Section 2: Manual model ID entry ──────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='font-size:.75rem;font-weight:700;color:#8090a8;"
        "text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>"
        "🤖 Add Any Model Manually</p>",
        unsafe_allow_html=True,
    )

    custom_input = st.text_input(
        "Groq Model ID",
        key="custom_model_input",
        placeholder="e.g. meta-llama/llama-4-maverick…",
        label_visibility="collapsed",
    )
    if st.button("➕ Add to Committee", use_container_width=True):
        cid = (custom_input or "").strip()
        if cid and cid not in st.session_state["custom_models"]:
            st.session_state["custom_models"].append(cid)
            st.rerun()

    # ── Section 3: Active custom models ───────────────────────────────
    if st.session_state["custom_models"]:
        st.markdown(
            "<p style='font-size:.75rem;color:#566880;margin:10px 0 4px;'>"
            "Your custom analysts:</p>",
            unsafe_allow_html=True,
        )
        for i, mid in enumerate(list(st.session_state["custom_models"])):
            c1, c2 = st.columns([5, 1])
            c1.markdown(
                "<div class='custom-model-tag'>🤖 " + _short_label(mid) + "</div>",
                unsafe_allow_html=True,
            )
            if c2.button("✕", key="rm_" + str(i), help="Remove"):
                st.session_state["custom_models"].pop(i)
                st.rerun()
    else:
        st.markdown(
            "<p style='font-size:.78rem;color:#1e2e40;font-style:italic;margin-top:8px;'>"
            "No custom models added yet.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:.7rem;color:#1e2e40;'>"
        "Gemini 2.5 Flash &amp; DeepSeek-Chat always participate automatically.<br>"
        "All added models use the same JSON vote rules.</p>",
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

asset_name = chart_data.get("asset", "asset")
# Store trader's extra context in chart_data so ALL prompts can see it
chart_data["user_context"] = extra_ctx.strip() if extra_ctx else ""

with st.spinner("🌐 Searching latest market news for " + asset_name + "…"):
    chart_data = _inject_real_news(chart_data)

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
    [("gemini", "gemini-2.5-flash", "Gemini 2.5 Flash")]
    + [("groq", mid, _short_label(mid)) for mid, _ in _groq_text_models]
    + [("deepseek", "", "DeepSeek-Chat")]
    + [("groq", mid, mid) for mid in st.session_state["custom_models"]]
)

analyst_votes:  list[dict] = []
offline_models: list[str]  = []

for backend, model_id, model_name in ANALYST_MODELS:
    with st.spinner(_model_icon(model_name) + " " + model_name + " is analysing…"):
        try:
            if backend == "gemini":
                result = _call_gemini(chart_data)
            elif backend == "groq":
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
# DELIBERATION — AIs share full reasoning and submit revised final votes
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("🔄 Running deliberation round — AIs reading each other's reasoning…"):
    revised_votes = run_deliberation_round(analyst_votes)

render_deliberation_round(analyst_votes, revised_votes)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cross-examination & final verdict (uses post-deliberation votes)
# ══════════════════════════════════════════════════════════════════════════════
n_online = 1 + len(revised_votes)
n_total  = 1 + len(ANALYST_MODELS)
st.markdown("---")
st.markdown(
    "<div class='step-header'>Step 3 — Cross-Examination & Final Verdict"
    "<span style='font-size:.8rem;font-weight:400;color:#566880;margin-left:8px;'>"
    "based on post-deliberation positions · "
    + str(n_online) + "/" + str(n_total) + " models online</span></div>",
    unsafe_allow_html=True,
)
if offline_models:
    st.markdown(
        "<p style='font-size:.78rem;color:#566880;'>🔴 Offline this run: "
        + ", ".join(offline_models) + "</p>",
        unsafe_allow_html=True,
    )

with st.spinner("🧠 Cross-examining all deliberated positions and computing final verdict…"):
    try:
        synthesis = step3_synthesize(chart_data, revised_votes)
    except Exception as exc:
        st.error("**Final synthesis failed:** " + str(exc))
        st.stop()

render_final_decision(synthesis)
render_trade_recommendation(synthesis)
st.balloons()
