"""
AI Trading Debate — 5 live models, real APIs, full cross-examination round.

Pipeline
────────
Step 1  Gemini 2.5 Flash  →  reads the chart image + fetches live market news
Step 2  Llama 3 70b / Mixtral 8x7b / Gemma 2 9b (Groq) + DeepSeek-Chat
        →  each receives Gemini's findings and gives an independent analysis
Step 3  Cross-Examination  →  every model critiques the others and refines its
        position
Step 4  Final Synthesis  →  Gemini moderates and delivers FINAL_DECISION
"""

import base64
import json
import os
import textwrap
import time

import requests
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Trading Debate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Secrets
# ─────────────────────────────────────────────────────────────────────────────
def _secret(key: str) -> str:
    # 1. Try Streamlit secrets (.streamlit/secrets.toml)
    val = ""
    try:
        val = st.secrets.get(key, "") or ""
    except Exception:
        pass
    # 2. Fall back to Replit Secrets (stored as environment variables)
    if not val:
        val = os.environ.get(key, "") or ""
    if not val:
        st.error(
            f"**{key}** is not set.  \n\n"
            "The key was not found in Streamlit secrets or environment variables.  \n"
            "Add it via the Replit **Secrets** panel (🔒 icon in the left sidebar)."
        )
        st.stop()
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Cached clients
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _gemini_client():
    from google import genai
    return genai.Client(api_key=_secret("GEMINI_API_KEY"))


@st.cache_resource
def _groq_client():
    from groq import Groq
    return Groq(api_key=_secret("GROQ_API_KEY"))


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction helper
# ─────────────────────────────────────────────────────────────────────────────
def _parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    raw = text
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1:
        raise ValueError(f"No JSON object found in response:\n{text}")
    return json.loads(raw[start:end])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Gemini: chart vision + live news (search grounding)
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_CHART_PROMPT = """You are a senior technical analyst and financial journalist.

Analyse the trading chart image in detail, then use your search capability to
find the latest real-time market news related to the asset shown.

Return ONLY valid JSON (no markdown, no extra text) in this exact structure:
{
  "asset":          "<ticker or name you identified from the chart>",
  "timeframe":      "<detected or estimated timeframe>",
  "current_price":  "<approximate price visible on chart>",
  "trend":          "<Bullish | Bearish | Sideways>",
  "key_levels": {
    "support":    ["<level 1>", "<level 2>"],
    "resistance": ["<level 1>", "<level 2>"]
  },
  "indicators": {
    "<indicator name>": "<reading and interpretation>"
  },
  "chart_patterns":  ["<pattern 1>", "<pattern 2>"],
  "technical_summary": "<3-4 sentence technical narrative>",
  "live_news": [
    {"headline": "<headline>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<source or search result>"},
    {"headline": "<headline>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<source or search result>"},
    {"headline": "<headline>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<source or search result>"}
  ],
  "news_summary": "<2 sentence summary of overall news sentiment>",
  "gemini_vote":  "<UP|DOWN|WAIT>",
  "gemini_confidence": <integer 0-100>,
  "gemini_reasoning": "<one concise sentence>"
}"""

GEMINI_SYNTH_PROMPT_TPL = """You are the debate moderator and final decision-maker.

Below is the complete debate transcript between five AI trading analysts.
Study all initial positions and all cross-examination responses, then deliver
a final verdict AND a precise candle-based trade recommendation.

Chart timeframe detected by Gemini: {timeframe}

=== DEBATE TRANSCRIPT ===
{transcript}
=========================

RECOMMENDED ACTION rules (apply strictly):
- If FINAL_DECISION is UP or DOWN:
    * Estimate how many consecutive candles (1–5) the move is likely to last,
      based on momentum, volume, and pattern strength visible in the debate.
    * Calculate total_duration_minutes = candle_count × candle_duration_minutes.
    * candle_duration_minutes must match the detected timeframe exactly
      (e.g. 1 for 1-min, 5 for 5-min, 10 for 10-min, 15 for 15-min, 60 for 1H, etc.).
    * Set should_trade to true.
    * Set dont_trade_reason to null.
    * Build display_text in exactly this format:
      "TRADE DIRECTION: <UP|DOWN> | TARGET: Next <N> candles will go <UP|DOWN> (Duration: <total> minutes on a <tf>-min chart)"
- If FINAL_DECISION is WAIT:
    * Set should_trade to false, trade_direction to null, candle_count to null,
      candle_duration_minutes to null, total_duration_minutes to null.
    * Write a specific dont_trade_reason (volatility, low volume, conflicting signals, etc.).
    * Build display_text in exactly this format:
      "DON'T TRADE: <specific reason — market conditions, asset name, and timeframe>"

Return ONLY valid JSON (no markdown):
{{
  "vote_tally": {{"UP": <int>, "DOWN": <int>, "WAIT": <int>}},
  "consensus_strength": "<STRONG | MODERATE | DIVIDED>",
  "key_agreements":    "<what the majority agreed on>",
  "key_disagreements": "<main point(s) of disagreement>",
  "FINAL_DECISION":    "<UP|DOWN|WAIT>",
  "confidence":        <integer 0-100>,
  "action":            "<one clear, actionable sentence for a trader>",
  "moderator_note":    "<2-3 sentence moderator summary>",
  "recommended_action": {{
    "should_trade":             <true|false>,
    "trade_direction":          "<UP|DOWN|null>",
    "candle_count":             <integer or null>,
    "candle_duration_minutes":  <integer or null>,
    "total_duration_minutes":   <integer or null>,
    "dont_trade_reason":        "<string or null>",
    "display_text":             "<formatted string per rules above>"
  }}
}}"""


def step1_gemini_analyze(image_bytes: bytes, mime: str) -> dict:
    from google.genai import types

    client = _gemini_client()
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(inline_data=types.Blob(mime_type=mime, data=image_bytes)),
                types.Part(text=GEMINI_CHART_PROMPT),
            ],
        )
    ]
    # Use search grounding for live news
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )
    except Exception:
        # Fallback without grounding if unavailable
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
    return _parse_json(resp.text)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Initial votes from text-only models
# ─────────────────────────────────────────────────────────────────────────────
INITIAL_VOTE_PROMPT_TPL = """You are {model_name}, a specialist AI trading analyst.

Gemini has analysed a trading chart and retrieved live market news. Your job is
to study Gemini's findings and provide your own independent trading opinion.

=== GEMINI'S CHART ANALYSIS ===
Asset       : {asset}
Timeframe   : {timeframe}
Trend       : {trend}
Support     : {support}
Resistance  : {resistance}
Indicators  : {indicators}
Patterns    : {patterns}
Technical   : {technical_summary}
Gemini Vote : {gemini_vote} ({gemini_confidence}% confidence)
Gemini Says : {gemini_reasoning}

=== LIVE MARKET NEWS ===
{news_block}

News Summary: {news_summary}
===============================

Provide your own analysis and vote. Return ONLY valid JSON:
{{
  "model":      "{model_name}",
  "analysis":   "<2-3 sentence technical + fundamental analysis>",
  "key_risks":  ["<risk 1>", "<risk 2>"],
  "vote":       "<UP|DOWN|WAIT>",
  "confidence": <integer 0-100>,
  "reasoning":  "<one concise sentence explaining your vote>"
}}"""


def _build_initial_prompt(model_name: str, g: dict) -> str:
    news_lines = "\n".join(
        f"  [{i+1}] ({item.get('sentiment','?')}) {item.get('headline','')}"
        f"  — {item.get('source','')}"
        for i, item in enumerate(g.get("live_news", []))
    )
    indicators_str = "; ".join(
        f"{k}: {v}" for k, v in g.get("indicators", {}).items()
    ) or "N/A"
    return INITIAL_VOTE_PROMPT_TPL.format(
        model_name=model_name,
        asset=g.get("asset", "Unknown"),
        timeframe=g.get("timeframe", "Unknown"),
        trend=g.get("trend", "Unknown"),
        support=", ".join(g.get("key_levels", {}).get("support", [])),
        resistance=", ".join(g.get("key_levels", {}).get("resistance", [])),
        indicators=indicators_str,
        patterns=", ".join(g.get("chart_patterns", [])),
        technical_summary=g.get("technical_summary", ""),
        gemini_vote=g.get("gemini_vote", "WAIT"),
        gemini_confidence=g.get("gemini_confidence", 0),
        gemini_reasoning=g.get("gemini_reasoning", ""),
        news_block=news_lines or "No live news retrieved.",
        news_summary=g.get("news_summary", ""),
    )


GROQ_MODELS = [
    ("llama3-70b-8192",   "Llama 3 70B"),
    ("mixtral-8x7b-32768","Mixtral 8x7B"),
    ("gemma2-9b-it",      "Gemma 2 9B"),
]

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


def groq_initial_vote(model_id: str, model_name: str, gemini_data: dict) -> dict:
    client = _groq_client()
    prompt = _build_initial_prompt(model_name, gemini_data)
    chat = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert AI trading analyst. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=600,
    )
    return _parse_json(chat.choices[0].message.content)


def deepseek_initial_vote(gemini_data: dict) -> dict:
    api_key = _secret("DEEPSEEK_API_KEY")
    prompt = _build_initial_prompt("DeepSeek-Chat", gemini_data)
    resp = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an expert AI trading analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 600,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return _parse_json(resp.json()["choices"][0]["message"]["content"])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Cross-Examination round
# ─────────────────────────────────────────────────────────────────────────────
DEBATE_PROMPT_TPL = """You are {model_name}, participating in a structured
cross-examination debate with four other AI trading analysts.

=== INITIAL POSITIONS ===
{initial_positions}
=========================

Your task:
1. Address at least TWO other analysts by name — challenge, support, or
   refine their arguments with specific technical or fundamental reasoning.
2. State your FINAL refined vote for this debate round (can differ from your
   initial if persuaded).

Return ONLY valid JSON:
{{
  "model":             "{model_name}",
  "responses_to": [
    {{"target": "<model name>", "stance": "<AGREE|DISAGREE|PARTIALLY_AGREE>",
      "argument": "<1-2 sentence specific technical rebuttal or support>"}},
    {{"target": "<model name>", "stance": "<AGREE|DISAGREE|PARTIALLY_AGREE>",
      "argument": "<1-2 sentence specific technical rebuttal or support>"}}
  ],
  "refined_vote":      "<UP|DOWN|WAIT>",
  "refined_confidence":<integer 0-100>,
  "final_reasoning":   "<one concise sentence>"
}}"""


def _build_positions_block(initial_votes: list[dict], gemini_data: dict) -> str:
    gemini_block = (
        f"• Gemini 2.5 Flash  →  {gemini_data.get('gemini_vote','WAIT')} "
        f"({gemini_data.get('gemini_confidence',0)}%): "
        f"{gemini_data.get('gemini_reasoning','')}"
    )
    analyst_blocks = "\n".join(
        f"• {v.get('model', '?')}  →  {v.get('vote','WAIT')} "
        f"({v.get('confidence',0)}%): {v.get('reasoning','')}"
        for v in initial_votes
    )
    return gemini_block + "\n" + analyst_blocks


def groq_debate(model_id: str, model_name: str,
                initial_votes: list[dict], gemini_data: dict) -> dict:
    client = _groq_client()
    positions = _build_positions_block(initial_votes, gemini_data)
    prompt = DEBATE_PROMPT_TPL.format(
        model_name=model_name,
        initial_positions=positions,
    )
    chat = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert AI trading analyst in a live debate. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=700,
    )
    return _parse_json(chat.choices[0].message.content)


def deepseek_debate(initial_votes: list[dict], gemini_data: dict) -> dict:
    api_key = _secret("DEEPSEEK_API_KEY")
    positions = _build_positions_block(initial_votes, gemini_data)
    prompt = DEBATE_PROMPT_TPL.format(
        model_name="DeepSeek-Chat",
        initial_positions=positions,
    )
    resp = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an expert AI trading analyst in a live debate. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.5,
            "max_tokens": 700,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return _parse_json(resp.json()["choices"][0]["message"]["content"])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Gemini final synthesis
# ─────────────────────────────────────────────────────────────────────────────
def step4_gemini_synthesize(gemini_data: dict,
                             initial_votes: list[dict],
                             debate_responses: list[dict]) -> dict:
    from google.genai import types

    lines = []
    lines.append("── INITIAL POSITIONS ──")
    lines.append(
        f"Gemini 2.5 Flash: {gemini_data.get('gemini_vote','WAIT')} "
        f"({gemini_data.get('gemini_confidence',0)}%) — "
        f"{gemini_data.get('gemini_reasoning','')}"
    )
    for v in initial_votes:
        lines.append(
            f"{v.get('model','?')}: {v.get('vote','WAIT')} "
            f"({v.get('confidence',0)}%) — {v.get('reasoning','')}"
        )
    lines.append("\n── CROSS-EXAMINATION RESPONSES ──")
    for d in debate_responses:
        lines.append(
            f"\n{d.get('model','?')} [refined: {d.get('refined_vote','WAIT')} "
            f"@ {d.get('refined_confidence',0)}%] — {d.get('final_reasoning','')}"
        )
        for r in d.get("responses_to", []):
            lines.append(
                f"  ↳ To {r.get('target','?')} [{r.get('stance','?')}]: "
                f"{r.get('argument','')}"
            )

    transcript = "\n".join(lines)
    timeframe = gemini_data.get("timeframe", "unknown")
    prompt = GEMINI_SYNTH_PROMPT_TPL.format(transcript=transcript, timeframe=timeframe)

    client = _gemini_client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
    )
    result = _parse_json(resp.text)
    result["_transcript"] = transcript
    return result


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────
VOTE_COLOR = {"UP": "green", "DOWN": "red", "WAIT": "orange"}
VOTE_ICON  = {"UP": "⬆️", "DOWN": "⬇️", "WAIT": "⏸️"}
MODEL_ICON = {
    "Gemini 2.5 Flash": "✨",
    "Llama 3 70B":      "🦙",
    "Mixtral 8x7B":     "⚗️",
    "Gemma 2 9B":       "💎",
    "DeepSeek-Chat":    "🔭",
}


def _vote_badge(vote: str) -> str:
    c = VOTE_COLOR.get(vote, "gray")
    i = VOTE_ICON.get(vote, "❓")
    return f":{c}[**{i} {vote}**]"


def render_gemini_analysis(g: dict):
    with st.expander("✨ Gemini 2.5 Flash — Chart Analysis + Live News", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset", g.get("asset", "—"))
        c2.metric("Timeframe", g.get("timeframe", "—"))
        c3.metric("Trend", g.get("trend", "—"))
        vote = g.get("gemini_vote", "WAIT")
        c4.metric("Gemini Vote", f"{VOTE_ICON.get(vote,'')} {vote}", f"{g.get('gemini_confidence',0)}% confidence")

        st.markdown("**Technical Summary**")
        st.info(g.get("technical_summary", ""))

        lvls = g.get("key_levels", {})
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Support Levels**")
            for s in lvls.get("support", []):
                st.markdown(f"- `{s}`")
        with cb:
            st.markdown("**Resistance Levels**")
            for r in lvls.get("resistance", []):
                st.markdown(f"- `{r}`")

        inds = g.get("indicators", {})
        if inds:
            st.markdown("**Indicators**")
            for k, v in inds.items():
                st.markdown(f"- **{k}**: {v}")

        if g.get("chart_patterns"):
            st.markdown("**Patterns detected:** " + " · ".join(f"`{p}`" for p in g["chart_patterns"]))

        st.markdown("---")
        st.markdown("**📰 Live Market News**")
        for item in g.get("live_news", []):
            sent = item.get("sentiment", "Neutral")
            icon = "🟢" if sent == "Bullish" else ("🔴" if sent == "Bearish" else "🟡")
            st.markdown(f"{icon} **{item.get('headline','')}**  \n*{item.get('source','')}*")

        st.markdown(f"**News Sentiment Summary:** {g.get('news_summary','')}")
        st.markdown(f"💬 *{g.get('gemini_reasoning','')}*")


def render_initial_vote(v: dict):
    model = v.get("model", "Unknown")
    vote  = v.get("vote", "WAIT")
    icon  = MODEL_ICON.get(model, "🤖")
    with st.expander(
        f"{icon} **{model}** — {_vote_badge(vote)} — {v.get('confidence',0)}% confidence",
        expanded=True,
    ):
        st.markdown(f"**Analysis:** {v.get('analysis','')}")
        risks = v.get("key_risks", [])
        if risks:
            st.markdown("**Key Risks:** " + " · ".join(f"`{r}`" for r in risks))
        st.markdown(f"💬 *{v.get('reasoning','')}*")


def render_debate_response(d: dict):
    model = d.get("model", "Unknown")
    vote  = d.get("refined_vote", "WAIT")
    icon  = MODEL_ICON.get(model, "🤖")
    with st.expander(
        f"{icon} **{model}** — Refined: {_vote_badge(vote)} — {d.get('refined_confidence',0)}%",
        expanded=True,
    ):
        for r in d.get("responses_to", []):
            stance = r.get("stance", "?")
            s_icon = "✅" if stance == "AGREE" else ("❌" if stance == "DISAGREE" else "🔄")
            st.markdown(
                f"{s_icon} **To {r.get('target','?')}** [{stance}]: {r.get('argument','')}"
            )
        st.markdown(f"💬 *Final reasoning: {d.get('final_reasoning','')}*")


def render_trade_recommendation(synth: dict):
    """Render the bold highlighted RECOMMENDED ACTION box at the very bottom."""
    ra = synth.get("recommended_action", {})
    if not ra:
        return

    should_trade = ra.get("should_trade", False)
    display_text = ra.get("display_text", "")
    direction    = ra.get("trade_direction", "")
    candles      = ra.get("candle_count")
    tf_mins      = ra.get("candle_duration_minutes")
    total_mins   = ra.get("total_duration_minutes")
    no_trade_rsn = ra.get("dont_trade_reason", "")

    st.markdown("---")
    st.markdown("## 🎯 Recommended Action")

    if should_trade and direction in ("UP", "DOWN"):
        bg_color  = "#0a3d0a" if direction == "UP" else "#3d0a0a"
        txt_color = "#00ff88" if direction == "UP" else "#ff4444"
        border    = "#00cc66" if direction == "UP" else "#cc0000"
        arrow     = "⬆️" if direction == "UP" else "⬇️"

        st.markdown(
            f"""
            <div style="
                background-color:{bg_color};
                border:3px solid {border};
                border-radius:12px;
                padding:28px 32px;
                margin:12px 0 8px 0;
                text-align:center;
            ">
                <div style="font-size:2rem;margin-bottom:8px;">{arrow}</div>
                <div style="
                    color:{txt_color};
                    font-size:1.55rem;
                    font-weight:900;
                    letter-spacing:0.02em;
                    line-height:1.5;
                ">{display_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if candles and tf_mins and total_mins:
            c1, c2, c3 = st.columns(3)
            c1.metric("Trade Direction", f"{arrow} {direction}")
            c2.metric("Candles to Hold", f"{candles} candle{'s' if candles != 1 else ''}")
            c3.metric("Estimated Duration", f"{total_mins} min")

    else:
        st.markdown(
            f"""
            <div style="
                background-color:#2d2200;
                border:3px solid #cc8800;
                border-radius:12px;
                padding:28px 32px;
                margin:12px 0 8px 0;
                text-align:center;
            ">
                <div style="font-size:2rem;margin-bottom:8px;">🚫</div>
                <div style="
                    color:#ffcc00;
                    font-size:1.55rem;
                    font-weight:900;
                    letter-spacing:0.02em;
                    line-height:1.5;
                ">{display_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if no_trade_rsn and no_trade_rsn not in ("null", ""):
            st.warning(f"⚠️ **Reason:** {no_trade_rsn}")


def render_final_decision(synth: dict):
    decision   = synth.get("FINAL_DECISION", "WAIT")
    color      = VOTE_COLOR.get(decision, "gray")
    arrow      = VOTE_ICON.get(decision, "❓")
    strength   = synth.get("consensus_strength", "")
    tally      = synth.get("vote_tally", {})
    confidence = synth.get("confidence", 0)

    st.markdown("---")
    st.markdown("## 🏆 Final Consensus Decision")

    st.markdown(
        f"<div style='text-align:center;padding:24px 0 8px;'>"
        f"<span style='font-size:5rem;font-weight:900;color:{color};'>"
        f"{arrow} {decision}</span></div>"
        f"<p style='text-align:center;font-size:1.1rem;margin-top:0;'>"
        f"Consensus Strength: <strong>{strength}</strong> &nbsp;|&nbsp; "
        f"Confidence: <strong>{confidence}%</strong></p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("⬆️ UP", tally.get("UP", 0))
    c2.metric("⬇️ DOWN", tally.get("DOWN", 0))
    c3.metric("⏸️ WAIT", tally.get("WAIT", 0))

    st.success(f"✅ **Recommended Action:** {synth.get('action','')}")
    st.markdown(f"**Moderator Note:** {synth.get('moderator_note','')}")

    agreed    = synth.get("key_agreements", "")
    disagreed = synth.get("key_disagreements", "")
    if agreed:
        st.markdown(f"**Agreed on:** {agreed}")
    if disagreed and disagreed.lower() not in ("", "none"):
        st.warning(f"⚠️ **Disagreements:** {disagreed}")

    with st.expander("📋 Full Debate Transcript", expanded=False):
        st.code(synth.get("_transcript", ""), language="")


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📊 AI Trading Debate")
st.caption(
    "Five live AI models analyse your chart, debate each other, and deliver a "
    "consensus **UP / DOWN / WAIT** decision."
)

st.markdown("### Upload Chart")
uploaded = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "webp"],
    help="Screenshot of any trading chart (candlestick, line, Heikin-Ashi, etc.)",
)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded chart", use_container_width=True)

st.markdown("### Additional Context *(optional)*")
extra_ctx = st.text_area(
    "Paste any extra context you want the AIs to consider",
    placeholder="e.g. I'm watching BTC/USDT 10-min chart during the NY session…",
    height=80,
)

st.markdown("### Start the Debate")

col_btn, col_info = st.columns([2, 3])
with col_btn:
    run = st.button(
        "🚀 START AI DEBATE",
        disabled=uploaded is None,
        use_container_width=True,
        type="primary",
    )
with col_info:
    st.caption(
        "Pipeline: Gemini vision → Llama 3 / Mixtral / Gemma 2 / DeepSeek "
        "initial votes → Cross-Examination → Gemini final synthesis"
    )

if not run:
    st.stop()

if uploaded is None:
    st.error("Please upload a chart image first.")
    st.stop()

# ── Read image ────────────────────────────────────────────────────────────────
uploaded.seek(0)
image_bytes = uploaded.read()
ext = uploaded.name.rsplit(".", 1)[-1].lower()
mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}
mime_type = mime_map.get(ext, "image/jpeg")

if extra_ctx.strip():
    GEMINI_CHART_PROMPT_USED = (
        GEMINI_CHART_PROMPT
        + f"\n\nAdditional context from the user: {extra_ctx.strip()}"
    )
else:
    GEMINI_CHART_PROMPT_USED = GEMINI_CHART_PROMPT

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Gemini chart analysis + live news
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Step 1 — Gemini Chart Analysis & Live News")
with st.spinner("✨ Gemini 2.5 Flash is reading the chart and searching for live news…"):
    try:
        gemini_data = step1_gemini_analyze(image_bytes, mime_type)
        # inject extra context into prompt used
        if extra_ctx.strip() and "technical_summary" in gemini_data:
            gemini_data["_extra_ctx"] = extra_ctx.strip()
    except Exception as exc:
        st.error(f"Gemini error: {exc}")
        st.stop()

render_gemini_analysis(gemini_data)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Initial votes from all four text models
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Step 2 — Independent Initial Votes")

initial_votes: list[dict] = []

text_models = [
    ("groq",     "llama3-70b-8192",    "Llama 3 70B"),
    ("groq",     "mixtral-8x7b-32768", "Mixtral 8x7B"),
    ("groq",     "gemma2-9b-it",       "Gemma 2 9B"),
    ("deepseek", None,                  "DeepSeek-Chat"),
]

for backend, model_id, model_name in text_models:
    icon = MODEL_ICON.get(model_name, "🤖")
    with st.spinner(f"{icon} {model_name} is forming an independent opinion…"):
        try:
            if backend == "groq":
                result = groq_initial_vote(model_id, model_name, gemini_data)
            else:
                result = deepseek_initial_vote(gemini_data)
            result.setdefault("model", model_name)
            initial_votes.append(result)
            render_initial_vote(result)
        except Exception as exc:
            st.warning(f"{model_name} error (skipping): {exc}")
            initial_votes.append({
                "model": model_name,
                "analysis": f"API error: {exc}",
                "key_risks": [],
                "vote": "WAIT",
                "confidence": 0,
                "reasoning": "Unable to retrieve response.",
            })

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Cross-Examination / Debate Round
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Step 3 — Cross-Examination Debate Round")
st.caption("Each AI critiques the others and refines its position.")

debate_responses: list[dict] = []

for backend, model_id, model_name in text_models:
    icon = MODEL_ICON.get(model_name, "🤖")
    with st.spinner(f"{icon} {model_name} is cross-examining the other analysts…"):
        try:
            if backend == "groq":
                resp = groq_debate(model_id, model_name, initial_votes, gemini_data)
            else:
                resp = deepseek_debate(initial_votes, gemini_data)
            resp.setdefault("model", model_name)
            debate_responses.append(resp)
            render_debate_response(resp)
        except Exception as exc:
            st.warning(f"{model_name} debate error (skipping): {exc}")
            debate_responses.append({
                "model": model_name,
                "responses_to": [],
                "refined_vote": initial_votes[text_models.index((backend, model_id, model_name))].get("vote", "WAIT"),
                "refined_confidence": 0,
                "final_reasoning": f"Error: {exc}",
            })

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Gemini final synthesis
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Step 4 — Final Synthesis")

with st.spinner("✨ Gemini is moderating the debate and computing the final verdict…"):
    try:
        synthesis = step4_gemini_synthesize(gemini_data, initial_votes, debate_responses)
    except Exception as exc:
        st.error(f"Synthesis error: {exc}")
        st.stop()

render_final_decision(synthesis)
render_trade_recommendation(synthesis)
st.balloons()
