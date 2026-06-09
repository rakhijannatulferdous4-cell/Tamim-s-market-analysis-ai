"""
AI Trading Debate — 5 live AI models, 3 real APIs.

Pipeline
────────
Step 1  Gemini 2.5 Flash   — reads chart image + fetches live market news
Step 2  Llama 3 70B        — independent vote via Groq
        Mixtral 8x7B       — independent vote via Groq
        Gemma 2 9B         — independent vote via Groq
        DeepSeek-Chat      — independent vote via DeepSeek API
Step 3  Gemini 2.5 Flash   — receives all 5 opinions, runs cross-examination,
                             delivers FINAL_DECISION + candle recommendation
"""

import json
import os

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
# Secret loader — tries st.secrets first, then Replit env var
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
# JSON extractor
# ─────────────────────────────────────────────────────────────────────────────
def parse_json(text: str) -> dict:
    raw = text or ""
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start < 0:
        raise ValueError(f"No JSON found in model response:\n{text[:500]}")
    return json.loads(raw[start:end])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Gemini: chart vision + live news
# ─────────────────────────────────────────────────────────────────────────────
def step1_gemini_analyze(image_bytes: bytes, mime: str, extra_ctx: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_secret("GEMINI_API_KEY"))

    extra = f"\n\nExtra context from user: {extra_ctx.strip()}" if extra_ctx.strip() else ""

    prompt = (
        "You are a senior technical analyst and financial journalist.\n\n"
        "Analyse the trading chart image in full detail, then use Google Search "
        "to find the latest real-time market news for the asset shown." + extra + "\n\n"
        "Return ONLY valid JSON — no markdown, no extra text:\n"
        "{\n"
        '  "asset": "<ticker/name>",\n'
        '  "timeframe": "<e.g. 10-min, 1H>",\n'
        '  "timeframe_minutes": <integer minutes per candle>,\n'
        '  "current_price": "<price on chart>",\n'
        '  "trend": "<Bullish|Bearish|Sideways>",\n'
        '  "support": ["<level>", "<level>"],\n'
        '  "resistance": ["<level>", "<level>"],\n'
        '  "indicators": {"<name>": "<reading>"},\n'
        '  "patterns": ["<pattern>"],\n'
        '  "technical_summary": "<3-4 sentence analysis>",\n'
        '  "live_news": [\n'
        '    {"headline": "<text>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<src>"},\n'
        '    {"headline": "<text>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<src>"},\n'
        '    {"headline": "<text>", "sentiment": "<Bullish|Bearish|Neutral>", "source": "<src>"}\n'
        "  ],\n"
        '  "news_summary": "<2-sentence sentiment summary>",\n'
        '  "gemini_vote": "<UP|DOWN|WAIT>",\n'
        '  "gemini_confidence": <0-100>,\n'
        '  "gemini_reasoning": "<one concise sentence>"\n'
        "}"
    )

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

    return parse_json(resp.text)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Groq: Llama 3, Mixtral, Gemma 2
# ─────────────────────────────────────────────────────────────────────────────
def build_analyst_prompt(model_name: str, g: dict) -> str:
    news_lines = "\n".join(
        f"  • [{n.get('sentiment','?')}] {n.get('headline','')} — {n.get('source','')}"
        for n in g.get("live_news", [])
    ) or "  No live news retrieved."

    indicators = "; ".join(
        f"{k}: {v}" for k, v in g.get("indicators", {}).items()
    ) or "N/A"

    return (
        f"You are {model_name}, an expert AI trading analyst.\n\n"
        "Gemini has analysed a trading chart and fetched live market news. "
        "Study its findings and give your own independent trading opinion.\n\n"
        "=== GEMINI CHART ANALYSIS ===\n"
        f"Asset       : {g.get('asset','?')}\n"
        f"Timeframe   : {g.get('timeframe','?')}\n"
        f"Price       : {g.get('current_price','?')}\n"
        f"Trend       : {g.get('trend','?')}\n"
        f"Support     : {', '.join(g.get('support', []))}\n"
        f"Resistance  : {', '.join(g.get('resistance', []))}\n"
        f"Indicators  : {indicators}\n"
        f"Patterns    : {', '.join(g.get('patterns', []))}\n"
        f"Technical   : {g.get('technical_summary','')}\n"
        f"Gemini Vote : {g.get('gemini_vote','?')} ({g.get('gemini_confidence',0)}%)\n"
        f"Gemini Says : {g.get('gemini_reasoning','')}\n\n"
        "=== LIVE MARKET NEWS ===\n"
        f"{news_lines}\n"
        f"Summary: {g.get('news_summary','')}\n"
        "=========================\n\n"
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


def groq_vote(model_id: str, model_name: str, gemini_data: dict) -> dict:
    from groq import Groq

    client = Groq(api_key=require_secret("GROQ_API_KEY"))
    prompt = build_analyst_prompt(model_name, gemini_data)

    chat = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are an expert AI trading analyst. Always respond with valid JSON only."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=600,
    )
    result = parse_json(chat.choices[0].message.content)
    result.setdefault("model", model_name)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DeepSeek
# ─────────────────────────────────────────────────────────────────────────────
def deepseek_vote(gemini_data: dict) -> dict:
    api_key = require_secret("DEEPSEEK_API_KEY")
    prompt  = build_analyst_prompt("DeepSeek-Chat", gemini_data)

    resp = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an expert AI trading analyst. Always respond with valid JSON only."},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 600,
        },
        timeout=40,
    )
    resp.raise_for_status()
    result = parse_json(resp.json()["choices"][0]["message"]["content"])
    result.setdefault("model", "DeepSeek-Chat")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Gemini final synthesis (cross-examination + verdict)
# ─────────────────────────────────────────────────────────────────────────────
def step3_gemini_synthesize(gemini_data: dict, analyst_votes: list[dict]) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_secret("GEMINI_API_KEY"))

    tf          = gemini_data.get("timeframe", "unknown")
    tf_minutes  = gemini_data.get("timeframe_minutes", 0)

    # Build positions block (no f-string interpolation into user content to avoid format issues)
    positions = "Gemini 2.5 Flash: " + gemini_data.get("gemini_vote", "WAIT")
    positions += f" ({gemini_data.get('gemini_confidence', 0)}%) — "
    positions += gemini_data.get("gemini_reasoning", "") + "\n"
    for v in analyst_votes:
        positions += (
            f"{v.get('model','?')}: {v.get('vote','WAIT')} "
            f"({v.get('confidence',0)}%) — {v.get('reasoning','')}\n"
            f"  Analysis: {v.get('analysis','')}\n"
            f"  Key Risks: {', '.join(v.get('key_risks', []))}\n"
        )

    prompt = (
        "You are the debate moderator and final decision-maker for a panel of "
        "5 AI trading analysts.\n\n"
        "=== ALL 5 ANALYST POSITIONS ===\n"
        + positions +
        "================================\n\n"
        "Your tasks:\n"
        "1. Cross-examine the positions — identify where models agree, disagree, "
        "and why.\n"
        "2. Deliver a FINAL_DECISION (UP, DOWN, or WAIT) based on the weight of evidence.\n"
        "3. Compute a RECOMMENDED ACTION based on the chart timeframe:\n"
        "   Chart timeframe: " + tf + " (" + str(tf_minutes) + " minutes per candle)\n"
        "   - If FINAL_DECISION is UP or DOWN:\n"
        "       * Estimate candle_count (1-5) for the expected move duration.\n"
        "       * total_duration_minutes = candle_count × timeframe_minutes.\n"
        "       * display_text format exactly:\n"
        '         "TRADE DIRECTION: UP|DOWN | TARGET: Next N candles will go UP|DOWN '
        '(Duration: X minutes on a Y-min chart)"\n'
        "   - If FINAL_DECISION is WAIT:\n"
        "       * display_text format exactly:\n"
        '         "DON\'T TRADE: <specific reason with asset and timeframe>"\n\n'
        "Return ONLY valid JSON — no markdown:\n"
        "{\n"
        '  "cross_examination": "<3-4 sentence moderator analysis of agreements and disagreements>",\n'
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

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
    )
    return parse_json(resp.text)


# ─────────────────────────────────────────────────────────────────────────────
# UI constants
# ─────────────────────────────────────────────────────────────────────────────
VOTE_COLOR = {"UP": "green", "DOWN": "red", "WAIT": "orange"}
VOTE_ICON  = {"UP": "⬆️",   "DOWN": "⬇️", "WAIT": "⏸️"}
MODEL_ICON = {
    "Gemini 2.5 Flash": "✨",
    "Llama 3 70B":      "🦙",
    "Mixtral 8x7B":     "⚗️",
    "Gemma 2 9B":       "💎",
    "DeepSeek-Chat":    "🔭",
}


def badge(vote: str) -> str:
    c = VOTE_COLOR.get(vote, "gray")
    i = VOTE_ICON.get(vote, "❓")
    return f":{c}[**{i} {vote}**]"


# ─────────────────────────────────────────────────────────────────────────────
# Render: Gemini chart analysis
# ─────────────────────────────────────────────────────────────────────────────
def render_gemini_analysis(g: dict):
    with st.expander("✨ Gemini 2.5 Flash — Chart Analysis + Live News", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Asset",     g.get("asset", "—"))
        c2.metric("Timeframe", g.get("timeframe", "—"))
        c3.metric("Trend",     g.get("trend", "—"))
        v = g.get("gemini_vote", "WAIT")
        c4.metric("Gemini Vote", f"{VOTE_ICON.get(v,'')} {v}",
                  f"{g.get('gemini_confidence',0)}% confidence")

        st.markdown("**Technical Summary**")
        st.info(g.get("technical_summary", ""))

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
            for k, v in inds.items():
                st.markdown(f"- **{k}**: {v}")

        if g.get("patterns"):
            st.markdown("**Patterns:** " + " · ".join(f"`{p}`" for p in g["patterns"]))

        st.markdown("---")
        st.markdown("**📰 Live Market News**")
        for item in g.get("live_news", []):
            sent = item.get("sentiment", "Neutral")
            dot  = "🟢" if sent == "Bullish" else ("🔴" if sent == "Bearish" else "🟡")
            st.markdown(f"{dot} **{item.get('headline','')}**  \n*{item.get('source','')}*")

        st.markdown(f"**News Summary:** {g.get('news_summary','')}")
        st.markdown(f"💬 *{g.get('gemini_reasoning','')}*")


# ─────────────────────────────────────────────────────────────────────────────
# Render: individual analyst vote card
# ─────────────────────────────────────────────────────────────────────────────
def render_vote_card(v: dict):
    model = v.get("model", "Unknown")
    vote  = v.get("vote", "WAIT")
    icon  = MODEL_ICON.get(model, "🤖")
    with st.expander(
        f"{icon} **{model}** — {badge(vote)} — {v.get('confidence', 0)}% confidence",
        expanded=True,
    ):
        st.markdown(f"**Analysis:** {v.get('analysis', '')}")
        risks = v.get("key_risks", [])
        if risks:
            st.markdown("**Key Risks:** " + " · ".join(f"`{r}`" for r in risks))
        st.markdown(f"💬 *{v.get('reasoning', '')}*")


# ─────────────────────────────────────────────────────────────────────────────
# Render: vote summary scoreboard
# ─────────────────────────────────────────────────────────────────────────────
def render_vote_scoreboard(gemini_data: dict, analyst_votes: list[dict]):
    st.markdown("### 🗳️ All 5 AI Votes at a Glance")

    all_votes = [
        {
            "model": "✨ Gemini 2.5 Flash",
            "vote":  gemini_data.get("gemini_vote", "WAIT"),
            "conf":  gemini_data.get("gemini_confidence", 0),
        }
    ] + [
        {
            "model": MODEL_ICON.get(v.get("model",""), "🤖") + " " + v.get("model","?"),
            "vote":  v.get("vote", "WAIT"),
            "conf":  v.get("confidence", 0),
        }
        for v in analyst_votes
    ]

    cols = st.columns(len(all_votes))
    for col, entry in zip(cols, all_votes):
        vote  = entry["vote"]
        color = VOTE_COLOR.get(vote, "gray")
        icon  = VOTE_ICON.get(vote, "❓")
        col.markdown(
            f"<div style='text-align:center;padding:12px 4px;border:1px solid #444;"
            f"border-radius:8px;'>"
            f"<div style='font-size:0.72rem;color:#aaa;margin-bottom:4px;'>{entry['model']}</div>"
            f"<div style='font-size:1.6rem;'>{icon}</div>"
            f"<div style='font-size:1rem;font-weight:800;color:{color};'>{vote}</div>"
            f"<div style='font-size:0.72rem;color:#aaa;'>{entry['conf']}% conf</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Render: final decision
# ─────────────────────────────────────────────────────────────────────────────
def render_final_decision(synth: dict):
    decision  = synth.get("FINAL_DECISION", "WAIT")
    color     = VOTE_COLOR.get(decision, "gray")
    arrow     = VOTE_ICON.get(decision, "❓")
    strength  = synth.get("consensus_strength", "")
    tally     = synth.get("vote_tally", {})
    conf      = synth.get("confidence", 0)

    st.markdown("---")
    st.markdown("## 🏆 Final Consensus Decision")

    st.markdown(
        f"<div style='text-align:center;padding:20px 0 6px;'>"
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

    st.markdown(f"**Cross-Examination Summary:** {synth.get('cross_examination','')}")
    st.info(f"📋 **Moderator Note:** {synth.get('moderator_note','')}")


# ─────────────────────────────────────────────────────────────────────────────
# Render: trade recommendation box
# ─────────────────────────────────────────────────────────────────────────────
def render_trade_recommendation(synth: dict):
    ra = synth.get("recommended_action", {})
    if not ra:
        return

    should_trade  = ra.get("should_trade", False)
    display_text  = ra.get("display_text", "")
    direction     = ra.get("trade_direction") or ""
    candles       = ra.get("candle_count")
    total_mins    = ra.get("total_duration_minutes")
    no_trade_rsn  = ra.get("dont_trade_reason") or ""

    st.markdown("---")
    st.markdown("## 🎯 Recommended Action")

    if should_trade and direction in ("UP", "DOWN"):
        bg     = "#062e0f" if direction == "UP" else "#2e0606"
        txt    = "#00e676" if direction == "UP" else "#ff5252"
        border = "#00c853" if direction == "UP" else "#d50000"
        arrow  = "⬆️"     if direction == "UP" else "⬇️"

        st.markdown(
            f"<div style='background:{bg};border:3px solid {border};border-radius:14px;"
            f"padding:30px 24px;margin:10px 0;text-align:center;'>"
            f"<div style='font-size:2.4rem;'>{arrow}</div>"
            f"<div style='color:{txt};font-size:1.5rem;font-weight:900;"
            f"letter-spacing:.02em;line-height:1.55;margin-top:8px;'>"
            f"{display_text}</div></div>",
            unsafe_allow_html=True,
        )

        if candles and total_mins:
            m1, m2, m3 = st.columns(3)
            m1.metric("Direction",     f"{arrow} {direction}")
            m2.metric("Candles to Hold", f"{candles}")
            m3.metric("Est. Duration",   f"{total_mins} min")

    else:
        st.markdown(
            f"<div style='background:#2b1e00;border:3px solid #ff8f00;border-radius:14px;"
            f"padding:30px 24px;margin:10px 0;text-align:center;'>"
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
    "Gemini reads your chart + live news → 4 models vote independently → "
    "Gemini cross-examines all 5 opinions → **FINAL DECISION + candle target**"
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

# Read image bytes
uploaded.seek(0)
image_bytes = uploaded.read()
ext      = uploaded.name.rsplit(".", 1)[-1].lower()
mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp"}
mime_type = mime_map.get(ext, "image/jpeg")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Gemini chart analysis + live news
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Step 1 — Gemini Chart Analysis & Live News")
with st.spinner("✨ Gemini is reading the chart and fetching live news…"):
    try:
        gemini_data = step1_gemini_analyze(image_bytes, mime_type, extra_ctx)
    except Exception as exc:
        st.error(f"Gemini Step 1 error: {exc}")
        st.stop()

render_gemini_analysis(gemini_data)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Independent votes from Llama 3, Mixtral, Gemma 2, DeepSeek
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Step 2 — Independent Analyst Votes")

GROQ_MODELS = [
    ("llama3-70b-8192",    "Llama 3 70B"),
    ("mixtral-8x7b-32768", "Mixtral 8x7B"),
    ("gemma2-9b-it",       "Gemma 2 9B"),
]

analyst_votes: list[dict] = []

for model_id, model_name in GROQ_MODELS:
    icon = MODEL_ICON.get(model_name, "🤖")
    with st.spinner(f"{icon} {model_name} (Groq) is forming its opinion…"):
        try:
            result = groq_vote(model_id, model_name, gemini_data)
            analyst_votes.append(result)
            render_vote_card(result)
        except Exception as exc:
            st.warning(f"{model_name} error: {exc}")
            analyst_votes.append({
                "model": model_name, "analysis": f"Error: {exc}",
                "key_risks": [], "vote": "WAIT", "confidence": 0,
                "reasoning": "API call failed.",
            })

with st.spinner("🔭 DeepSeek-Chat is forming its opinion…"):
    try:
        ds = deepseek_vote(gemini_data)
        analyst_votes.append(ds)
        render_vote_card(ds)
    except Exception as exc:
        st.warning(f"DeepSeek error: {exc}")
        analyst_votes.append({
            "model": "DeepSeek-Chat", "analysis": f"Error: {exc}",
            "key_risks": [], "vote": "WAIT", "confidence": 0,
            "reasoning": "API call failed.",
        })

# Scoreboard — all 5 votes at a glance
render_vote_scoreboard(gemini_data, analyst_votes)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Gemini cross-examination + final synthesis
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Step 3 — Gemini Cross-Examination & Final Verdict")
with st.spinner("✨ Gemini is cross-examining all 5 opinions and computing the final verdict…"):
    try:
        synthesis = step3_gemini_synthesize(gemini_data, analyst_votes)
    except Exception as exc:
        st.error(f"Step 3 synthesis error: {exc}")
        st.stop()

render_final_decision(synthesis)
render_trade_recommendation(synthesis)
st.balloons()
