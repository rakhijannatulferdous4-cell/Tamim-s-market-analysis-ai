import streamlit as st
import json
import os
import time
from PIL import Image
import io

st.set_page_config(
    page_title="AI Trading Debate",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── API key ──────────────────────────────────────────────────────────────────
def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            st.error(
                "**GEMINI_API_KEY not found.**\n\n"
                "Add it to `.streamlit/secrets.toml`:\n```\nGEMINI_API_KEY = 'your-key'\n```\n"
                "or set it as an environment variable."
            )
            st.stop()
        return key

# ── Gemini client (cached across reruns) ─────────────────────────────────────
@st.cache_resource
def build_client():
    from google import genai
    return genai.Client(api_key=get_api_key())

# ── Persona definitions ───────────────────────────────────────────────────────
PERSONAS = [
    {
        "id": "chatgpt",
        "name": "ChatGPT (GPT-4o)",
        "emoji": "🤖",
        "style": (
            "You are simulating ChatGPT (GPT-4o) by OpenAI. "
            "Be analytical and structured. Emphasise support/resistance levels, "
            "moving averages, RSI, MACD, and volume. Provide probability estimates."
        ),
    },
    {
        "id": "claude",
        "name": "Claude (Anthropic)",
        "emoji": "🧠",
        "style": (
            "You are simulating Claude by Anthropic. "
            "Be cautious and nuanced. Stress risk management, multiple scenarios, "
            "and potential black-swan events. Prefer waiting when evidence is mixed."
        ),
    },
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "emoji": "🔬",
        "style": (
            "You are simulating DeepSeek. "
            "Be data-driven and quantitative. Focus on statistical chart patterns, "
            "Fibonacci levels, Elliott Wave, and mathematical price projections."
        ),
    },
    {
        "id": "llama3",
        "name": "Llama 3 (Meta)",
        "emoji": "🦙",
        "style": (
            "You are simulating Llama 3 by Meta. "
            "Be pragmatic and direct. Focus on market sentiment, trend momentum, "
            "news catalysts, and crowd psychology."
        ),
    },
]

DECISION_COLORS = {"UP": "green", "DOWN": "red", "WAIT": "orange"}
DECISION_ARROWS = {"UP": "⬆️", "DOWN": "⬇️", "WAIT": "⏸️"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a (possibly markdown-wrapped) response."""
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    # find first { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


def call_persona(client, persona: dict, image_bytes: bytes, mime: str,
                 news: str, previous: list[dict]) -> dict:
    from google.genai import types

    prior_txt = ""
    if previous:
        lines = [
            f"- {p['ai_name']}: {p['decision']} — {p['reasoning']}"
            for p in previous
        ]
        prior_txt = (
            "\n\nEarlier opinions from this debate panel:\n"
            + "\n".join(lines)
            + "\n\nConsider these but form your own independent view."
        )

    news_txt = f"\n\nLatest economic / market news provided by the user:\n{news}" if news.strip() else ""

    prompt = (
        f"{persona['style']}\n\n"
        "Analyse the provided trading chart image carefully."
        f"{news_txt}{prior_txt}\n\n"
        "Return ONLY valid JSON — no markdown, no extra text — in exactly this shape:\n"
        "{\n"
        '  "ai_name": "<your model name>",\n'
        '  "analysis": "<2-3 sentence technical + contextual analysis>",\n'
        '  "key_signals": ["<signal 1>", "<signal 2>", "<signal 3>"],\n'
        '  "risk_level": "<LOW|MEDIUM|HIGH>",\n'
        '  "confidence": <integer 0-100>,\n'
        '  "decision": "<UP|DOWN|WAIT>",\n'
        '  "reasoning": "<one-sentence final justification>"\n'
        "}\n\n"
        "decision MUST be exactly one of: UP, DOWN, WAIT."
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=types.Blob(mime_type=mime, data=image_bytes)
                ),
                types.Part(text=prompt),
            ],
        )
    ]

    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=contents
    )
    result = extract_json(resp.text)
    result.setdefault("ai_name", persona["name"])
    result.setdefault("decision", "WAIT")
    return result


def call_consensus(client, all_opinions: list[dict]) -> dict:
    from google.genai import types

    votes = [o["decision"] for o in all_opinions]
    tally = {v: votes.count(v) for v in ["UP", "DOWN", "WAIT"]}
    lines = "\n".join(
        f"- {o['ai_name']}: {o['decision']} (confidence {o.get('confidence', '?')}%) — {o['reasoning']}"
        for o in all_opinions
    )

    prompt = (
        "You are a senior trading analyst moderating an AI debate panel.\n\n"
        f"Panel votes: UP={tally['UP']}, DOWN={tally['DOWN']}, WAIT={tally['WAIT']}\n\n"
        f"Individual opinions:\n{lines}\n\n"
        "Synthesise a final consensus recommendation.\n"
        "Return ONLY valid JSON in exactly this shape:\n"
        "{\n"
        '  "final_decision": "<UP|DOWN|WAIT>",\n'
        '  "consensus_strength": "<STRONG|MODERATE|DIVIDED>",\n'
        '  "summary": "<2-3 sentence collective reasoning>",\n'
        '  "key_agreement": "<what most AIs agreed on>",\n'
        '  "key_disagreement": "<main disagreement, or \'None\'>",\n'
        '  "action_recommendation": "<one-sentence actionable advice>"\n'
        "}\n\n"
        "final_decision MUST be exactly one of: UP, DOWN, WAIT."
    )

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
    )
    result = extract_json(resp.text)
    result["vote_counts"] = tally
    return result

# ── Render helpers ────────────────────────────────────────────────────────────
def render_opinion_card(opinion: dict, persona: dict):
    decision = opinion.get("decision", "WAIT")
    color = DECISION_COLORS.get(decision, "gray")
    arrow = DECISION_ARROWS.get(decision, "❓")
    conf = opinion.get("confidence", "—")

    with st.expander(
        f"{persona['emoji']} **{persona['name']}** — "
        f":{color}[**{arrow} {decision}**] — confidence {conf}%",
        expanded=True,
    ):
        st.markdown(f"**Analysis:** {opinion.get('analysis', '')}")

        signals = opinion.get("key_signals", [])
        if signals:
            st.markdown("**Key signals:**")
            for s in signals:
                st.markdown(f"- {s}")

        col1, col2 = st.columns(2)
        col1.metric("Risk", opinion.get("risk_level", "—"))
        col2.metric("Confidence", f"{conf}%")

        st.info(f"💬 *{opinion.get('reasoning', '')}*")


def render_consensus(consensus: dict):
    decision = consensus.get("final_decision", "WAIT")
    strength = consensus.get("consensus_strength", "")
    color = DECISION_COLORS.get(decision, "gray")
    arrow = DECISION_ARROWS.get(decision, "❓")
    tally = consensus.get("vote_counts", {})

    st.markdown("---")
    st.markdown("## 🏆 Final Consensus")

    st.markdown(
        f"<h1 style='text-align:center;color:{color};font-size:4rem;'>"
        f"{arrow} {decision}</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;font-size:1.1rem;'>Consensus strength: "
        f"<strong>{strength}</strong></p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("⬆️ UP votes", tally.get("UP", 0))
    c2.metric("⬇️ DOWN votes", tally.get("DOWN", 0))
    c3.metric("⏸️ WAIT votes", tally.get("WAIT", 0))

    st.markdown(f"**Summary:** {consensus.get('summary', '')}")
    st.success(f"✅ **Action:** {consensus.get('action_recommendation', '')}")

    agreed = consensus.get("key_agreement", "")
    disagreed = consensus.get("key_disagreement", "None")
    if agreed:
        st.markdown(f"**Agreed on:** {agreed}")
    if disagreed and disagreed.lower() != "none":
        st.warning(f"⚠️ **Disagreement:** {disagreed}")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("📈 AI Trading Debate")
st.caption(
    "Upload a trading chart screenshot. Four AI models debate the chart and "
    "reach a consensus: **UP / DOWN / WAIT**."
)

st.markdown("### 1️⃣ Upload Chart")
uploaded = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg", "webp", "gif"],
    help="Screenshot of any trading chart (candlestick, line, etc.)",
)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded chart", use_container_width=True)

st.markdown("### 2️⃣ Economic News (optional)")
news_input = st.text_area(
    "Paste any relevant economic headlines or context",
    placeholder="e.g. Fed holds rates steady; CPI beats expectations; earnings season underway…",
    height=100,
)

st.markdown("### 3️⃣ Run the Debate")
start_btn = st.button(
    "🚀 START AI DEBATE",
    disabled=uploaded is None,
    use_container_width=True,
    type="primary",
)

if start_btn:
    if uploaded is None:
        st.error("Please upload a chart image first.")
        st.stop()

    # Read image bytes and MIME type
    uploaded.seek(0)
    image_bytes = uploaded.read()
    ext = uploaded.name.rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/jpeg")

    client = build_client()
    all_opinions: list[dict] = []

    st.markdown("---")
    st.markdown("## 🎙️ The Debate")

    for i, persona in enumerate(PERSONAS):
        with st.spinner(f"{persona['emoji']} {persona['name']} is analysing…"):
            try:
                opinion = call_persona(
                    client, persona, image_bytes, mime_type,
                    news_input, all_opinions
                )
                all_opinions.append(opinion)
                render_opinion_card(opinion, persona)
            except Exception as e:
                st.error(f"{persona['name']} error: {e}")
                all_opinions.append({
                    "ai_name": persona["name"],
                    "decision": "WAIT",
                    "reasoning": "Analysis failed.",
                    "confidence": 0,
                    "analysis": str(e),
                    "key_signals": [],
                    "risk_level": "HIGH",
                })

    # Consensus
    with st.spinner("🧮 Computing final consensus…"):
        try:
            consensus = call_consensus(client, all_opinions)
            render_consensus(consensus)
        except Exception as e:
            st.error(f"Consensus error: {e}")

    st.balloons()
