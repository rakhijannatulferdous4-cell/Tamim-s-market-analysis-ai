import streamlit as st
import os
import json
import re
import base64
from google import genai
from google.genai import types
from groq import Groq

# Page Config with Dark Premium Theme Styling
st.set_page_config(page_title="AI Trading Committee", layout="wide")

# Custom CSS for Premium Neon/Dark Theme & Glow Effects
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    h1 { color: #00ffcc; text-shadow: 0 0 10px rgba(0,255,204,0.3); font-family: 'Courier New', monospace; }
    .stButton>button {
        background: linear-gradient(45deg, #00ffcc, #0077ff);
        color: black !important;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        box-shadow: 0 4px 15px rgba(0,255,204,0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,255,204,0.6);
    }
    .analyst-card {
        background: #1a1c23;
        border: 1px solid #2d313f;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }
    .analyst-card:hover { transform: translateY(-3px); border-color: #00ffcc; }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Trading Committee — Live Debate")

def get_secret(name):
    """Read a key from Streamlit secrets first, then the OS environment."""
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.environ.get(name, "") or "")


# API keys stay server-side and are never placed in the page or URL.
GEMINI_KEY = get_secret("GEMINI_API_KEY")
GROQ_KEY = get_secret("GROQ_API_KEY")
DEEPSEEK_KEY = get_secret("DEEPSEEK_API_KEY")

def clean_json_response(text):
    try:
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = re.sub(r"```json\s*|```", "", cleaned).strip()
        return json.loads(cleaned)
    except:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match: return json.loads(match.group(0))
        except: pass
        return {"vote": "WAIT", "confidence": "50%", "reason": "Failed to parse clean JSON"}

def encode_image_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

uploaded_file = st.file_uploader("Upload Market Chart Screenshot", type=["jpg", "jpeg", "png"])

if uploaded_file and st.button("🚀 START AI DEBATE"):
    image_bytes = uploaded_file.read()
    chart_context = ""
    vision_success = False

    st.subheader("Step 1 — Chart Analysis & Market News")

    # 1. BULLETPROOF GEMINI CALL
    if GEMINI_KEY:
        try:
            with st.spinner("Analyzing chart with Gemini..."):
                client = genai.Client(api_key=GEMINI_KEY)
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes, mime_type="image/jpeg"
                        ),
                        "Analyze this trading chart layout. Identify support, resistance, and current macro trend.",
                    ],
                )
                chart_context = response.text
                if chart_context:
                    vision_success = True
                    st.success("✅ Chart successfully analyzed by Gemini!")
        except Exception as e:
            st.warning(f"⚠️ Gemini Vision skipped: {str(e)}")

    # 2. PROPER BASE64 GROQ FALLBACK
    if not vision_success and GROQ_KEY:
        try:
            with st.spinner("Gemini offline. Switching to Groq Llama-Vision..."):
                groq_client = Groq(api_key=GROQ_KEY)
                base64_image = encode_image_base64(image_bytes)

                response = groq_client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this trading chart structure. Determine the market trend, support, and resistance."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ]
                )
                chart_context = response.choices[0].message.content
                vision_success = True
                st.success("✅ Chart successfully analyzed by Groq Llama-Vision!")
        except Exception as e:
            st.error(f"❌ Groq Vision also failed: {str(e)}")

    if not vision_success:
        chart_context = "High-volatility setup. Dynamic visual data unavailable. Relying on baseline algorithmic sentiment."
        st.info("🚨 Safe text-fallback active.")

    st.text_area("Extracted Market Data Context", value=chart_context, height=120)

    # --- DEBATE MATRIX ---
    st.subheader("AI Analyst Votes & Internal Debate")

    models_to_use = [("Gemini", "gemini"), ("Llama 3.3", "groq:llama-3.3-70b-versatile"), ("Qwen 2.5", "groq:qwen-2.5-32b")]
    votes = []

    prompt_template = f"""
    Analyze market context: "{chart_context}". Provide decision matching this JSON schema:
    {{"vote": "UP" or "DOWN" or "WAIT", "confidence": "0-100%", "reason": "One line technical summary"}}
    Return ONLY valid raw JSON.
    """

    cols = st.columns(3)
    for idx, (name, provider) in enumerate(models_to_use):
        with cols[idx]:
            st.markdown(f'<div class="analyst-card"><h3>{name} Analyst</h3>', unsafe_allow_html=True)
            vote_data = None
            try:
                if provider == "gemini" and vision_success:
                    client = genai.Client(api_key=GEMINI_KEY)
                    res = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt_template,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            max_output_tokens=8192,
                        ),
                    )
                    vote_data = clean_json_response(res.text)
                elif provider.startswith("groq:"):
                    m_id = provider.split(":")[1]
                    client = Groq(api_key=GROQ_KEY)
                    res = client.chat.completions.create(model=m_id, messages=[{"role": "user", "content": prompt_template}])
                    vote_data = clean_json_response(res.choices[0].message.content)
            except:
                st.write("🔴 Status: Temporarily Offline")

            if vote_data:
                votes.append(vote_data)
                color = "🟩" if vote_data['vote'] == "UP" else "🟥" if vote_data['vote'] == "DOWN" else "🟨"
                st.markdown(f"## {color} {vote_data['vote']}")
                st.markdown(f"**Confidence:** {vote_data['confidence']}")
                st.caption(f"**Reason:** {vote_data['reason']}")
            st.markdown('</div>', unsafe_allow_html=True)

    # Execution Summary
    st.markdown("---")
    if votes:
        up = sum(1 for v in votes if v['vote'] == "UP")
        down = sum(1 for v in votes if v['vote'] == "DOWN")
        if up > down: st.success(f"### 🚀 FINAL RECOMMENDATION: CALL (UP)")
        elif down > up: st.error(f"### 📉 FINAL RECOMMENDATION: PUT (DOWN)")
        else: st.warning(f"### ⏳ FINAL RECOMMENDATION: NO TRADE (WAIT)")