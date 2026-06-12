import streamlit as st
import os
import json
import re
import base64
import google.generativeai as genai
from groq import Groq

# Page Config
st.set_page_config(page_title="AI Trading Committee", layout="wide")
st.title("📈 AI Trading Committee — Live Debate")

# API Keys Initialization
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
DEEPSEEK_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")

# Clean reasoning tokens like <think>...</think> from Qwen/DeepSeek R1
def clean_json_response(text):
    try:
        # Strip thinking process
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Strip markdown json blocks
        cleaned = re.sub(r'```json\s*|\s*```', '', cleaned).strip()
        return json.loads(cleaned)
    except Exception as e:
        try:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        raise ValueError(f"Failed to parse JSON response. Raw output: {text[:200]}")

# Helper to encode image for Groq Vision
def encode_image_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# File Uploader
uploaded_file = st.file_uploader("Upload Market Chart Screenshot", type=["jpg", "jpeg", "png"])

if uploaded_file and st.button("START AI DEBATE"):
    image_bytes = uploaded_file.read()
    chart_context = ""
    vision_success = False

    st.subheader("Step 1 — Chart Analysis & Market News")

    # 1. TRY GEMINI VISION FIRST
    if GEMINI_KEY:
        try:
            with st.spinner("Analyzing chart with Gemini..."):
                genai.configure(api_key=GEMINI_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content([
                    "Analyze this trading chart. Identify key support, resistance, current trend, indicators (RSI/MACD if visible), and output a structured analysis text.",
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ])
                chart_context = response.text
                vision_success = True
                st.success("✅ Chart successfully analyzed by Gemini!")
        except Exception as e:
            st.warning(f"⚠️ Gemini Vision failed (Quota/Server Issue): {str(e)}")

    # 2. FALLBACK TO GROQ VISION IF GEMINI FAILS
    if not vision_success and GROQ_KEY:
        try:
            with st.spinner("Gemini busy. Falling back to Groq Llama-Vision..."):
                groq_client = Groq(api_key=GROQ_KEY)
                base64_image = encode_image_base64(image_bytes)

                response = groq_client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this trading chart layout. Extract asset trend, immediate support/resistance levels, and overall structure."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ]
                )
                chart_context = response.choices[0].message.content
                vision_success = True
                st.success("✅ Chart successfully analyzed by Groq Llama-Vision!")
        except Exception as e:
            st.error(f"❌ Groq