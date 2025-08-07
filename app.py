import os
from dotenv import load_dotenv
import streamlit as st
import requests

# Replace with your actual Hugging Face API token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

if not HF_TOKEN:
    st.error("❌ Hugging Face API token is missing. Please check your .env file.")
    st.stop()

# Choose a hosted model (Zephyr is free, fast, and high quality)
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Prompt Generator: Wrap the users question into a prompt to give the LLM context and tone
def create_prompt(user_question):
    return f"""
You are a helpful, friendly AI assistant. Be concise and clear.
Tone: Friendly, concise, and professional.
Format: Use short paragraphs. Bullet points when listing steps.
Role: Act like a helpful, smart support agent. Do not say you're an AI.

User: {user_question}
Assistant:"""

# Call the Hugging Face Inference API
def query_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 200
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        output = response.json()
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"].split("Assistant:")[-1].strip()
        else:
            return "⚠️ Unexpected response format from model."
    else:
        return f"❌ API Error {response.status_code}: {response.text}"

# Streamlit UI
st.set_page_config(page_title="AI Prompt Assistant", layout="centered")
st.title("🧠 AI Prompt Assistant")
st.write("Ask me anything — powered by Hugging Face's hosted models.")

user_input = st.text_area("Enter your question:", height=150)

if st.button("Generate Response"):
    if user_input.strip() == "":
        st.warning("Please enter a question before clicking.")
    else:
        with st.spinner("Generating response..."):
            prompt = create_prompt(user_input)
            answer = query_llm(prompt)
            st.success(answer)
