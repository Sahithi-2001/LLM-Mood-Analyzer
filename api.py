# api.py

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from transformers import pipeline

# ✅ MUST be first Streamlit command
st.set_page_config(page_title="🧠 LLM Mood Analyzer", layout="centered")

# ⏳ Display loading message before loading model
st.write("⏳ Loading Hugging Face model...")

@st.cache_resource
def get_emotion_pipeline():
    return pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")

emotion_classifier = get_emotion_pipeline()
st.success("✅ Model loaded successfully!")

def get_motivational_message(mood):
    return {
        "Joy": "Keep shining! It's wonderful to feel good.",
        "Sadness": "It’s okay to feel down. Better days are ahead.",
        "Anger": "Take a deep breath. Peace is stronger than frustration.",
        "Fear": "Courage grows by facing fears. You’ve got this.",
        "Love": "Stay connected. Love is your strength.",
        "Surprise": "Embrace the unexpected. Growth lives there."
    }.get(mood, "Thanks for reflecting.")

def analyze_with_llm(entry):
    try:
        result = emotion_classifier(entry)[0]
        label = result['label']
        score = result['score']
        mood = label.title()
        summary = f"Detected emotion: {label} ({score:.2f} confidence)"
        response = get_motivational_message(mood)
        return {"summary": summary, "mood": mood, "response": response}
    except Exception as e:
        return {"summary": "Could not analyze mood.", "mood": "Unknown", "response": str(e)}

def save_entry(date, entry, mood):
    log_file = "journal_log.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame(columns=["Date", "Entry", "Mood"])
    df = pd.concat([df, pd.DataFrame([[date, entry, mood]], columns=["Date", "Entry", "Mood"])], ignore_index=True)
    df.to_csv(log_file, index=False)

# -------------------- UI ------------------------

st.title("🧠 Daily Journal Mood Analyzer & Tracker (LLM-Powered)")

entry = st.text_area("Write your journal entry below:", height=200)

if st.button("Analyze"):
    if not entry.strip():
        st.warning("Please write something.")
    else:
        result = analyze_with_llm(entry)
        save_entry(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), entry, result["mood"])

        st.subheader("📝 Summary")
        st.write(result["summary"])
        st.subheader("😊 Detected Mood")
        st.write(result["mood"])
        st.subheader("💬 Motivational Message")
        st.success(result["response"])

log_file = "journal_log.csv"
if os.path.exists(log_file):
    st.markdown("## 📈 Mood Tracker History")
    df_log = pd.read_csv(log_file)
    if not df_log.empty:
        mood_counts = df_log["Mood"].value_counts()
        st.bar_chart(mood_counts)
