import streamlit as st
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# ================= CONFIG =================
LABEL_COLS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
MODEL_NAME = "roberta-base"
DEVICE = "cpu"  # HF Spaces free tier runs on CPU
MODEL_PATH = "best_roberta_emotion.pth"
THRESHOLDS_PATH = "optimal_thresholds.npy"
MAX_LEN = 128

# ================= MODEL DEFINITION =================
class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_labels=5):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]  # CLS token
        return self.classifier(self.dropout(pooled))

# ================= CACHED LOADING =================
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(THRESHOLDS_PATH):
        st.error(f"❌ Missing files: {MODEL_PATH} or {THRESHOLDS_PATH}. Please upload them to the Space.")
        st.stop()
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EmotionClassifier(MODEL_NAME)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        st.error(f"⚠️ Failed to load model weights: {e}")
        st.stop()
        
    model.to(DEVICE)
    model.eval()
    thresholds = np.load(THRESHOLDS_PATH)
    return tokenizer, model, thresholds

# ================= PREDICTION =================
def predict_emotions(text, tokenizer, model, thresholds):
    text = str(text).strip().lower()
    inputs = tokenizer(
        text, 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LEN, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        logits = model(inputs["input_ids"].to(DEVICE), inputs["attention_mask"].to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
    preds = (probs > thresholds).astype(int)
    return dict(zip(LABEL_COLS, preds)), probs

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Emotion Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Multi-Label Emotion Classifier")
st.markdown("""
Detect overlapping emotions in short English text.  
**Supported emotions:** 😡 Anger | 😨 Fear | 😊 Joy | 😢 Sadness | 😲 Surprise
""")

# Load model once
tokenizer, model, thresholds = load_pipeline()

# Input
user_input = st.text_area(
    "Enter your text here:",
    height=100,
    placeholder="e.g., I can't believe they announced the results already! I'm both thrilled and nervous."
)

# Prediction
if st.button("🔍 Analyze Emotions", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🤖 Processing..."):
            preds, probs = predict_emotions(user_input, tokenizer, model, thresholds)
            
        st.markdown("---")
        st.subheader("📊 Detected Emotions")
        
        # Display as tags
        cols = st.columns(5)
        for i, (emotion, detected) in enumerate(preds.items()):
            with cols[i]:
                if detected:
                    st.success(f"✅ {emotion.capitalize()}")
                else:
                    st.info(f"⬜ {emotion.capitalize()}")
                    
        st.markdown("---")
        st.subheader("📈 Confidence Scores")
        for emo, prob in zip(LABEL_COLS, probs):
            st.progress(float(prob), text=f"{emo.capitalize()}: {prob:.2%}")
            
        # Tip
        st.caption("💡 *Thresholds are optimized per-class for maximum Macro F1. Multiple emotions can be active simultaneously.*")