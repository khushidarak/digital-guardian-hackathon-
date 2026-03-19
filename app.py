"""
STEP 2 - THE STREAMLIT WEB APP
================================
Run this AFTER you have run train_model.py

How to run:
  streamlit run app.py

This app uses 3 AI models:
  1. ASR   (Automatic Speech Recognition) - converts audio to text
  2. NER   (Named Entity Recognition)     - finds names, phone numbers, money amounts
  3. Sentiment Analysis                   - detects scary/urgent language used by scammers
  + Your trained Random Forest            - gives final scam risk score
"""

import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Scam Call Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Scam Call Detector")
st.markdown("Upload a call recording — the AI will analyze it and give you a **risk score**.")
st.divider()

# ─────────────────────────────────────────────
# LOAD TRAINED MODELS (.pkl files)
# ─────────────────────────────────────────────
MODELS_FOLDER = "models"

@st.cache_resource
def load_ml_models():
    """Load the trained Random Forest model from .pkl files"""
    try:
        model   = joblib.load(os.path.join(MODELS_FOLDER, "scam_classifier.pkl"))
        le      = joblib.load(os.path.join(MODELS_FOLDER, "label_encoder.pkl"))
        scaler  = joblib.load(os.path.join(MODELS_FOLDER, "scaler.pkl"))
        return model, le, scaler, True
    except Exception as e:
        return None, None, None, False

# ─────────────────────────────────────────────
# MODEL 1: ASR - Speech to Text
# Converts audio recording → words
# ─────────────────────────────────────────────
@st.cache_resource
def load_asr_model():
    """Load Whisper ASR model (runs locally, no internet needed)"""
    try:
        import whisper
        model = whisper.load_model("base")  # small fast model
        return model, "whisper"
    except ImportError:
        pass

    try:
        import speech_recognition as sr
        return sr.Recognizer(), "google"
    except ImportError:
        return None, "none"

def transcribe_audio(audio_path, asr_model, asr_type):
    """Convert audio file to text"""
    try:
        if asr_type == "whisper":
            result = asr_model.transcribe(audio_path)
            return result["text"].strip()

        elif asr_type == "google":
            import speech_recognition as sr
            recognizer = asr_model
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)

        else:
            return "[ASR not available - install whisper or SpeechRecognition]"

    except Exception as e:
        return f"[Could not transcribe: {str(e)}]"


# ─────────────────────────────────────────────
# MODEL 2: SENTIMENT ANALYSIS
# Detects if the call uses scary/urgent language
# Scammers often say: "Act NOW!", "You'll be ARRESTED"
# ─────────────────────────────────────────────
@st.cache_resource
def load_sentiment_model():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer(), "vader"
    except ImportError:
        try:
            from transformers import pipeline
            return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"), "transformer"
        except ImportError:
            return None, "none"

def analyze_sentiment(text, sentiment_model, sentiment_type):
    """
    Returns:
      - sentiment label (POSITIVE / NEGATIVE)
      - score 0-1 (higher = more negative/threatening)
      - explanation
    """
    if not text or text.startswith("["):
        return "UNKNOWN", 0.5, "No transcript available"

    # Extra scam keywords (common in robocalls)
    scam_keywords = [
        "irs", "arrest", "warrant", "deport", "social security",
        "suspended", "legal action", "lawsuit", "fraud", "immediately",
        "urgent", "act now", "final notice", "court", "federal",
        "police", "crime", "victim", "gift card", "wire transfer",
        "bitcoin", "verify", "account", "overdue", "penalty",
        "medicare", "insurance", "refund", "claim", "prize", "won"
    ]

    text_lower = text.lower()
    keyword_hits = [kw for kw in scam_keywords if kw in text_lower]
    keyword_score = min(len(keyword_hits) / 5, 1.0)  # max out at 5 keywords

    if sentiment_type == "vader":
        scores = sentiment_model.polarity_scores(text)
        # compound: -1 = very negative, +1 = very positive
        # Scam calls are often very negative or urgent
        negativity = (1 - scores['compound']) / 2  # convert to 0-1
        combined = (negativity * 0.5) + (keyword_score * 0.5)
        label = "⚠️ THREATENING" if combined > 0.4 else "✅ NEUTRAL"
        explanation = f"Negativity: {negativity:.0%} | Scam keywords found: {keyword_hits[:5]}"
        return label, combined, explanation

    elif sentiment_type == "transformer":
        result = sentiment_model(text[:512])[0]
        score = result['score'] if result['label'] == 'NEGATIVE' else 1 - result['score']
        combined = (score * 0.5) + (keyword_score * 0.5)
        label = "⚠️ THREATENING" if combined > 0.4 else "✅ NEUTRAL"
        explanation = f"Sentiment: {result['label']} | Scam keywords found: {keyword_hits[:5]}"
        return label, combined, explanation

    else:
        # Fallback: keyword only
        label = "⚠️ SUSPICIOUS" if keyword_score > 0.3 else "✅ NEUTRAL"
        explanation = f"Scam keywords found: {keyword_hits[:5]}"
        return label, keyword_score, explanation


# ─────────────────────────────────────────────
# MODEL 3: NER - Named Entity Recognition
# Finds: names, phone numbers, money, organizations
# Scammers often mention: IRS, Social Security, large $$
# ─────────────────────────────────────────────
@st.cache_resource
def load_ner_model():
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp, "spacy"
    except Exception:
        try:
            from transformers import pipeline
            ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
                           aggregation_strategy="simple")
            return ner, "transformer"
        except Exception:
            return None, "none"

def extract_entities(text, ner_model, ner_type):
    """
    Find named entities in transcript
    Returns: list of (entity_text, entity_type) tuples + risk_score
    """
    if not text or text.startswith("["):
        return [], 0.0

    suspicious_orgs = ["irs", "fbi", "dea", "social security", "ssa",
                       "medicare", "federal", "government", "court", "police"]
    entities = []
    risk_score = 0.0

    if ner_type == "spacy":
        doc = ner_model(text)
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
            # Flag suspicious organizations
            if ent.label_ in ["ORG", "GPE"] and ent.text.lower() in suspicious_orgs:
                risk_score += 0.2
            if ent.label_ == "MONEY":
                risk_score += 0.15  # mentioning money = suspicious

    elif ner_type == "transformer":
        results = ner_model(text[:512])
        for r in results:
            entities.append((r['word'], r['entity_group']))
            if r['entity_group'] == "ORG" and r['word'].lower() in suspicious_orgs:
                risk_score += 0.2

    # Also check for phone number patterns and money manually
    import re
    phones = re.findall(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    for p in phones:
        entities.append((p, "PHONE"))
        risk_score += 0.1  # phone numbers in automated calls are suspicious

    money = re.findall(r'\$[\d,]+|\b\d+\s*dollars?\b', text, re.IGNORECASE)
    for m in money:
        entities.append((m, "MONEY"))
        risk_score += 0.1

    return entities, min(risk_score, 1.0)


# ─────────────────────────────────────────────
# AUDIO FEATURE EXTRACTION (for trained model)
# Same as in train_model.py
# ─────────────────────────────────────────────
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=10)
        features = []
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spec_centroid))
        features.append(np.std(spec_centroid))
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(rolloff))
        return np.array(features).reshape(1, -1)
    except Exception as e:
        return None


# ─────────────────────────────────────────────
# FINAL RISK SCORE CALCULATOR
# Combines all 3 models into one score
# ─────────────────────────────────────────────
def calculate_final_risk(ml_prob, sentiment_score, ner_score):
    """
    Weighted average of all signals:
    - 50% from trained ML model (audio features)
    - 30% from sentiment analysis (language tone)
    - 20% from NER (suspicious entities mentioned)
    """
    final = (ml_prob * 0.50) + (sentiment_score * 0.30) + (ner_score * 0.20)
    return round(min(final * 100, 100), 1)

def get_risk_label(score):
    if score >= 75:
        return "🔴 HIGH RISK - Likely Scam", "error"
    elif score >= 45:
        return "🟡 MEDIUM RISK - Suspicious", "warning"
    else:
        return "🟢 LOW RISK - Probably Safe", "success"


# ─────────────────────────────────────────────
# LOAD ALL MODELS AT STARTUP
# ─────────────────────────────────────────────
with st.spinner("Loading AI models..."):
    ml_model, le, scaler, ml_loaded = load_ml_models()
    asr_model, asr_type = load_asr_model()
    sentiment_model, sentiment_type = load_sentiment_model()
    ner_model, ner_type = load_ner_model()

# Status bar
col1, col2, col3, col4 = st.columns(4)
col1.metric("🤖 Scam Classifier", "✅ Ready" if ml_loaded else "⚠️ Not trained yet")
col2.metric("🎤 ASR (Speech→Text)", f"✅ {asr_type.upper()}" if asr_type != "none" else "❌ Not installed")
col3.metric("😠 Sentiment Analysis", f"✅ {sentiment_type.upper()}" if sentiment_type != "none" else "❌ Not installed")
col4.metric("🏷️ NER (Entities)", f"✅ {ner_type.upper()}" if ner_type != "none" else "❌ Not installed")

if not ml_loaded:
    st.warning("⚠️ The trained model is not found. Please run `python train_model.py` first!")

st.divider()

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload a call recording (.wav file)",
    type=["wav"],
    help="Upload any .wav audio file of a phone call"
)

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("🔬 Analyzing call... (this may take 30-60 seconds)"):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # ── MODEL 1: ASR ──────────────────────────
        st.markdown("### 🎤 Step 1: Converting speech to text...")
        transcript = transcribe_audio(tmp_path, asr_model, asr_type)
        st.text_area("📝 Transcript", transcript, height=120)

        # ── MODEL 2: SENTIMENT ────────────────────
        st.markdown("### 😠 Step 2: Analyzing language tone...")
        sent_label, sent_score, sent_explanation = analyze_sentiment(
            transcript, sentiment_model, sentiment_type
        )
        st.write(f"**Sentiment:** {sent_label}")
        st.write(f"*{sent_explanation}*")
        st.progress(sent_score, text=f"Threat level: {sent_score:.0%}")

        # ── MODEL 3: NER ──────────────────────────
        st.markdown("### 🏷️ Step 3: Finding suspicious names & entities...")
        entities, ner_score = extract_entities(transcript, ner_model, ner_type)
        if entities:
            entity_df = {"Entity": [e[0] for e in entities], "Type": [e[1] for e in entities]}
            import pandas as pd
            st.dataframe(pd.DataFrame(entity_df), use_container_width=True)
        else:
            st.write("No named entities found.")
        st.progress(ner_score, text=f"Entity risk: {ner_score:.0%}")

        # ── TRAINED ML MODEL ──────────────────────
        st.markdown("### 🤖 Step 4: Running trained scam classifier...")
        ml_prob = 0.5  # default if model not available
        ml_label = "Unknown"

        if ml_loaded:
            features = extract_features(tmp_path)
            if features is not None:
                features_scaled = scaler.transform(features)
                proba = ml_model.predict_proba(features_scaled)[0]
                pred_class = ml_model.predict(features_scaled)[0]
                ml_label = le.inverse_transform([pred_class])[0]

                # Find the probability for scam class
                classes = list(le.classes_)
                scam_classes = [c for c in classes if 'scam' in c.lower() or 'robocall' in c.lower() or 'spam' in c.lower()]
                if scam_classes:
                    scam_idx = classes.index(scam_classes[0])
                    ml_prob = proba[scam_idx]
                else:
                    ml_prob = max(proba)  # use highest probability

                st.write(f"**Predicted class:** `{ml_label}`")
                st.progress(float(ml_prob), text=f"ML confidence: {ml_prob:.0%}")
            else:
                st.warning("Could not extract audio features")
        else:
            st.warning("Trained model not loaded - skipping ML step")

        # ── FINAL RESULT ──────────────────────────
        st.divider()
        st.markdown("## 🎯 Final Analysis")

        final_score = calculate_final_risk(ml_prob, sent_score, ner_score)
        risk_label, risk_type = get_risk_label(final_score)

        # Big score display
        col_score, col_detail = st.columns([1, 2])

        with col_score:
            if risk_type == "error":
                st.error(f"**Risk Score: {final_score}/100**\n\n{risk_label}")
            elif risk_type == "warning":
                st.warning(f"**Risk Score: {final_score}/100**\n\n{risk_label}")
            else:
                st.success(f"**Risk Score: {final_score}/100**\n\n{risk_label}")

        with col_detail:
            st.markdown("**Score Breakdown:**")
            st.write(f"🤖 ML Model (50% weight): `{ml_prob:.0%}`")
            st.write(f"😠 Sentiment (30% weight): `{sent_score:.0%}`")
            st.write(f"🏷️ NER Entities (20% weight): `{ner_score:.0%}`")

        # Security Warnings
        st.markdown("### ⚠️ Security Warnings")
        warnings_list = []

        if final_score >= 75:
            warnings_list.append("🔴 **DO NOT** provide any personal information")
            warnings_list.append("🔴 **DO NOT** send money, gift cards, or wire transfers")
            warnings_list.append("🔴 **Hang up immediately** and block the number")
            warnings_list.append("🔴 Report to FTC at **reportfraud.ftc.gov**")
        elif final_score >= 45:
            warnings_list.append("🟡 Be cautious — verify caller identity independently")
            warnings_list.append("🟡 Do not give out Social Security or bank account numbers")
            warnings_list.append("🟡 Call back using official numbers from the organization's website")
        else:
            warnings_list.append("🟢 Call appears safe, but always verify unknown callers")
            warnings_list.append("🟢 Never share sensitive info with unexpected callers")

        for w in warnings_list:
            st.markdown(w)

        # Cleanup temp file
        os.unlink(tmp_path)

st.divider()
st.caption("Built with ASR + Sentiment Analysis + NER | Trained on Robocall Audio Dataset (WSPR-NCSU)")