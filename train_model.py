"""
STEP 1 - TRAIN THE AI MODEL (Fixed for robocall-audio-dataset)
===============================================================
Run this file FIRST before opening the app.
  python train_model.py
"""

import os
import numpy as np
import pandas as pd
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

AUDIO_FOLDER  = "audio-wav-16khz"
METADATA_FILE = "metadata.csv"
MODELS_FOLDER = "models"
SAMPLE_RATE   = 16000
DURATION      = 10

def create_label_from_case(case_text):
    if pd.isna(case_text) or str(case_text).strip() == "":
        return "scam"
    text = str(case_text).lower()
    scam_keywords = [
        "scam", "fraud", "illegal", "robocall", "spoofing", "ftc",
        "do not call", "violation", "deceptive", "misleading", "fake",
        "impersonat", "irs", "social security", "medicare", "warrant",
        "arrest", "lawsuit", "penalty", "debt", "loan", "prize",
        "won", "lottery", "gift card", "wire transfer", "bitcoin",
        "unauthorized", "complaint", "enforcement", "cease", "prohibited"
    ]
    legit_keywords = [
        "legitimate", "authorized", "consent", "opt-in", "approved",
        "healthcare provider", "pharmacy", "bank notification",
        "appointment reminder", "school", "weather alert"
    ]
    scam_score  = sum(1 for kw in scam_keywords  if kw in text)
    legit_score = sum(1 for kw in legit_keywords if kw in text)
    if legit_score > scam_score:
        return "legit"
    return "scam"

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
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
        return np.array(features)
    except Exception as e:
        return None

def load_dataset():
    print("\n📂 Loading dataset...")
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Cannot find '{METADATA_FILE}'")
        return None, None

    df = pd.read_csv(METADATA_FILE)
    print(f"   Found {len(df)} entries in metadata.csv")
    print(f"   Columns: {list(df.columns)}")

    print("\n🏷️  Creating labels from 'case_details' column...")
    df['label'] = df['case_details'].apply(create_label_from_case)
    print(f"   Label counts:\n{df['label'].value_counts().to_string()}")

    print(f"\n🔍 Checking audio folder '{AUDIO_FOLDER}'...")
    if not os.path.exists(AUDIO_FOLDER):
        print(f"❌ Folder '{AUDIO_FOLDER}' not found in: {os.getcwd()}")
        return None, None

    wav_files_on_disk = set(os.listdir(AUDIO_FOLDER))
    print(f"   Found {len(wav_files_on_disk)} files in folder")

    if len(wav_files_on_disk) == 0:
        print("❌ The audio-wav-16khz folder is EMPTY!")
        return None, None

    print(f"\n   CSV filename examples:    {list(df['file_name'].head(3).values)}")
    print(f"   Disk filename examples:   {list(wav_files_on_disk)[:3]}")

    X, y, not_found = [], [], 0
    total = len(df)

    for i, row in df.iterrows():
        filename = str(row['file_name']).strip()
        label    = row['label']

        candidates = [
            filename,
            filename + ".wav",
            filename.replace(".wav","") + ".wav",
            os.path.basename(filename),
            os.path.basename(filename) + ".wav",
        ]

        file_path = None
        for c in candidates:
            fp = os.path.join(AUDIO_FOLDER, c)
            if os.path.exists(fp):
                file_path = fp
                break

        if file_path:
            print(f"   [{i+1}/{total}] Processing: {filename}", end="\r")
            feat = extract_features(file_path)
            if feat is not None:
                X.append(feat)
                y.append(label)
        else:
            not_found += 1

    print(f"\n   ✅ Processed : {len(X)} files")
    print(f"   ⚠️  Not found : {not_found} files")

    if len(X) == 0:
        print("\n❌ No files processed! Filename mismatch between CSV and folder.")
        print("   First 5 CSV names :")
        for f in list(df['file_name'].head(5)): print(f"     {f}")
        print("   First 5 disk names:")
        for f in list(wav_files_on_disk)[:5]:   print(f"     {f}")
        return None, None

    return np.array(X), np.array(y)

def train_model(X, y):
    print("\n🤖 Training the AI model...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"   Classes: {list(le.classes_)}")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_enc, test_size=0.2,
                                               random_state=42, stratify=y_enc)
    print(f"   Train: {len(X_tr)}  |  Test: {len(X_te)}")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    acc = (model.predict(X_te) == y_te).mean() * 100
    print(f"\n   🎯 Accuracy: {acc:.1f}%")
    print(classification_report(y_te, model.predict(X_te), target_names=le.classes_))
    return model, le, scaler

def save_models(model, le, scaler):
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    joblib.dump(model,  os.path.join(MODELS_FOLDER, "scam_classifier.pkl"))
    joblib.dump(le,     os.path.join(MODELS_FOLDER, "label_encoder.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_FOLDER, "scaler.pkl"))
    print(f"\n💾 Saved to '{MODELS_FOLDER}/'")
    print(f"   ✅ scam_classifier.pkl")
    print(f"   ✅ label_encoder.pkl")
    print(f"   ✅ scaler.pkl")

if __name__ == "__main__":
    print("=" * 50)
    print("  SCAM CALL DETECTOR - MODEL TRAINING")
    print("=" * 50)
    print(f"   Working dir: {os.getcwd()}")
    X, y = load_dataset()
    if X is not None:
        model, le, scaler = train_model(X, y)
        save_models(model, le, scaler)
        print("\n✅ TRAINING COMPLETE! Now run:  streamlit run app.py")
    else:
        print("\n❌ Training failed. Share the output above so we can fix it!")