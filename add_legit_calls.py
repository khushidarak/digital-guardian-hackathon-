"""
ADD LEGIT CALLS TO DATASET
===========================
Ye script:
1. LibriSpeech dev-clean se .flac files dhundhega
2. .wav mein convert karega
3. audio-wav-16khz folder mein copy karega
4. metadata.csv mein 'legit' label add karega

Run karo:
  python add_legit_calls.py
"""

import os
import shutil
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
LIBRISPEECH_FOLDER = r"C:\Users\SHLOK\Desktop\dev-clean"
AUDIO_FOLDER       = "audio-wav-16khz"
METADATA_FILE      = "metadata.csv"
MAX_FILES          = 300  # kitni legit files add karni hain

print("=" * 50)
print("  ADDING LEGIT CALLS TO DATASET")
print("=" * 50)

# ─────────────────────────────────────────────
# STEP 1: Find all .flac files
# ─────────────────────────────────────────────
print(f"\n📂 Searching for .flac files in: {LIBRISPEECH_FOLDER}")

flac_files = []
for root, dirs, files in os.walk(LIBRISPEECH_FOLDER):
    for f in files:
        if f.endswith('.flac'):
            flac_files.append(os.path.join(root, f))

print(f"   Found {len(flac_files)} .flac files")

if len(flac_files) == 0:
    print("❌ No .flac files found!")
    print(f"   Make sure dev-clean folder is at: {LIBRISPEECH_FOLDER}")
    exit()

# Limit to MAX_FILES
flac_files = flac_files[:MAX_FILES]
print(f"   Using first {len(flac_files)} files")

# ─────────────────────────────────────────────
# STEP 2: Convert .flac to .wav and copy
# ─────────────────────────────────────────────
print(f"\n🔄 Converting .flac → .wav ...")

try:
    import librosa
    import soundfile as sf
    use_librosa = True
except ImportError:
    use_librosa = False
    print("   librosa not found, trying soundfile only...")

os.makedirs(AUDIO_FOLDER, exist_ok=True)

converted = []
failed = 0

for i, flac_path in enumerate(flac_files):
    # New filename: legit_001.wav, legit_002.wav etc
    new_filename = f"legit_{i+1:04d}.wav"
    out_path = os.path.join(AUDIO_FOLDER, new_filename)

    print(f"   [{i+1}/{len(flac_files)}] {new_filename}", end="\r")

    try:
        if use_librosa:
            audio, sr = librosa.load(flac_path, sr=16000)
            sf.write(out_path, audio, 16000)
        else:
            import soundfile as sf
            audio, sr = sf.read(flac_path)
            sf.write(out_path, audio, sr)
        converted.append(new_filename)
    except Exception as e:
        failed += 1

print(f"\n   ✅ Converted: {len(converted)} files")
if failed > 0:
    print(f"   ⚠️  Failed: {failed} files")

# ─────────────────────────────────────────────
# STEP 3: Update metadata.csv
# ─────────────────────────────────────────────
print(f"\n📝 Updating metadata.csv ...")

# Load existing metadata
df_existing = pd.read_csv(METADATA_FILE)
print(f"   Existing entries: {len(df_existing)} (all scam)")

# Create new rows for legit calls
new_rows = []
for fname in converted:
    new_rows.append({
        'file_name': f"audio-wav-16khz/{fname}",
        'language': 'en',
        'transcript': '',
        'case_details': 'legit',
        'case_pdf': ''
    })

df_new = pd.DataFrame(new_rows)

# Combine both
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined.to_csv(METADATA_FILE, index=False)

print(f"   Scam calls:  {len(df_existing)}")
print(f"   Legit calls: {len(df_new)}")
print(f"   Total:       {len(df_combined)}")

print("\n✅ DONE!")
print("   Now run: python train_model.py")
