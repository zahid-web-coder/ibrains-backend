# app.py
import os
import io
import tempfile
import base64
import traceback
from typing import List, Tuple, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
import mne
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "model/mdd_xgboost_model_epoched.joblib"
FEATURE_CHANNELS = ['F3', 'F4', 'Fz', 'Cz', 'O1', 'O2']
EOG_CHANNEL = 'Fp1'
EPOCH_DURATION_S = 10.0
CROP_TMIN_S = 30.0
CROP_TMAX_S = 300.0
BANDS = {"Theta": (4, 8), "Alpha": (8, 13)}

app = Flask(__name__)
CORS(app)

# Load model
if not os.path.exists(MODEL_PATH):
    app.logger.error(f"Model not found at {MODEL_PATH}")
    MODEL = None
else:
    MODEL = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully.")

# -----------------------------
# Save uploaded EEG files
# -----------------------------
def save_uploaded_files(files):
    tmpdir = tempfile.mkdtemp(prefix="eeg_upload_")
    saved = []
    for f in files:
        save_path = os.path.join(tmpdir, f.filename)
        f.save(save_path)
        saved.append(save_path)
    return tmpdir, saved

def find_vhdr(files):
    for p in files:
        if p.lower().endswith(".vhdr"):
            return p
    return ""

# -----------------------------
# Preprocessing WITHOUT ICA
# -----------------------------
def apply_preprocessing_and_ica(vhdr_path,
                                crop_tmin=CROP_TMIN_S,
                                crop_tmax=CROP_TMAX_S):

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

    # Convert to µV
    try:
        raw.apply_function(lambda x: x * 1e6, picks="eeg", verbose=False)
    except:
        pass

    raw.set_meas_date(None)

    # Crop
    tmax = min(raw.times[-1], crop_tmax)
    if crop_tmin < tmax:
        raw.crop(tmin=crop_tmin, tmax=tmax)

    # Basic filters
    raw.set_eeg_reference("average", projection=True)
    raw.notch_filter(50.0, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)

    # ICA DISABLED
    app.logger.warning("ICA disabled due to MNE get_pazams crash")

    return raw

# -----------------------------
# Feature extraction
# -----------------------------
def epoch_and_extract_features(raw,
                               channels=FEATURE_CHANNELS,
                               epoch_duration=EPOCH_DURATION_S):

    picks = [ch for ch in channels if ch in raw.ch_names]
    if not picks:
        raise ValueError("Required channels missing from recording.")

    raw_pick = raw.copy().pick_channels(picks)
    sfreq = raw_pick.info["sfreq"]

    epochs = mne.make_fixed_length_epochs(raw_pick, duration=epoch_duration,
                                          overlap=0.0, preload=True, verbose=False)

    if len(epochs) == 0:
        raise ValueError("No epochs created; EEG too short after cropping.")

    nperseg = int(min(sfreq * 2, epoch_duration * 0.4 * sfreq))
    if nperseg < 1:
        nperseg = int(min(sfreq * 1, int(epoch_duration * sfreq)))

    all_feats = []

    for ep in epochs.get_data():
        f, psd = welch(ep, fs=sfreq, nperseg=nperseg, axis=-1)
        feat = psd_features_from_psd(f, psd, epochs.ch_names)
        all_feats.append(feat)

    df = pd.DataFrame(all_feats).dropna()
    if df.empty:
        raise ValueError("All epochs produced NaN features.")

    return df.median().to_dict(), df

def psd_features_from_psd(f, psd, channel_names):
    ch_idx = {name: i for i, name in enumerate(channel_names)}

    def band_power(ch, band):
        low, high = BANDS[band]
        if ch not in ch_idx:
            return np.nan
        idx = np.logical_and(f >= low, f <= high)
        if not np.any(idx):
            return np.nan
        return np.mean(psd[ch_idx[ch], idx])

    # FAA
    a_f3 = band_power("F3", "Alpha")
    a_f4 = band_power("F4", "Alpha")
    faa = np.log(a_f4) - np.log(a_f3) if (a_f3 and a_f4 and a_f3 > 0 and a_f4 > 0) else np.nan

    theta_fz = band_power("Fz", "Theta")
    theta_cz = band_power("Cz", "Theta")

    alpha_idx = np.logical_and(f >= 8, f <= 13)
    if "O1" in ch_idx and "O2" in ch_idx and np.any(alpha_idx):
        p1 = psd[ch_idx["O1"], alpha_idx]
        p2 = psd[ch_idx["O2"], alpha_idx]
        if p1.size > 0 and p2.size > 0:
            freq_alpha = f[alpha_idx]
            oapf = float((freq_alpha[np.argmax(p1)] + freq_alpha[np.argmax(p2)]) / 2)
        else:
            oapf = np.nan
    else:
        oapf = np.nan

    return {
        "FAA_Score": float(faa) if not np.isnan(faa) else np.nan,
        "Theta_Power_Fz": float(theta_fz) if theta_fz else np.nan,
        "Theta_Power_Cz": float(theta_cz) if theta_cz else np.nan,
        "OAPF": float(oapf) if not np.isnan(oapf) else np.nan
    }

# -----------------------------
# API endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if MODEL is None:
            return jsonify({"error": "Model failed to load"}), 500

        if "file" not in request.files:
            return jsonify({"error": "Upload .vhdr, .eeg, .vmrk"}), 400

        files = request.files.getlist("file")
        tmpdir, saved_paths = save_uploaded_files(files)

        try:
            vhdr = find_vhdr(saved_paths)
            if not vhdr:
                return jsonify({"error": "Missing .vhdr file"}), 400

            raw = apply_preprocessing_and_ica(vhdr)
            feats, df_epochs = epoch_and_extract_features(raw)

            model_features = ['FAA_Score', 'Theta_Power_Fz', 'Theta_Power_Cz', 'OAPF']
            ordered = [feats.get(k, 0.0) for k in model_features]
            feat_df = pd.DataFrame([ordered], columns=model_features).fillna(0.0)

            probs = MODEL.predict_proba(feat_df.values)[0]
            pred = MODEL.predict(feat_df.values)[0]
            label = "Diagnosed with MDD" if int(pred) == 1 else "Healthy (Non-MDD)"

            # Plot
            img = make_feature_plot(feat_df, (float(probs[0]), float(probs[1])), label)
            b64 = base64.b64encode(img).decode("utf-8")

            return jsonify({
                "prediction": label,
                "confidence": float(probs[int(pred)]),
                "probabilities": {"healthy": float(probs[0]), "mdd": float(probs[1])},
                "features": {k: float(feat_df.iloc[0][k]) for k in model_features},
                "plot_base64": b64
            })

        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
