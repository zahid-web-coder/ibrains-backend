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
# Config (edit if desired)
# -----------------------------
MODEL_PATH = "model/mdd_xgboost_model_epoched.joblib"
# channels expected by model (training used these)
FEATURE_CHANNELS = ['F3', 'F4', 'Fz', 'Cz', 'O1', 'O2']
# optional channel used for EOG detection
EOG_CHANNEL = 'Fp1'
# epoch, cropping settings (match training)
EPOCH_DURATION_S = 10.0
CROP_TMIN_S = 30.0
CROP_TMAX_S = 300.0
BANDS = {"Theta": (4, 8), "Alpha": (8, 13)}

# Flask app
app = Flask(__name__)
CORS(app)

# Load model at startup
if not os.path.exists(MODEL_PATH):
    # If deployed somewhere else, ensure model is copied to model/ path.
    app.logger.error(f"Model not found at {MODEL_PATH}. Place joblib in that path.")
    MODEL = None
else:
    MODEL = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded successfully.")

# -----------------------------
# Utility functions
# -----------------------------
def save_uploaded_files(files) -> Tuple[str, List[str]]:
    """
    Save uploaded files to a temporary directory.
    Returns (tmpdir_path, saved_filenames)
    """
    tmpdir = tempfile.mkdtemp(prefix="eeg_upload_")
    saved = []
    for f in files:
        filename = f.filename
        save_path = os.path.join(tmpdir, filename)
        f.save(save_path)
        saved.append(save_path)
    return tmpdir, saved

def find_vhdr(saved_files: List[str]) -> str:
    for p in saved_files:
        if p.lower().endswith('.vhdr'):
            return p
    return ""

def apply_preprocessing_and_ica(vhdr_path: str,
                                crop_tmin=CROP_TMIN_S,
                                crop_tmax=CROP_TMAX_S) -> mne.io.Raw:
    """
    Read BrainVision file, convert units, crop, filter, run ICA and remove EOG/ECG-like components.
    Returns cleaned raw object.
    """
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

    # Convert to microvolts (matching training)
    try:
        raw.apply_function(lambda x: x * 1e6, picks='eeg', verbose=False)
    except Exception:
        # ignore if not needed
        pass

    raw.set_meas_date(None)

    # manual crop to remove setup artifacts
    tmax = min(raw.times[-1], crop_tmax)
    if crop_tmin < tmax:
        raw.crop(tmin=crop_tmin, tmax=tmax, verbose=False)

    # reference + filters
    raw.set_eeg_reference('average', projection=True, verbose=False)
    raw.notch_filter(50.0, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)

    # ICA
    raw_ica = raw.copy()
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter="auto", verbose=False)
    # fit may fail on very short data -> catch
    try:
        ica.fit(raw_ica)
    except Exception as e:
        # fallback: return filtered raw (no ICA)
        app.logger.warning(f"ICA fit failed: {e}. Returning filtered raw without ICA.")
        return raw

    # EOG component detection
    if EOG_CHANNEL in raw_ica.ch_names:
        try:
            eog_inds, _ = ica.find_bads_eog(raw_ica, ch_name=EOG_CHANNEL, threshold=3.0, measure='zscore', verbose=False)
            ica.exclude.extend(eog_inds)
        except Exception:
            pass

    # Simple heuristic for ECG-like components: compute PSD of components and look for very low frequency power
    try:
        sources = ica.get_sources(raw_ica).get_data()
        sf = raw_ica.info['sfreq']
        psd_comp = []
        freqs = None
        for comp in sources:
            f, p = welch(comp, fs=sf, nperseg=int(sf * 4))
            psd_comp.append(p)
            if freqs is None:
                freqs = f
        psd_comp = np.array(psd_comp)
        if freqs is not None and psd_comp.size:
            low_idx = np.where((freqs >= 0.8) & (freqs <= 2.5))[0]
            if low_idx.size:
                band_power = psd_comp[:, low_idx].mean(axis=1)
                thr = band_power.mean() + 2 * band_power.std()
                ecg_like_inds = list(np.where(band_power > thr)[0])
                # filter out already excluded
                ecg_like_filtered = [i for i in ecg_like_inds if i not in ica.exclude]
                ica.exclude.extend(ecg_like_filtered)
    except Exception:
        pass

    # apply ICA
    try:
        ica.apply(raw)
    except Exception:
        # if apply fails, continue with raw (best effort)
        app.logger.warning("ICA.apply failed; returning raw without ICA applied.")
    return raw

def epoch_and_extract_features(raw: mne.io.Raw,
                               channels: List[str]=FEATURE_CHANNELS,
                               epoch_duration: float=EPOCH_DURATION_S) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Epoch the raw data into non-overlapping windows and extract the same features used in training.
    Returns (feature_dict, epoch_dataframe)
    """
    # pick only channels that exist
    picks = [ch for ch in channels if ch in raw.ch_names]
    if not picks:
        raise ValueError("None of required channels present in recording.")

    raw_pick = raw.copy().pick_channels(picks)
    sfreq = raw_pick.info['sfreq']

    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw_pick, duration=epoch_duration, overlap=0.0, preload=True, verbose=False)
    if len(epochs) == 0:
        raise ValueError("No epochs created — data too short after cropping.")

    # determine nperseg for welch
    nperseg = int(min(sfreq * 2, epoch_duration * 0.4 * sfreq))
    if nperseg < 1:
        nperseg = int(min(sfreq * 1, int(epoch_duration * sfreq)))
    if nperseg < 1:
        raise ValueError("nperseg computed too small for PSD.")

    all_features = []
    for ep in epochs.get_data():
        # ep shape: (n_channels, n_times)
        f, psd = welch(ep, fs=sfreq, nperseg=nperseg, axis=-1)
        # psd shape (n_channels, n_freqs)
        feat = psd_features_from_psd(f, psd, epochs.ch_names)
        all_features.append(feat)

    df = pd.DataFrame(all_features).dropna()
    if df.empty:
        raise ValueError("Feature extraction yielded no valid epochs (all NaN).")
    median_features = df.median().to_dict()
    return median_features, df

def psd_features_from_psd(f: np.ndarray, psd: np.ndarray, channel_names: List[str]) -> Dict[str, float]:
    ch_idx = {name: i for i, name in enumerate(channel_names)}
    def band_power(channel_name, band):
        low, high = BANDS[band]
        if channel_name not in ch_idx:
            return np.nan
        idx = np.logical_and(f >= low, f <= high)
        if not np.any(idx):
            return np.nan
        return np.mean(psd[ch_idx[channel_name], idx])
    # FAA: log(Alpha_F4) - log(Alpha_F3)
    a_f3 = band_power('F3', 'Alpha')
    a_f4 = band_power('F4', 'Alpha')
    try:
        if a_f3 is None or a_f4 is None or np.isnan(a_f3) or np.isnan(a_f4) or a_f3 <= 0 or a_f4 <= 0:
            faa = np.nan
        else:
            faa = np.log(a_f4) - np.log(a_f3)
    except Exception:
        faa = np.nan

    theta_fz = band_power('Fz', 'Theta')
    theta_cz = band_power('Cz', 'Theta')

    # OAPF: peak alpha freq averaged over O1 and O2
    alpha_idx = np.logical_and(f >= BANDS['Alpha'][0], f <= BANDS['Alpha'][1])
    if 'O1' not in ch_idx or 'O2' not in ch_idx or not np.any(alpha_idx):
        oapf = np.nan
    else:
        p_o1 = psd[ch_idx['O1'], alpha_idx]
        p_o2 = psd[ch_idx['O2'], alpha_idx]
        alpha_f = f[alpha_idx]
        if p_o1.size == 0 or p_o2.size == 0 or np.max(p_o1) == 0 or np.max(p_o2) == 0:
            oapf = np.nan
        else:
            pk1 = alpha_f[np.argmax(p_o1)]
            pk2 = alpha_f[np.argmax(p_o2)]
            oapf = float((pk1 + pk2) / 2.0)
    return {
        'FAA_Score': float(faa) if not np.isnan(faa) else np.nan,
        'Theta_Power_Fz': float(theta_fz) if not np.isnan(theta_fz) else np.nan,
        'Theta_Power_Cz': float(theta_cz) if not np.isnan(theta_cz) else np.nan,
        'OAPF': float(oapf) if not np.isnan(oapf) else np.nan
    }

def make_feature_plot(feature_df: pd.DataFrame,
                      prediction_proba: Tuple[float, float],
                      status_text: str) -> bytes:
    """
    Make a bar chart of features and return PNG bytes.
    feature_df should be a 1-row dataframe with columns matching model features.
    prediction_proba: (prob_healthy, prob_mdd)
    """
    # sort for nicer visual (largest first)
    df = feature_df.copy()
    cols = df.columns.tolist()
    values = df.iloc[0].values
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bar_colors = ['#4DB6AC' if v >= 0 else '#F48FB1' for v in values]
    ax.bar(cols, values, color=bar_colors)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel("Feature Value (median)")
    ax.set_title("Extracted Subject Features for MDD Prediction")
    plt.xticks(rotation=35, ha='right')

    # prediction box
    pred_text = (f"Predicted: {status_text}\n"
                 f"Prob(Healthy=0): {prediction_proba[0]:.4f}\n"
                 f"Prob(MDD=1): {prediction_proba[1]:.4f}")
    plt.figtext(0.5, -0.12, pred_text, ha="center", fontsize=10,
                bbox=dict(facecolor='#ffd6d6' if status_text != "Healthy (Non-MDD)" else '#d6ffd8', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# -----------------------------
# Flask route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if MODEL is None:
            return jsonify({"error": "Model not loaded on server."}), 500

        # Accept multiple files (expect .vhdr/.eeg/.vmrk) or single .vhdr that references others in same folder
        if 'file' not in request.files:
            return jsonify({"error": "No files uploaded. Use key 'file' for multipart upload."}), 400

        # request.files.getlist handles multiple uploaded files under same key
        uploaded_files = request.files.getlist('file')
        tmpdir, saved_paths = save_uploaded_files(uploaded_files)

        try:
            vhdr_path = find_vhdr(saved_paths)
            if not vhdr_path:
                # Try also finding a file with .vhdr case-insensitive
                for p in saved_paths:
                    if p.lower().endswith('.vhdr'):
                        vhdr_path = p
                        break

            if not vhdr_path:
                return jsonify({"error": "No .vhdr file uploaded. Please upload the BrainVision .vhdr (plus .eeg/.vmrk)."}), 400

            # Preprocess & clean
            raw_clean = apply_preprocessing_and_ica(vhdr_path)

            # Extract features (median across epochs)
            median_feats, epoch_df = epoch_and_extract_features(raw_clean, channels=FEATURE_CHANNELS, epoch_duration=EPOCH_DURATION_S)
            # re-order features exactly as model expects (model feature names)
            model_feature_names = ['FAA_Score', 'Theta_Power_Fz', 'Theta_Power_Cz', 'OAPF']
            ordered = [median_feats.get(k, np.nan) for k in model_feature_names]

            feat_df_for_model = pd.DataFrame([ordered], columns=model_feature_names)

            # If any NaNs remain, handle gracefully (model may error)
            if feat_df_for_model.isnull().any().any():
                # Return features but warn; still try to predict (model might error)
                # Replace NaN with 0 for prediction fallback (you may choose different strategy)
                feat_df_for_model = feat_df_for_model.fillna(0.0)

            # Predict
            probs = MODEL.predict_proba(feat_df_for_model.values)[0]
            pred_class = MODEL.predict(feat_df_for_model.values)[0]
            status_text = "Diagnosed with MDD" if int(pred_class) == 1 else "Healthy (Non-MDD)"

            # Generate plot and return as base64
            png_bytes = make_feature_plot(pd.DataFrame(feat_df_for_model.values, columns=model_feature_names),
                                          prediction_proba=(float(probs[0]), float(probs[1])),
                                          status_text=status_text)
            png_b64 = base64.b64encode(png_bytes).decode('utf-8')

            # build response
            response = {
                "prediction": status_text,
                "confidence": float(probs[1]) if int(pred_class)==1 else float(probs[0]),
                "probabilities": {"healthy": float(probs[0]), "mdd": float(probs[1])},
                "features": {k: float(feat_df_for_model.iloc[0][k]) for k in model_feature_names},
                "plot_base64": png_b64
            }
            return jsonify(response)

        finally:
            # cleanup uploaded tempdir
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error(tb)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# -----------------------------
# Health check
# -----------------------------
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

# -----------------------------
# Run (only used for local dev)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
