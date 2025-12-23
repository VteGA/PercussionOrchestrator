"""
Percussion Orchestrator — Engine (para Streamlit)

Este módulo contiene el "motor" (carga de dataset + descriptores + búsqueda kNN/híbrida + render).
Está adaptado desde tu notebook de Colab exportado.

Estructura esperada del dataset:

perc_dataset/
  audio/           (samples del dataset)
  metadata.csv     (o cualquier .csv con al menos columnas: file, orchestration_role)
  test_sounds/     (opcional: ejemplos para probar)
  renders/         (salidas generadas)

Si no existe perc_dataset, puedes descargarlo desde un ZIP público de Google Drive (opcional).
"""

from __future__ import annotations

import os
import shutil
import zipfile
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


WEIGHT_COLS = ['w_energy','w_brightness','w_noisiness','w_spectral_shape','w_timbre']

ROLE_PRESETS: Dict[str, Dict[str, float]] = {
    'attack_layer':    dict(w_energy=0.7, w_brightness=1.2, w_noisiness=0.8, w_spectral_shape=1.2, w_timbre=0.8),
    'body_layer':      dict(w_energy=0.7, w_brightness=0.9, w_noisiness=0.7, w_spectral_shape=1.2, w_timbre=0.8),
    'resonance_layer': dict(w_energy=0.6, w_brightness=1.2, w_noisiness=0.8, w_spectral_shape=1.6, w_timbre=1.0),
    'noise_layer':     dict(w_energy=0.6, w_brightness=1.0, w_noisiness=1.8, w_spectral_shape=1.5, w_timbre=0.9),
    'special_fx':      dict(w_energy=0.8, w_brightness=1.1, w_noisiness=1.2, w_spectral_shape=1.4, w_timbre=1.2),
}

# Para "hybrid" (penalización de metadata)
ORD_MAPS = {
    'pitch_region': ['sub', 'low', 'mid', 'high', 'ultra'],
    'brightness':   ['dark', 'mid', 'bright'],
    'attack':       ['soft', 'mid', 'sharp'],
    'sustain':      ['short', 'mid', 'long'],
    'noise_tone':   ['tonal', 'mixed', 'noisy'],
    'dynamic':      ['pp', 'p', 'mp', 'mf', 'f', 'ff'],
}

META_WEIGHTS = {
    'material': 1.0,
    'gesture_type': 1.0,
    'pitch_region': 0.8,
    'brightness': 0.8,
    'attack': 0.8,
    'sustain': 0.8,
    'noise_tone': 0.8,
    'dynamic': 0.6,
}


def _clean(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


def _ordinal_dist(col, a, b):
    if a is None or b is None:
        return 0.5
    levels = ORD_MAPS[col]
    try:
        ia = levels.index(a)
        ib = levels.index(b)
    except ValueError:
        return 0.5
    return abs(ia - ib) / max(1, len(levels) - 1)


def _categorical_dist(a, b):
    if a is None or b is None:
        return 0.5
    return 0.0 if a == b else 1.0


def metadata_penalty(row: pd.Series, target: dict) -> float:
    p = 0.0
    if 'material' in row.index:
        p += META_WEIGHTS['material'] * _categorical_dist(_clean(row.get('material')), target.get('material'))
    if 'gesture_type' in row.index:
        p += META_WEIGHTS['gesture_type'] * _categorical_dist(_clean(row.get('gesture_type')), target.get('gesture_type'))
    for col in ['pitch_region','brightness','attack','sustain','noise_tone','dynamic']:
        if col in row.index:
            p += META_WEIGHTS[col] * _ordinal_dist(col, _clean(row.get(col)), target.get(col))
    return float(p)


def read_csv_auto(path: str, base_dir: str) -> Tuple[pd.DataFrame, str, str]:
    """Carga CSV probando separador ; y , y (si no existe path) detecta el primer CSV en base_dir."""
    if not os.path.exists(path):
        csvs = [f for f in os.listdir(base_dir) if f.lower().endswith('.csv')]
        if not csvs:
            raise FileNotFoundError("No encuentro ningún CSV en el directorio del dataset.")
        path = os.path.join(base_dir, csvs[0])

    # probar ; y ,
    try:
        df = pd.read_csv(path, sep=';')
        if 'file' in df.columns:
            return df, path, ';'
    except Exception:
        pass

    df = pd.read_csv(path, sep=',')
    if 'file' not in df.columns:
        raise ValueError("El CSV debe tener columna 'file'.")
    return df, path, ','


def _try_download_gdrive_zip(zip_url: str, zip_path: str) -> None:
    """
    Descarga un ZIP desde Drive. Requiere 'gdown' instalado.
    Probamos:
      - import gdown y fuzzy=True (si existe)
      - CLI: gdown --fuzzy URL -O zip_path
    """
    try:
        import gdown  # type: ignore
        # gdown.download suele aceptar fuzzy=True en versiones recientes
        try:
            gdown.download(url=zip_url, output=zip_path, quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(url=zip_url, output=zip_path, quiet=False)
        if not os.path.exists(zip_path):
            raise RuntimeError("Descarga fallida (no se creó el ZIP).")
        return
    except Exception:
        # fallback CLI
        subprocess.run(["gdown", "--fuzzy", zip_url, "-O", zip_path], check=True)


def ensure_dataset(project_root: str, zip_url: Optional[str] = None) -> Dict[str, str]:
    """
    Asegura que exista la estructura del dataset. Si falta y zip_url está definido, lo descarga y extrae.
    Devuelve rutas.
    """
    project_root = os.path.abspath(project_root)
    audio_dir = os.path.join(project_root, "audio")
    test_dir = os.path.join(project_root, "test_sounds")
    render_dir = os.path.join(project_root, "renders")

    # localizar CSV
    csv_candidates = []
    if os.path.isdir(project_root):
        csv_candidates = [os.path.join(project_root, f) for f in os.listdir(project_root) if f.lower().endswith(".csv")]
    csv_path = os.path.join(project_root, "metadata.csv") if os.path.exists(os.path.join(project_root, "metadata.csv")) else (csv_candidates[0] if csv_candidates else os.path.join(project_root, "metadata.csv"))

    ok = os.path.isdir(audio_dir) and os.path.exists(csv_path)

    if not ok:
        if not zip_url:
            raise FileNotFoundError(
                "No encuentro el dataset.\n"
                f"Espero: {project_root}/audio y un CSV (por ejemplo metadata.csv).\n"
                "O pásame un ZIP público de Drive (zip_url) para descargarlo."
            )

        # descargar + extraer
        os.makedirs(project_root, exist_ok=True)
        zip_path = os.path.join(project_root, "_dataset.zip")
        _try_download_gdrive_zip(zip_url, zip_path)

        # extraer
        tmp_extract = os.path.join(project_root, "_extract_tmp")
        if os.path.exists(tmp_extract):
            shutil.rmtree(tmp_extract)
        os.makedirs(tmp_extract, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_extract)

        # mover: si dentro hay una carpeta perc_dataset/ o similar, usamos esa como raíz
        # (si hay una sola carpeta dentro, asumimos que es la raíz real)
        entries = [os.path.join(tmp_extract, e) for e in os.listdir(tmp_extract)]
        roots = [e for e in entries if os.path.isdir(e)]
        if len(roots) == 1:
            src_root = roots[0]
        else:
            src_root = tmp_extract

        # copiar a project_root (sobrescribe)
        for name in os.listdir(src_root):
            src = os.path.join(src_root, name)
            dst = os.path.join(project_root, name)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        # limpiar
        try:
            os.remove(zip_path)
        except Exception:
            pass
        shutil.rmtree(tmp_extract, ignore_errors=True)

        # recalcular rutas y CSV
        audio_dir = os.path.join(project_root, "audio")
        test_dir = os.path.join(project_root, "test_sounds")
        render_dir = os.path.join(project_root, "renders")
        csv_candidates = [os.path.join(project_root, f) for f in os.listdir(project_root) if f.lower().endswith(".csv")]
        csv_path = os.path.join(project_root, "metadata.csv") if os.path.exists(os.path.join(project_root, "metadata.csv")) else (csv_candidates[0] if csv_candidates else os.path.join(project_root, "metadata.csv"))

        if not (os.path.isdir(audio_dir) and os.path.exists(csv_path)):
            raise FileNotFoundError("He descargado/extractado el ZIP pero no encuentro /audio o el CSV en la raíz.")

    os.makedirs(render_dir, exist_ok=True)

    return dict(PROJECT_ROOT=project_root, AUDIO_DIR=audio_dir, TEST_DIR=test_dir, RENDER_DIR=render_dir, CSV_PATH=csv_path)


def extract_features(filepath: str, sr: int = 22050, n_mfcc: int = 13,
                     feature_duration: float = 1.0, trim_db: int = 40, target_rms: float = 0.1) -> Optional[np.ndarray]:
    """Descriptores rápidos (idéntico al notebook)."""
    y, sr = librosa.load(filepath, sr=sr, mono=True)
    if len(y) == 0:
        return None

    y, _ = librosa.effects.trim(y, top_db=trim_db)
    if len(y) == 0:
        return None

    # ventana fija desde el inicio
    target_len = int(feature_duration * sr)
    if len(y) >= target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    # normalizar RMS (evita que un sample "fuerte" domine)
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    if rms > 0:
        y = y * (float(target_rms) / rms)

    rms_feat = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    feats = np.hstack([
        np.mean(rms_feat), np.std(rms_feat),
        np.mean(centroid), np.std(centroid),
        np.mean(rolloff), np.std(rolloff),
        np.mean(zcr), np.std(zcr),
        np.mean(flat), np.std(flat),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1)
    ]).astype(np.float32)

    return feats


def _feature_slices_from_dim(D: int) -> Dict[str, slice]:
    """Para construir los pesos por-bloques (energy, brightness, etc.)."""
    if (D - 24) % 2 != 0:
        raise ValueError(f"D={D} no cuadra con 24 + 2*n_mfcc. Revisa extract_features.")
    n_mfcc = (D - 24) // 2
    return {
        'energy':         slice(0, 2),
        'brightness':     slice(2, 6),
        'noisiness':      slice(6, 10),
        'spectral_shape': slice(10, 24),
        'timbre':         slice(24, 24 + 2*n_mfcc),
    }


def _weight_vector_from_row(row: pd.Series, D: int, slices: Dict[str, slice]) -> np.ndarray:
    w = np.ones(D, dtype=np.float32)
    w[slices['energy']]         *= float(row.get('w_energy', 1.0))
    w[slices['brightness']]     *= float(row.get('w_brightness', 1.0))
    w[slices['noisiness']]      *= float(row.get('w_noisiness', 1.0))
    w[slices['spectral_shape']] *= float(row.get('w_spectral_shape', 1.0))
    w[slices['timbre']]         *= float(row.get('w_timbre', 1.0))
    return w


def _weighted_euclidean(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.sum((d * d) * w)))


def render_mix_from_rows_with_negative_offsets(
    rows: pd.DataFrame,
    audio_dir: str,
    sr_target: int = 44100,
    gains: Optional[List[float]] = None,
    time_offsets: Optional[List[float]] = None,
    out_path: str = "mix_render.wav",
    final_peak: float = 0.99,
):
    """
    Mezcla N samples (mono) con offsets (pueden ser negativos).
    - gains: multiplicadores por sample
    - time_offsets: segundos (+ = retrasar sample, - = adelantar)
    """
    rows = rows.reset_index(drop=True)
    files = rows['file'].astype(str).tolist()
    n = len(files)
    if n == 0:
        raise ValueError("No hay filas que mezclar (rows está vacío).")

    if gains is None:
        gains = [1.0] * n
    if time_offsets is None:
        time_offsets = [0.0] * n

    if len(gains) != n or len(time_offsets) != n:
        raise ValueError("gains y time_offsets deben tener la misma longitud que rows.")

    signals = []
    starts = []
    lengths = []

    for f, gain, off in zip(files, gains, time_offsets):
        path = os.path.join(audio_dir, f)
        if not os.path.exists(path):
            continue
        y, _ = librosa.load(path, sr=sr_target, mono=True)
        y = y.astype(np.float32) * float(gain)

        start = int(round(float(off) * sr_target))  # puede ser negativo
        signals.append(y)
        starts.append(start)
        lengths.append(len(y))

    if not signals:
        raise ValueError("No se ha podido cargar ningún audio para mezclar.")

    min_start = min(starts)
    # desplazamos todo para que el más temprano empiece en 0
    shift = -min_start if min_start < 0 else 0
    starts = [s + shift for s in starts]

    total_len = max(s + L for s, L in zip(starts, lengths))
    mix = np.zeros(total_len, dtype=np.float32)

    for y, s in zip(signals, starts):
        mix[s:s+len(y)] += y

    max_abs = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if max_abs > 0:
        mix = mix / (max_abs + 1e-9) * float(final_peak)

    sf.write(out_path, mix, sr_target)
    return out_path


def crop_wav_to_input_duration(mix_path: str, input_audio_path: str, out_path: Optional[str] = None, sr: int = 44100) -> str:
    y_mix, _ = librosa.load(mix_path, sr=sr, mono=True)
    y_in, _ = librosa.load(input_audio_path, sr=sr, mono=True)
    n = len(y_in)
    y_mix = y_mix[:n] if len(y_mix) >= n else np.pad(y_mix, (0, n - len(y_mix)))
    if out_path is None:
        out_path = mix_path
    sf.write(out_path, y_mix.astype(np.float32), sr)
    return out_path


def _rms_env(path, sr=44100, hop=512, dur=3.0, trim_db=40):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    if dur is not None:
        n = int(sr * dur)
        y = y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))
    env = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    env = env - np.mean(env)
    env = env / (np.linalg.norm(env) + 1e-9)
    return env, sr, hop


def compute_offsets_by_envelope(input_audio_path, rows, audio_dir,
                               dur=3.0, max_shift_s=0.8, sr=44100, hop=512):
    env_in, sr, hop = _rms_env(input_audio_path, sr=sr, hop=hop, dur=dur)
    max_lag = int(max_shift_s * sr / hop)

    offsets, corrs = [], []
    for f in rows['file'].astype(str).tolist():
        p = os.path.join(audio_dir, f)
        if not os.path.exists(p):
            offsets.append(0.0); corrs.append(0.0); continue
        env_c, _, _ = _rms_env(p, sr=sr, hop=hop, dur=dur)

        best_score, best_lag = -1e9, 0
        for lag in range(-max_lag, max_lag+1):
            if lag < 0:
                a, b = env_in[-lag:], env_c[:len(env_in)+lag]
            elif lag > 0:
                a, b = env_in[:-lag], env_c[lag:lag+len(env_in)-lag]
            else:
                a, b = env_in, env_c[:len(env_in)]

            if len(a) < 8 or len(b) < 8:
                continue

            sc = float(np.dot(a[:len(b)], b[:len(a)]))
            if sc > best_score:
                best_score, best_lag = sc, lag

        offsets.append((best_lag * hop) / sr)
        corrs.append(best_score)

    return offsets, corrs


def peak_time_seconds(path, sr=44100, hop=512, dur=3.0, trim_db=40):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=trim_db)
    if dur is not None:
        n = int(sr * dur)
        y = y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))
    env = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    idx = int(np.argmax(env)) if len(env) else 0
    return (idx * hop) / sr


def compute_offsets_by_peak(input_audio_path, rows, audio_dir,
                            dur=3.0, max_shift_s=1.0, sr=44100, hop=512):
    t_in = peak_time_seconds(input_audio_path, sr=sr, hop=hop, dur=dur)
    offsets = []
    for f in rows['file'].astype(str).tolist():
        p = os.path.join(audio_dir, f)
        if not os.path.exists(p):
            offsets.append(0.0); continue
        t_c = peak_time_seconds(p, sr=sr, hop=hop, dur=dur)
        off = float(t_in - t_c)
        off = max(-max_shift_s, min(max_shift_s, off))
        offsets.append(off)
    return offsets


def build_vocab(series: pd.Series) -> Dict[str, int]:
    vals = sorted({_clean(v) for v in series if _clean(v) is not None})
    return {v: i for i, v in enumerate(vals)}


def one_hot(vocab: Dict[str, int], val) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    v = _clean(val)
    if v in vocab:
        vec[vocab[v]] = 1.0
    return vec


def encode_ordinal(col: str, val) -> float:
    levels = ORD_MAPS[col]
    v = _clean(val)
    if v not in levels:
        return 0.5
    if len(levels) == 1:
        return 0.5
    return float(levels.index(v)) / float(len(levels) - 1)


def row_l2_norm(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    return A / n


@dataclass
class EnginePaths:
    PROJECT_ROOT: str
    AUDIO_DIR: str
    TEST_DIR: str
    RENDER_DIR: str
    CSV_PATH: str


class PercussionOrchestratorEngine:
    """
    Motor: carga dataset + prepara índices + ofrece sugerencias + render.
    """

    def __init__(self, project_root: str, zip_url: Optional[str] = None):
        p = ensure_dataset(project_root, zip_url=zip_url)
        self.paths = EnginePaths(**p)

        # --- cargar CSV ---
        df, csv_path, sep = read_csv_auto(self.paths.CSV_PATH, self.paths.PROJECT_ROOT)
        self.csv_sep = sep
        self.paths.CSV_PATH = csv_path

        # columnas obligatorias
        for r in ['file', 'orchestration_role']:
            if r not in df.columns:
                raise ValueError(f"Falta columna obligatoria en CSV: {r}")

        # asegurar columnas w_*
        for c in WEIGHT_COLS:
            if c not in df.columns:
                df[c] = 1.0

        # aplicar presets por rol
        for role, preset in ROLE_PRESETS.items():
            mask = (df['orchestration_role'] == role)
            for k, v in preset.items():
                df.loc[mask, k] = float(v)

        for c in WEIGHT_COLS:
            df[c] = df[c].astype(float).clip(0.3, 3.0)

        self.df = df

        # --- extraer features del dataset ---
        feats = []
        missing = 0
        for _, row in self.df.iterrows():
            f = str(row['file'])
            path = os.path.join(self.paths.AUDIO_DIR, f)
            if not os.path.exists(path):
                missing += 1
                feats.append(None)
                continue
            feats.append(extract_features(path))

        self.df = self.df.copy()
        self.df['features'] = feats
        self.df = self.df[self.df['features'].notnull()].reset_index(drop=True)

        X = np.vstack(self.df['features'].values)

        # escalado 0-1
        self.scaler_audio = MinMaxScaler(feature_range=(0, 1))
        self.X_scaled = self.scaler_audio.fit_transform(X)

        # kNN audio
        self.knn_audio = NearestNeighbors(n_neighbors=min(200, len(self.df)), metric='euclidean')
        self.knn_audio.fit(self.X_scaled)

        # --- metadata matrices (para combined) ---
        self.material_vocab = build_vocab(self.df['material']) if 'material' in self.df.columns else {}
        self.gesture_vocab  = build_vocab(self.df['gesture_type']) if 'gesture_type' in self.df.columns else {}

        def encode_metadata_row(row: pd.Series) -> np.ndarray:
            parts = []
            for col in ['pitch_region','brightness','attack','sustain','noise_tone','dynamic']:
                if col in row.index:
                    parts.append(np.array([encode_ordinal(col, row[col])], dtype=np.float32))
                else:
                    parts.append(np.array([0.5], dtype=np.float32))

            if self.material_vocab:
                if 'material' in row.index:
                    parts.append(one_hot(self.material_vocab, row['material']))
                else:
                    parts.append(np.zeros(len(self.material_vocab), dtype=np.float32))

            if self.gesture_vocab:
                if 'gesture_type' in row.index:
                    parts.append(one_hot(self.gesture_vocab, row['gesture_type']))
                else:
                    parts.append(np.zeros(len(self.gesture_vocab), dtype=np.float32))

            return np.concatenate(parts).astype(np.float32)

        self._encode_metadata_row = encode_metadata_row  # guarda el callable

        self.M = np.vstack([encode_metadata_row(self.df.iloc[i]) for i in range(len(self.df))]).astype(np.float32)
        self.Mn = row_l2_norm(self.M)
        self.Xn = row_l2_norm(self.X_scaled.astype(np.float32))

    # ---------------------- SUGGESTIONS ----------------------

    def suggest_percussion_weighted(self, input_audio_path: str, n_return: int = 20, K: int = 80) -> pd.DataFrame:
        feats_in = extract_features(input_audio_path)
        if feats_in is None:
            raise ValueError("No se han podido extraer features del input.")
        xq = self.scaler_audio.transform([feats_in])[0]

        K = min(int(K), len(self.df))
        d, idx = self.knn_audio.kneighbors([xq], n_neighbors=K)
        cand = self.df.iloc[idx[0]].copy().reset_index(drop=True)

        Xcand = np.vstack(cand['features'].values)
        Xcand = self.scaler_audio.transform(Xcand)

        D = Xcand.shape[1]
        slices = _feature_slices_from_dim(D)

        scores = []
        for i, row in cand.iterrows():
            w = _weight_vector_from_row(row, D, slices)
            scores.append(_weighted_euclidean(xq, Xcand[i], w))

        cand['score'] = scores
        return cand.sort_values('score').head(int(n_return)).reset_index(drop=True)

    def infer_profile_from_audio(self, input_audio_path: str, nn_for_cats: int = 25, sr: int = 22050, dur: float = 2.0) -> dict:
        y, _ = librosa.load(input_audio_path, sr=sr, mono=True)
        y, _ = librosa.effects.trim(y, top_db=40)
        if len(y) == 0:
            raise ValueError("Audio de entrada vacío tras trim.")
        n = int(sr * dur)
        y = y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))

        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        attack_val = float(np.max(onset_env)) if len(onset_env) else 0.0
        rms = float(np.mean(librosa.feature.rms(y=y)[0]))

        frame = librosa.util.frame(y, frame_length=2048, hop_length=512)
        env = np.sqrt(np.mean(frame**2, axis=0) + 1e-12)
        if len(env) >= 4:
            peak_i = int(np.argmax(env))
            tail = env[min(len(env)-1, peak_i+2):]
            sustain_val = float(np.mean(tail) / (np.max(env) + 1e-9)) if len(tail) else 0.0
        else:
            sustain_val = 0.0

        pitch_region = None
        try:
            f0, _, _ = librosa.pyin(y, fmin=55, fmax=1760, sr=sr)
            f0_med = float(np.nanmedian(f0)) if f0 is not None else np.nan
            if np.isfinite(f0_med):
                if f0_med < 55: pitch_region = 'sub'
                elif f0_med < 110: pitch_region = 'low'
                elif f0_med < 440: pitch_region = 'mid'
                elif f0_med < 1760: pitch_region = 'high'
                else: pitch_region = 'ultra'
        except Exception:
            pitch_region = None

        brightness = 'dark' if centroid < 1500 else ('mid' if centroid < 3500 else 'bright')
        noise_tone = 'tonal' if flatness < 0.15 else ('mixed' if flatness < 0.35 else 'noisy')
        attack = 'soft' if attack_val < 2.0 else ('mid' if attack_val < 6.0 else 'sharp')
        sustain = 'short' if sustain_val < 0.25 else ('mid' if sustain_val < 0.55 else 'long')
        dynamic = 'pp' if rms < 0.01 else ('p' if rms < 0.02 else ('mp' if rms < 0.05 else ('mf' if rms < 0.10 else 'f')))

        # estimar material/gesture por vecinos audio
        feats_in = extract_features(input_audio_path)
        xq = self.scaler_audio.transform([feats_in])
        _, idx = self.knn_audio.kneighbors(xq, n_neighbors=min(int(nn_for_cats), len(self.df)))
        neigh = self.df.iloc[idx[0]]

        material = _clean(neigh['material'].mode().iloc[0]) if ('material' in self.df.columns and neigh['material'].notna().any()) else None
        gesture  = _clean(neigh['gesture_type'].mode().iloc[0]) if ('gesture_type' in self.df.columns and neigh['gesture_type'].notna().any()) else None

        return {
            'material': material,
            'gesture_type': gesture,
            'pitch_region': _clean(pitch_region),
            'brightness': _clean(brightness),
            'attack': _clean(attack),
            'sustain': _clean(sustain),
            'noise_tone': _clean(noise_tone),
            'dynamic': _clean(dynamic),
        }

    def suggest_percussion_hybrid(self, input_audio_path: str, n_return: int = 20, K: int = 120, lambda_meta: float = 0.8):
        ranked = self.suggest_percussion_weighted(input_audio_path, n_return=min(int(K), len(self.df)), K=min(int(K), len(self.df)))
        target = self.infer_profile_from_audio(input_audio_path)

        ranked = ranked.copy()
        ranked['meta_penalty'] = ranked.apply(lambda r: metadata_penalty(r, target), axis=1)
        ranked['final_score'] = ranked['score'] + float(lambda_meta) * ranked['meta_penalty']

        return ranked.sort_values('final_score').head(int(n_return)).reset_index(drop=True), target

    @staticmethod
    def _pick_quota_per_role(ranked_df: pd.DataFrame, quotas: Dict[str, int]) -> pd.DataFrame:
        blocks = []
        for role, n in quotas.items():
            if n <= 0:
                continue
            sub = ranked_df[ranked_df['orchestration_role'] == role].head(int(n))
            if len(sub) > 0:
                blocks.append(sub)
        if not blocks:
            return ranked_df.head(0)
        return pd.concat(blocks, axis=0).reset_index(drop=True)

    def suggest_hybrid_quota(self, input_audio_path: str, quotas: Dict[str, int], K: int = 200, lambda_meta: float = 0.8, fill_missing: bool = True):
        ranked, target = self.suggest_percussion_hybrid(
            input_audio_path,
            n_return=min(int(K), len(self.df)),
            K=min(int(K), len(self.df)),
            lambda_meta=lambda_meta
        )
        final = self._pick_quota_per_role(ranked, quotas)
        target_n = int(sum(quotas.values()))
        if fill_missing and len(final) < target_n:
            used = set(final['file'].astype(str).tolist())
            rest = ranked[~ranked['file'].astype(str).isin(used)]
            final = pd.concat([final, rest.head(target_n - len(final))], axis=0).reset_index(drop=True)
        return final, target, ranked

    def _encode_metadata_from_profile(self, profile: dict) -> np.ndarray:
        fake = pd.Series({
            'pitch_region': profile.get('pitch_region'),
            'brightness': profile.get('brightness'),
            'attack': profile.get('attack'),
            'sustain': profile.get('sustain'),
            'noise_tone': profile.get('noise_tone'),
            'dynamic': profile.get('dynamic'),
            'material': profile.get('material'),
            'gesture_type': profile.get('gesture_type'),
        })
        return self._encode_metadata_row(fake)

    def suggest_combined(self, input_audio_path: str, n_return: int = 20, K: int = 120, beta: float = 0.6, alpha: float = 1.0):
        """
        Combined = concatenar (audio_norm + beta*meta_norm) y ordenar por distancia euclídea.
        Importante: aquí sí respetamos beta dinámico SIN reconstruir índices (dataset pequeño).
        """
        xa = extract_features(input_audio_path)
        if xa is None:
            raise ValueError("No se han podido extraer features del input.")
        xa = self.scaler_audio.transform([xa])[0].astype(np.float32)
        xa = xa / (np.linalg.norm(xa) + 1e-9)

        prof = self.infer_profile_from_audio(input_audio_path)
        xm = self._encode_metadata_from_profile(prof).astype(np.float32)
        xm = xm / (np.linalg.norm(xm) + 1e-9)

        zq = np.concatenate([alpha * xa, float(beta) * xm], axis=0)  # (D,)
        Zb = np.concatenate([alpha * self.Xn, float(beta) * self.Mn], axis=1)  # (N, D)

        # distancias vectorizadas
        diff = Zb - zq[None, :]
        dist = np.sqrt(np.sum(diff * diff, axis=1))
        order = np.argsort(dist)[:min(int(K), len(self.df))]

        cand = self.df.iloc[order].copy()
        cand['distance'] = dist[order]
        return cand.sort_values('distance').head(int(n_return)).reset_index(drop=True), prof

    # ---------------------- IO helpers ----------------------

    def list_test_sounds(self, extensions=('.wav', '.flac', '.mp3', '.ogg')) -> List[str]:
        if not os.path.isdir(self.paths.TEST_DIR):
            return []
        files = [f for f in os.listdir(self.paths.TEST_DIR) if f.lower().endswith(extensions)]
        files.sort()
        return files

    def unique_render_path(self, method: str, input_path: str) -> str:
        base = os.path.splitext(os.path.basename(input_path))[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.paths.RENDER_DIR, f"mix_{method}_{base}_{stamp}.wav")

    def run_orchestrator(
        self,
        input_audio_path: str,
        method: str = "hybrid_quota",
        n_return: int = 20,
        K: int = 200,
        lambda_meta: float = 0.8,
        beta: float = 0.6,
        use_offsets: bool = True,
        offsets_mode: str = "envelope",
        env_dur: float = 3.0,
        max_shift: float = 0.8,
        do_crop: bool = True,
        quotas: Optional[Dict[str, int]] = None,
        gain_map: Optional[Dict[str, float]] = None,
    ):
        quotas = quotas or {"attack_layer": 2, "body_layer": 2, "resonance_layer": 2, "noise_layer": 1, "special_fx": 1}
        gain_map = gain_map or {"attack_layer": 1.0, "body_layer": 0.7, "resonance_layer": 0.6, "noise_layer": 0.6, "special_fx": 0.8}

        if method == "weighted":
            result = self.suggest_percussion_weighted(input_audio_path, n_return=n_return, K=K)
            profile = None
        elif method == "hybrid":
            result, profile = self.suggest_percussion_hybrid(input_audio_path, n_return=n_return, K=K, lambda_meta=lambda_meta)
        elif method == "hybrid_quota":
            result, profile, _ = self.suggest_hybrid_quota(input_audio_path, quotas=quotas, K=K, lambda_meta=lambda_meta, fill_missing=True)
            if int(n_return) < len(result):
                result = result.head(int(n_return)).reset_index(drop=True)
        elif method == "combined":
            result, profile = self.suggest_combined(input_audio_path, n_return=n_return, K=K, beta=beta)
        else:
            raise ValueError("Método inválido")

        # offsets
        time_offsets = [0.0] * len(result)
        if bool(use_offsets) and len(result) > 0:
            if offsets_mode == "envelope":
                time_offsets, corrs = compute_offsets_by_envelope(
                    input_audio_path, result, audio_dir=self.paths.AUDIO_DIR,
                    dur=env_dur, max_shift_s=max_shift
                )
                result = result.copy()
                result["time_offset_s"] = time_offsets
                result["env_corr"] = corrs
            else:
                time_offsets = compute_offsets_by_peak(
                    input_audio_path, result, audio_dir=self.paths.AUDIO_DIR,
                    dur=env_dur, max_shift_s=max_shift
                )
                result = result.copy()
                result["time_offset_s"] = time_offsets

        gains = [gain_map.get(r, 1.0) for r in result["orchestration_role"]] if "orchestration_role" in result.columns else None

        out_path = self.unique_render_path(method, input_audio_path)
        render_mix_from_rows_with_negative_offsets(
            result, audio_dir=self.paths.AUDIO_DIR,
            gains=gains, time_offsets=time_offsets, out_path=out_path
        )

        if bool(do_crop):
            crop_wav_to_input_duration(out_path, input_audio_path, out_path=out_path)

        return out_path, result, profile
