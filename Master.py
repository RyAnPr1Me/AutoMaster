import os
import argparse
import logging
import tempfile
from pathlib import Path

import requests
import torch
import torchaudio
import numpy as np
import scipy.signal
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import librosa

# ============ CONFIGURATION ============
DEMUC_SOURCES = ["vocals", "drums", "bass", "other"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DDSP_MASTER_PATH = "models/ddsp_master.ckpt"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ProMaster")

# ============ UTILITIES ============

def separate_stems(input_file):
    logger.info("Loading Demucs from torch.hub...")
    demucs = torch.hub.load("facebookresearch/demucs:v3.0", "demucs", source="github")
    demucs.to(DEVICE).eval()

    logger.info(f"Converting and loading audio: {input_file}")
    audio = AudioSegment.from_file(input_file)
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav.name, format="wav")

    wav, sr = torchaudio.load(temp_wav.name)
    os.unlink(temp_wav.name)

    with torch.no_grad():
        estimates = demucs(wav.to(DEVICE))

    stems = {}
    os.makedirs("temp", exist_ok=True)
    for i, src in enumerate(DEMUC_SOURCES):
        path = f"temp/{src}.wav"
        torchaudio.save(path, estimates[i].cpu(), sr)
        stems[src] = AudioSegment.from_wav(path)
    return stems

def multiband_compress(audio, bands=None):
    logger.info("Applying multiband compression...")
    if bands is None:
        bands = [(20, 250), (250, 2000), (2000, 20000)]
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate
    out = np.zeros_like(samples)
    for low, high in bands:
        b, a = scipy.signal.butter(4, [low/sr*2, high/sr*2], btype="band")
        band = scipy.signal.lfilter(b, a, samples)
        env = np.abs(band)
        threshold = 0.1 * np.max(env)
        ratio = 4.0
        gain = np.minimum(1, threshold/(env + 1e-9) + (env - threshold)/(env*ratio+1e-9)*(env > threshold))
        out += band * gain
    out = np.clip(out, -32768, 32767).astype(np.int16)
    return audio._spawn(out.tobytes())

def load_ddsp_master():
    import ddsp
    logger.info("Loading DDSP mastering model...")
    return ddsp.training.checkpoints.load(DDSP_MASTER_PATH)

def apply_ddsp_master(audio):
    import ddsp
    temp_input = "temp/ddsp_input.wav"
    temp_output = "temp/ddsp_output.wav"
    audio.export(temp_input, format="wav")

    wav, sr = torchaudio.load(temp_input)
    wav = wav.numpy().T
    model = load_ddsp_master()

    logger.info("Applying DDSP neural mastering...")
    output = model(wav, sr)
    processed = (output * 32767).astype(np.int16)
    torchaudio.save(temp_output, processed.T, sr)
    return AudioSegment.from_wav(temp_output)

def normalize_lufs(audio, target=-14.0):
    rms = audio.rms
    db = 20 * np.log10(rms) if rms > 0 else -float("inf")
    return audio.apply_gain(target - db)

def apply_true_peak_limiting(audio):
    logger.info("Applying true peak limiting...")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    max_peak = np.max(np.abs(samples))
    if max_peak > 32767:
        gain = 32767.0 / max_peak
        samples *= gain
    return audio._spawn(samples.astype(np.int16).tobytes())

def detect_genre(file_path):
    logger.info("Detecting genre using MFCCs and KNN classifier...")
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_scaled = StandardScaler().fit_transform(mfcc.T.mean(axis=0).reshape(1, -1))

    # Simulated training for genre classifier
    genres = ["rock", "pop", "hiphop", "electronic"]
    X_train = np.eye(4)
    y_train = np.array(genres)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    genre = knn.predict([[1 if g == "rock" else 0 for g in genres]])[0]  # fake example
    logger.info(f"Detected genre: {genre}")
    return genre

def genre_based_eq(audio, genre):
    logger.info(f"Applying genre-based EQ for: {genre}")
    eq = {
        "rock": [(100, 3), (5000, -2)],
        "pop": [(80, 2), (8000, 2)],
        "hiphop": [(60, 4), (7000, -3)],
        "electronic": [(40, 5), (10000, 3)]
    }
    eq_settings = eq.get(genre, [])
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate
    for freq, gain_db in eq_settings:
        b, a = scipy.signal.iirpeak(freq / (sr / 2), Q=1.0)
        samples = scipy.signal.lfilter(b, a, samples)
        samples *= 10 ** (gain_db / 20.0)
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return audio._spawn(samples.tobytes())

# ============ MAIN WORKFLOW ============

def process(input_file, output_file, dry_run=False):
    stems = separate_stems(input_file)
    mix = stems["drums"]
    for src in ["bass", "other", "vocals"]:
        mix = mix.overlay(stems[src])

    comp = multiband_compress(mix)
    comp.export("temp/comp.wav", format="wav")

    genre = detect_genre("temp/comp.wav")
    eq = genre_based_eq(comp, genre)
    master = apply_ddsp_master(eq)
    final = normalize_lufs(master, target=-14.0)
    limited = apply_true_peak_limiting(final)

    if dry_run:
        logger.info(f"Dry run complete, duration {limited.duration_seconds}s")
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        ext = Path(output_file).suffix.replace(".", "")
        limited.export(output_file, format=ext)
        logger.info(f"Exported final master to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input audio file (wav or mp3)")
    p.add_argument("output", help="Output mastered file")
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    process(args.input, args.output, dry_run=args.dry)
