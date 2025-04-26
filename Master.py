import os
import argparse
import logging
import tempfile
from pathlib import Path

import torch
import torchaudio
import numpy as np
import scipy.signal
from pydub import AudioSegment, effects
from pydub.playback import play
import librosa

# ============ CONFIGURATION ============
DEMUC_SOURCES = ["vocals", "drums", "bass", "other"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DDSP_MASTER_PATH = "models/ddsp_master.ckpt"  # Your exported DDSP checkpoint

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ProMaster")

# ============ UTILITIES ============

def separate_stems(input_file):
    """Use Demucs from torch.hub directly for 4-stem separation."""
    logger.info("Loading Demucs from torch.hub...")
    demucs = torch.hub.load("facebookresearch/demucs", "hdemucs_mmi", source="github")
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

def transient_shaper(audio):
    logger.info("Applying transient enhancement...")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    envelope = np.abs(samples)
    threshold = np.percentile(envelope, 90)
    transients = (envelope > threshold).astype(float)
    shaped = samples * (1 + transients * 0.5)
    shaped = np.clip(shaped, -32768, 32767).astype(np.int16)
    return audio._spawn(shaped.tobytes())

def stereo_imager(audio):
    logger.info("Applying stereo imaging automation...")
    mid = audio.split_to_mono()[0]
    side = audio.split_to_mono()[1] if audio.channels > 1 else mid
    widen = side.apply_gain(3)
    return AudioSegment.from_mono_audiosegments(mid, widen)

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

def spectral_eq(audio):
    logger.info("Applying auto EQ based on spectral profile...")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    fft = np.fft.rfft(samples)
    magnitude = np.abs(fft)
    if np.mean(magnitude[:100]) > np.mean(magnitude[500:1000]):
        logger.info("Detected muddy low-end. Applying low shelf cut.")
        samples = scipy.signal.lfilter(*scipy.signal.butter(2, 150 / audio.frame_rate * 2, btype="high"), samples)
    if np.mean(magnitude[3000:8000]) < 0.1 * np.max(magnitude):
        logger.info("Detected dull high-end. Boosting highs.")
        samples = scipy.signal.lfilter(*scipy.signal.butter(2, 5000 / audio.frame_rate * 2, btype="high"), samples)
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return audio._spawn(samples.tobytes())

# ============ MAIN WORKFLOW ============

def process(input_file, output_file, dry_run=False):
    stems = separate_stems(input_file)
    mix = stems["drums"]
    for src in ["bass", "other", "vocals"]:
        mix = mix.overlay(stems[src])

    comp = multiband_compress(mix)
    shaped = transient_shaper(comp)
    widened = stereo_imager(shaped)
    eqed = spectral_eq(widened)
    eqed.export("temp/eqed.wav", format="wav")

    master = apply_ddsp_master(eqed)
    final = normalize_lufs(master)

    if dry_run:
        logger.info(f"Dry run complete, duration {final.duration_seconds}s")
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        ext = Path(output_file).suffix.replace(".", "")
        final.export(output_file, format=ext, bitrate="320k")
        logger.info(f"Exported final master to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input audio file (wav or mp3)")
    p.add_argument("output", help="Output mastered file")
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()
    process(args.input, args.output, dry_run=args.dry)

