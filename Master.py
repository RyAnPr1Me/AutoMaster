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
import librosa
import soundfile as sf

# ============ CONFIGURATION ============
DEMUC_SOURCES = ["vocals", "drums", "bass", "other"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DDSP_MASTER_PATH = "models/ddsp_master.ckpt"
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("MasterBot")

# ============ UTILITIES ============

def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    return audio.set_channels(2).set_frame_rate(44100)

def separate_stems(input_file):
    logger.info("Running Demucs stem separation...")
    demucs = torch.hub.load("facebookresearch/demucs", "hdemucs_mmi", source="github")
    demucs.to(DEVICE).eval()
    wav, sr = torchaudio.load(input_file)
    with torch.no_grad():
        estimates = demucs(wav.to(DEVICE))
    stems = {}
    for i, src in enumerate(DEMUC_SOURCES):
        path = f"temp/{src}.wav"
        torchaudio.save(path, estimates[i].cpu(), sr)
        stems[src] = AudioSegment.from_wav(path)
    return stems

# ============ EQ FUNCTIONS ============

class ParametricEQBand:
    def __init__(self, center_freq, bandwidth, gain):
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.gain = gain
        self.b, self.a = self._create_bandpass_filter()

    def _create_bandpass_filter(self):
        low = self.center_freq - self.bandwidth / 2
        high = self.center_freq + self.bandwidth / 2
        nyquist = 44100 / 2
        low = low / nyquist
        high = high / nyquist
        b, a = scipy.signal.butter(2, [low, high], btype='band')
        return b, a

    def process(self, audio_samples):
        filtered = scipy.signal.lfilter(self.b, self.a, audio_samples)
        filtered *= 10 ** (self.gain / 20)
        return filtered

def energy_based_eq(audio):
    logger.info("Applying energy-based EQ...")
    
    # Analyze energy across different frequency bands
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate
    
    # Analyze frequency content with FFT
    fft = np.fft.rfft(samples)
    magnitude = np.abs(fft)
    
    bands = {
        'low': (20, 250),   # Sub-bass and bass
        'mid': (250, 2000), # Midrange
        'high': (2000, 12000) # Treble
    }
    
    energy_profile = {}
    
    for band, (low, high) in bands.items():
        band_idx = np.where((np.fft.fftfreq(len(magnitude), 1/sr) > low) & (np.fft.fftfreq(len(magnitude), 1/sr) < high))[0]
        band_magnitude = magnitude[band_idx]
        band_energy = np.sum(band_magnitude) / len(band_magnitude)
        energy_profile[band] = band_energy
    
    # Adjust EQ based on energy profile
    eq_bands = []
    if energy_profile['low'] < 0.5:  # Boost low end if it's too weak
        eq_bands.append(ParametricEQBand(center_freq=60, bandwidth=120, gain=4))
    if energy_profile['mid'] < 0.5:  # Boost mids for clarity
        eq_bands.append(ParametricEQBand(center_freq=1000, bandwidth=400, gain=2))
    if energy_profile['high'] < 0.5:  # Boost highs for brightness
        eq_bands.append(ParametricEQBand(center_freq=5000, bandwidth=1000, gain=3))
    
    processed_samples = np.zeros_like(samples)
    for band in eq_bands:
        processed_samples += band.process(samples)

    processed_samples = np.clip(processed_samples, -32768, 32767).astype(np.int16)
    return audio._spawn(processed_samples.tobytes())

# ============ MASTERING CHAIN ============

def multiband_compress(audio):
    logger.info("Multiband compression applied.")
    bands = [(20, 250), (250, 2000), (2000, 16000)]
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate
    out = np.zeros_like(samples)
    for low, high in bands:
        b, a = scipy.signal.butter(2, [low/sr*2, high/sr*2], btype="band")
        band = scipy.signal.lfilter(b, a, samples)
        gain = 0.8 / (np.std(band) + 1e-9)  # Dynamic compression
        out += band * gain
    out = np.clip(out, -32768, 32767).astype(np.int16)
    return audio._spawn(out.tobytes())

def transient_shaper(audio):
    logger.info("Enhancing transients...")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    envelope = np.abs(samples)
    threshold = np.percentile(envelope, 90)
    transients = (envelope > threshold).astype(float)
    shaped = samples * (1 + transients * 0.5)
    shaped = np.clip(shaped, -32768, 32767).astype(np.int16)
    return audio._spawn(shaped.tobytes())

def stereo_imager(audio):
    logger.info("Widening stereo image...")
    if audio.channels == 1:
        return audio.set_channels(2)
    left, right = audio.split_to_mono()
    widened = AudioSegment.from_mono_audiosegments(left - 1, right + 1)
    return widened

def normalize_lufs(audio, target=-14.0):
    logger.info("Normalizing LUFS...")
    rms = audio.rms
    db = 20 * np.log10(rms) if rms > 0 else -float("inf")
    return audio.apply_gain(target - db)

def soft_limiter(audio):
    logger.info("Applying soft limiter...")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    max_val = np.max(np.abs(samples))
    if max_val > 30000:
        samples = samples * (30000.0 / max_val)
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return audio._spawn(samples.tobytes())

def apply_mastering_chain(audio):
    logger.info("Running full mastering chain...")
    chain = [
        multiband_compress,
        transient_shaper,
        stereo_imager,
        energy_based_eq,  # Replaced advanced EQ with energy-based EQ
        normalize_lufs,
        soft_limiter,
    ]
    for stage in chain:
        audio = stage(audio)
    return audio

# ============ MAIN WORKFLOW ============

def process(input_file, output_file, dry_run=False):
    logger.info(f"Loading: {input_file}")
    audio = load_audio(input_file)
    master = apply_mastering_chain(audio)
    if dry_run:
        logger.info(f"Dry run complete. Output not saved.")
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        master.export(output_file, format="wav")
        logger.info(f"Exported mastered file to {output_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input audio file (wav or mp3)")
    p.add_argument("output", help="Output mastered file")
    p.add_argument("--dry", action="store_true", help="Dry run without exporting")
    args = p.parse_args()
    process(args.input, args.output, dry_run=args.dry)
