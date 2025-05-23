import sys
import os
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter

def remove_background_noise(audio, sr, strength=0.5):
    """
    Simple spectral gating noise reduction using median filtering.
    """
    stft = librosa.stft(audio)
    magnitude, phase = np.abs(stft), np.angle(stft)
    # Estimate noise profile as median across time
    noise_profile = np.median(magnitude, axis=1, keepdims=True)
    # Reduce noise by gating below noise profile
    mask = magnitude > (noise_profile * (1 + strength))
    cleaned_mag = magnitude * mask
    cleaned_stft = cleaned_mag * np.exp(1j * phase)
    cleaned_audio = librosa.istft(cleaned_stft)
    return cleaned_audio

def trim_silence(audio, sr, top_db=30):
    """
    Trim leading and trailing silence from audio.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def process_vocal(input_path, output_path, noise_strength=0.5, silence_db=30):
    audio, sr = librosa.load(input_path, sr=None)
    print(f"Loaded {input_path} (sr={sr}, duration={len(audio)/sr:.2f}s)")
    cleaned = remove_background_noise(audio, sr, strength=noise_strength)
    trimmed = trim_silence(cleaned, sr, top_db=silence_db)
    sf.write(output_path, trimmed, sr)
    print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python remove_vocal_noise.py <input_wav> <output_wav>")
        sys.exit(1)
    process_vocal(sys.argv[1], sys.argv[2])
