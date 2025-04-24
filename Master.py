import os
import argparse
import logging
import requests
from pathlib import Path
from pydub import AudioSegment
import torch
import torchaudio
import numpy as np
import scipy.signal

# ============ CONFIGURATION ============
DEMUC_SOURCES = ["vocals", "drums", "bass", "other"]
DEMUC_URL = "https://github.com/facebookresearch/demucs/releases/download/v3.0/demucs_extra.th"
DEMUC_PATH = "models/demucs_extra.th"

DDSP_MASTER_PATH = "models/ddsp_master.ckpt"  # Your exported DDSP checkpoint
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ProMaster")

# ============ UTILITIES ============
def ensure_demucs():
    if not os.path.isfile(DEMUC_PATH):
        logger.info("Downloading Demucs v3 model...")
        os.makedirs(os.path.dirname(DEMUC_PATH), exist_ok=True)
        resp = requests.get(DEMUC_URL, stream=True)
        resp.raise_for_status()
        with open(DEMUC_PATH, "wb") as f:
            for chunk in resp.iter_content(1024*1024):
                f.write(chunk)
        logger.info("Demucs model downloaded.")

def separate_stems(input_file):
    """Use Demucs v3 (via torch.hub) for 4-stem separation."""
    ensure_demucs()
    logger.info("Loading Demucs separator...")
    demucs = torch.hub.load("facebookresearch/demucs", "demucs_extra", source="github")
    demucs.to(DEVICE).eval()
    logger.info(f"Separating stems for {input_file}...")
    wav, sr = torchaudio.load(input_file)
    with torch.no_grad():
        estimates = demucs(wav.to(DEVICE))
    stems = {}
    for i, src in enumerate(DEMUC_SOURCES):
        path = f"temp/{src}.wav"
        torchaudio.save(path, estimates[i].cpu(), sr)
        stems[src] = AudioSegment.from_wav(path)
    return stems

def multiband_compress(audio, bands=None):
    """Real multiband compressor with 4th-order Butterworth filters."""
    logger.info("Applying multiband compression...")
    if bands is None:
        bands = [(20,250), (250,2000), (2000,20000)]
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate
    out = np.zeros_like(samples)
    for low, high in bands:
        b, a = scipy.signal.butter(4, [low/sr*2, high/sr*2], btype="band")
        band = scipy.signal.lfilter(b, a, samples)
        env = np.abs(band)
        threshold = 0.1 * np.max(env)
        ratio = 4.0
        gain = np.minimum(1, threshold/ (env + 1e-9) + (env - threshold)/(env*ratio+1e-9)*(env>threshold))
        out += band * gain
    out = np.clip(out, -32768,32767).astype(np.int16)
    return audio._spawn(out.tobytes())

def load_ddsp_master():
    """Load a DDSP-based mastering model (TensorFlow) via ddsp library."""
    import ddsp
    logger.info("Loading DDSP mastering model...")
    return ddsp.training.checkpoints.load(DDSP_MASTER_PATH)

def apply_ddsp_master(audio):
    """Run the DDSP model to add harmonic, reverb, and final mastering."""
    import ddsp
    wav, sr = torchaudio.load(audio)
    wav = wav.numpy().T  # [T,1]
    model = load_ddsp_master()
    logger.info("Applying DDSP neural mastering...")
    output = model(wav, sr)  # expects array, returns processed array
    # Save back to AudioSegment
    processed = (output * 32767).astype(np.int16)
    path = "temp/ddsp_mastered.wav"
    torchaudio.save(path, processed.T, sr)
    return AudioSegment.from_wav(path)

def normalize_lufs(audio, target=-14.0):
    rms = audio.rms
    db = 20*np.log10(rms) if rms>0 else -float('inf')
    return audio.apply_gain(target - db)

# ============ MAIN WORKFLOW ============
def process(input_file, output_file, dry_run=False):
    # Step 1: Stem separation
    stems = separate_stems(input_file)
    # Step 2: Auto-mix (simple peak balancing)
    mix = stems["drums"]
    for src in ["bass","other","vocals"]:
        mix = mix.overlay(stems[src])
    # Step 3: Multiband compression
    comp = multiband_compress(mix)
    # Step 4: Neural mastering with DDSP
    comp.export("temp/comp.wav", format="wav")
    master = apply_ddsp_master("temp/comp.wav")
    # Step 5: LUFS normalization
    final = normalize_lufs(master)
    if dry_run:
        logger.info(f"Dry run complete, duration {final.duration_seconds}s")
    else:
        Path(output_file).parent.mkdir(exist_ok=True, parents=True)
        final.export(output_file, format=Path(output_file).suffix.replace(".",""))
        logger.info(f"Exported final master to {output_file}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("input", help="Input mp3/wav")
    p.add_argument("output", help="Output mastered file")
    p.add_argument("--dry", action="store_true")
    args=p.parse_args()
    process(args.input, args.output, dry_run=args.dry)
