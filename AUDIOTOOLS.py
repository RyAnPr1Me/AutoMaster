# === IMPORTS ===
import sys
import os
import subprocess
import numpy as np
import librosa
import soundfile as sf
from scipy.ndimage import median_filter
import torch
import torchaudio
import logging
from pathlib import Path
from pydub import AudioSegment, effects
from mido import MidiFile, Message, MidiTrack, MetaMessage
import random
import scipy.signal

# NOTE: Requires pydub, torchaudio, librosa, soundfile, mido, scipy, numpy
# Install with: pip install pydub torchaudio librosa soundfile mido scipy numpy

# === WATERMARK DETECTION ===
WATERMARK_PATTERN = [60, 62, 64, 65, 67]
WATERMARK_CHANNEL = 15

def detect_watermark_patterns(midi_path, patterns=[[60, 62, 64, 65, 67]], velocity_threshold=5, tolerance=1, min_matches=4):
    """
    Robustly detect watermark patterns in any channel, at low velocity, with tolerance for note/velocity/timing.
    - patterns: list of watermark patterns to check (can be transposed, reversed, etc)
    - velocity_threshold: max velocity to consider as watermark
    - tolerance: max note difference allowed
    - min_matches: minimum number of notes in sequence to consider a match
    """
    try:
        mid = MidiFile(midi_path)
        found_any = False
        for track_idx, track in enumerate(mid.tracks):
            channel_notes = {}
            channel_velocities = {}
            channel_times = {}
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity <= velocity_threshold:
                    ch = getattr(msg, 'channel', 0)
                    channel_notes.setdefault(ch, []).append(msg.note)
                    channel_velocities.setdefault(ch, []).append(msg.velocity)
                    channel_times.setdefault(ch, []).append(abs_time)
            for ch in channel_notes:
                notes = channel_notes[ch]
                for pattern in patterns:
                    for i in range(len(notes) - len(pattern) + 1):
                        window = notes[i:i+len(pattern)]
                        if sum(abs(a-b) <= tolerance for a, b in zip(window, pattern)) >= min_matches:
                            print(f"Watermark pattern found in track {track_idx}, channel {ch}")
                            found_any = True
        # ENHANCED: Try to find watermark notes even if not in strict sequence
        if not found_any:
            all_low_vel_notes = set()
            all_low_vel_times = []
            for track in mid.tracks:
                abs_time = 0
                for msg in track:
                    abs_time += msg.time
                    if msg.type == 'note_on' and msg.velocity <= velocity_threshold:
                        all_low_vel_notes.add(msg.note)
                        all_low_vel_times.append(abs_time)
            for pattern in patterns:
                if all(n in all_low_vel_notes for n in pattern):
                    print(f"Watermark pattern (unordered) found in low-velocity notes")
                    found_any = True
        # ADVANCED: Try more transpositions (octave, fifth, etc)
        if not found_any:
            transpositions = [2, 3, 5, 7, 12, -2, -3, -5, -7, -12]
            for t in transpositions:
                for pattern in patterns:
                    transposed = [n + t for n in pattern]
                    if all(n in all_low_vel_notes for n in transposed):
                        print(f"Transposed watermark pattern found (t={t})")
                        found_any = True
        # ADVANCED: Rhythmic fingerprinting (relative time deltas)
        if not found_any and len(all_low_vel_times) >= min_matches:
            deltas = [all_low_vel_times[i+1] - all_low_vel_times[i] for i in range(len(all_low_vel_times)-1)]
            if any(d < 100 for d in deltas):
                print("Short time deltas found in low-velocity notes (possible watermark)")
                found_any = True
        if found_any:
            print(f"Watermark detected in {midi_path}")
        else:
            print(f"No watermark detected in {midi_path}")
        return found_any
    except Exception as e:
        print(f"[ERROR] Watermark detection failed: {e}")
        return False

# === VOCAL NOISE REMOVAL ===
def remove_background_noise(audio, sr, strength=0.5):
    """
    Simple spectral gating noise reduction using median filtering.
    """
    stft = librosa.stft(audio)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_profile = np.median(magnitude, axis=1, keepdims=True)
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
    try:
        audio, sr = librosa.load(input_path, sr=None)
        print(f"Loaded {input_path} (sr={sr}, duration={len(audio)/sr:.2f}s)")
        cleaned = remove_background_noise(audio, sr, strength=noise_strength)
        trimmed = trim_silence(cleaned, sr, top_db=silence_db)
        sf.write(output_path, trimmed, sr)
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"[ERROR] Vocal noise removal failed: {e}")

# === AUDIO SLICER ===
def audio_slice(input_file, segment_length, prefix):
    try:
        audio = AudioSegment.from_file(input_file)
        segment_ms = int(segment_length * 1000)
        for i, start in enumerate(range(0, len(audio), segment_ms)):
            segment = audio[start:start+segment_ms]
            outname = f"{prefix}_slice_{i+1}.wav"
            segment.export(outname, format="wav")
            print(f"Exported {outname}")
    except Exception as e:
        print(f"[ERROR] Audio slicing failed: {e}")

# === BATCH NORMALIZE ===
def batch_normalize(folder):
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.wav', '.mp3')):
            path = os.path.join(folder, fname)
            try:
                audio = AudioSegment.from_file(path)
                normalized = effects.normalize(audio)
                outname = os.path.join(folder, f"normalized_{fname}")
                normalized.export(outname, format="wav")
                print(f"Exported {outname}")
            except Exception as e:
                print(f"[ERROR] Failed to normalize {fname}: {e}")

# === BPM DETECT ===
def detect_bpm(filename):
    try:
        y, sr = librosa.load(filename)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Estimated BPM: {tempo:.2f}")
    except Exception as e:
        print(f"[ERROR] BPM detection failed: {e}")

# === KEY DETECT ===
def detect_key(filename):
    try:
        y, sr = librosa.load(filename)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_index = np.argmax(chroma_mean)
        print(f"Estimated key: {keys[key_index]}")
    except Exception as e:
        print(f"[ERROR] Key detection failed: {e}")

# === MIDI TO WAV ===
def midi_to_wav(midi_file, wav_file, sf2_file):
    try:
        cmd = ["fluidsynth", "-ni", sf2_file, midi_file, "-F", wav_file, "-r", "44100"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Exported {wav_file}")
    except Exception as e:
        print(f"[ERROR] MIDI to WAV conversion failed: {e}")

# === TRAP BEAT GENERATOR ===
BPM = 140
PPQ = 480
NUM_BARS = 8
FILENAME = 'max_channels_trap_gold_watermarked.mid'
MINOR_NATURAL = [48, 50, 51, 53, 55, 56, 58]
KICK = 36
SNARE = 38
CLAP = 39
HIHAT_CLOSED = 42
HIHAT_OPEN = 46
RIDE = 51
CRASH = 49
TOM_LOW = 45
TOM_MID = 47
TOM_HIGH = 50
BASE_808 = 36

def add_note(track, note, velocity, start_tick, duration, channel):
    if len(track) == 0:
        delta = start_tick
    else:
        delta = max(0, start_tick - sum(msg.time for msg in track if hasattr(msg, 'time')))
    track.append(Message('note_on', note=note, velocity=velocity, time=delta, channel=channel))
    track.append(Message('note_off', note=note, velocity=0, time=duration, channel=channel))

def add_inaudible_signature(mid, signature='TS'):
    char_to_note = {c: (ord(c) % 32) for c in signature.upper()}
    channel = 15
    velocity = 1
    duration = 1
    spacing = PPQ // 4
    track = mid.tracks[channel]
    if not track:
        track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM), time=0))
    current_tick = 0
    for char in signature:
        note = char_to_note.get(char, 0)
        track.append(Message('note_on', note=note, velocity=velocity, time=current_tick, channel=channel))
        track.append(Message('note_off', note=note, velocity=0, time=duration, channel=channel))
        current_tick = spacing

def embed_watermark_in_music(tracks, pattern=[60, 62, 64, 65, 67], channels=[1,2,5,6,7,8], velocity=2, duration=PPQ//8):
    for i, note in enumerate(pattern):
        channel = random.choice(channels)
        bar = random.randint(0, NUM_BARS-1)
        bar_start = bar * BAR_LENGTH
        tick = bar_start + random.randint(0, BAR_LENGTH//2)
        add_note(tracks[channel], note, velocity, tick, duration, channel)

def energy_based_eq(audio):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate
    fft = np.fft.rfft(samples)
    magnitude = np.abs(fft)
    bands = {'low': (20, 250), 'mid': (250, 2000), 'high': (2000, 12000)}
    energy_profile = {}
    for band, (low, high) in bands.items():
        band_idx = np.where((np.fft.fftfreq(len(magnitude), 1/sr) > low) & (np.fft.fftfreq(len(magnitude), 1/sr) < high))[0]
        band_magnitude = magnitude[band_idx]
        band_energy = np.sum(band_magnitude) / len(band_magnitude) if len(band_magnitude) > 0 else 0
        energy_profile[band] = band_energy
    eq_bands = []
    if energy_profile.get('low', 1) < 0.5:
        eq_bands.append(ParametricEQBand(center_freq=60, bandwidth=120, gain=4))
    if energy_profile.get('mid', 1) < 0.5:
        eq_bands.append(ParametricEQBand(center_freq=1000, bandwidth=400, gain=2))
    if energy_profile.get('high', 1) < 0.5:
        eq_bands.append(ParametricEQBand(center_freq=5000, bandwidth=1000, gain=3))
    processed_samples = np.zeros_like(samples)
    for band in eq_bands:
        processed_samples += band.process(samples)
    if len(eq_bands) > 0:
        processed_samples = np.clip(processed_samples, -32768, 32767).astype(np.int16)
        return audio._spawn(processed_samples.tobytes())
    else:
        return audio

def multiband_compress(audio):
    bands = [(20, 250), (250, 2000), (2000, 16000)]
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    sr = audio.frame_rate
    out = np.zeros_like(samples)
    for low, high in bands:
        b, a = scipy.signal.butter(2, [low/sr*2, high/sr*2], btype="band")
        band = scipy.signal.lfilter(b, a, samples)
        gain = 0.8 / (np.std(band) + 1e-9)
        out += band * gain
    out = np.clip(out, -32768, 32767).astype(np.int16)
    return audio._spawn(out.tobytes())

def transient_shaper(audio):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    envelope = np.abs(samples)
    threshold = np.percentile(envelope, 90)
    transients = (envelope > threshold).astype(float)
    shaped = samples * (1 + transients * 0.5)
    shaped = np.clip(shaped, -32768, 32767).astype(np.int16)
    return audio._spawn(shaped.tobytes())

def stereo_imager(audio):
    if audio.channels == 1:
        return audio.set_channels(2)
    left, right = audio.split_to_mono()
    widened = AudioSegment.from_mono_audiosegments(left - 1, right + 1)
    return widened

def normalize_lufs(audio, target=-14.0):
    rms = audio.rms
    db = 20 * np.log10(rms) if rms > 0 else -float("inf")
    return audio.apply_gain(target - db)

def soft_limiter(audio):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    max_val = np.max(np.abs(samples))
    if max_val > 30000:
        samples = samples * (30000.0 / max_val)
    samples = np.clip(samples, -32768, 32767).astype(np.int16)
    return audio._spawn(samples.tobytes())

def apply_mastering_chain(audio):
    chain = [
        multiband_compress,
        transient_shaper,
        stereo_imager,
        energy_based_eq,
        normalize_lufs,
        soft_limiter,
    ]
    for stage in chain:
        audio = stage(audio)
    return audio

def generate_full_trap_beat(
    num_bars=NUM_BARS,
    bpm=BPM,
    complexity=0.7,
    swing=0.1,
    random_seed=None,
    watermark_pattern=[60, 62, 64, 65, 67],
    watermark_channels=[1,2,5,6,7,8],
    watermark_velocity=2,
    watermark_duration=PPQ//8,
    signature='TS',
    filename=FILENAME
):
    """
    Generate a robust, musical trap beat with watermarking and signature embedding.
    - num_bars: number of bars in the beat
    - bpm: tempo
    - complexity: 0-1, controls fill density and variation
    - swing: 0-0.5, amount of swing (timing offset on even 16th notes)
    - random_seed: for reproducibility
    - watermark_pattern, watermark_channels, watermark_velocity, watermark_duration: watermarking params
    - signature: string to embed as inaudible signature
    - filename: output MIDI file
    """
    if random_seed is not None:
        random.seed(random_seed)
    mid = MidiFile(ticks_per_beat=PPQ)
    tracks = [MidiTrack() for _ in range(16)]
    for t in tracks:
        mid.tracks.append(t)
    tempo = mido.bpm2tempo(bpm)
    for track in tracks:
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    # Set instruments
    for i, track in enumerate(tracks):
        if i == 9:
            pass
        elif i in [3, 4]:
            track.append(Message('program_change', program=38, time=0, channel=i))
        elif i in [1, 2, 5, 6]:
            track.append(Message('program_change', program=81, time=0, channel=i))
        elif i in [7, 8]:
            track.append(Message('program_change', program=89, time=0, channel=i))
        elif i in [11, 12]:
            track.append(Message('program_change', program=95, time=0, channel=i))
        elif i in [13, 14, 15]:
            track.append(Message('program_change', program=52, time=0, channel=i))
        else:
            track.append(Message('program_change', program=0, time=0, channel=i))
    # --- Beat generation ---
    for bar in range(num_bars):
        bar_start = bar * BAR_LENGTH
        if bar < 2:
            section = 'intro'
        elif bar > num_bars - 3:
            section = 'outro'
        elif bar in [num_bars//2-1, num_bars//2]:
            section = 'bridge'
        else:
            section = 'drop'
        # --- Drums ---
        for step in range(16):
            tick = bar_start + step * (BAR_LENGTH // 16)
            # Add swing to even 16th notes
            if step % 2 == 1:
                tick += int(PPQ * swing)
            # Humanize timing
            tick += random.randint(-8, 8)
            # KICK: main pattern + random ghost notes
            if (section != 'intro' and step in [0, 6, 10, 14]) or (section == 'intro' and step == 0) or (random.random() < 0.08 * complexity):
                vel = 120 + random.randint(-10, 10)
                add_note(tracks[9], KICK, vel, tick, PPQ // 8, 9)
            # SNARE/CLAP: main + ghost notes
            if step in [4, 12]:
                vel = 110 + random.randint(-10, 10)
                add_note(tracks[9], SNARE, vel, tick, PPQ // 8, 9)
                if random.random() < 0.5:
                    add_note(tracks[9], CLAP, 90 + random.randint(-10, 10), tick + PPQ // 32, PPQ // 8, 9)
            elif step in [6, 14] and random.random() < 0.2 * complexity:
                add_note(tracks[9], SNARE, 60 + random.randint(-10, 10), tick, PPQ // 16, 9)
            # HIHAT: 16th grid, velocity groove, open hats, rolls, swing
            if section != 'outro' or step % 4 == 0:
                base_vel = 70 + int(25 * (1 if step % 4 == 0 else random.random()))
                vel = base_vel + random.randint(-10, 10)
                add_note(tracks[9], HIHAT_CLOSED, vel, tick, PPQ // 16, 9)
                # Hi-hat rolls
                if random.random() < 0.15 * complexity and step % 4 == 2:
                    for roll in range(2 + int(complexity*2)):
                        roll_tick = tick + roll * (PPQ // 32) + random.randint(-2, 2)
                        add_note(tracks[9], HIHAT_CLOSED, vel - 10, roll_tick, PPQ // 32, 9)
                # Open hats
                if step % 4 == 2 and random.random() < 0.3 * complexity:
                    add_note(tracks[9], HIHAT_OPEN, 80 + random.randint(-10, 10), tick, PPQ // 16, 9)
        # --- 808 Bass with slides and fills ---
        base_pattern = [0, 0, 3, 5, 0, 7, 5, 0]
        for i, channel in enumerate([3, 4]):
            last_note = None
            for step in range(8):
                note = BASE_808 + base_pattern[step] + (random.choice([-2, 0, 2]) if random.random() < 0.18 * complexity else 0)
                velocity = 115 - step * 2 + int(10 * random.random())
                start = bar_start + step * (BAR_LENGTH // 8) + random.randint(-8, 8)
                dur = PPQ // 2
                # Slide/overlap
                if random.random() < 0.18 * complexity and last_note is not None:
                    add_note(tracks[channel], last_note, velocity - 10, start, dur // 2, channel)
                add_note(tracks[channel], note, velocity, start, dur, channel)
                last_note = note
        # --- Melodic leads with call/response, humanization, fills ---
        for idx, channel in enumerate([1, 2, 5, 6]):
            note = random.choice(MINOR_NATURAL) + 12
            time_cursor = bar_start
            for n in range(6):
                interval = random.choice([-2, -1, 0, 1, 2])
                note_index = MINOR_NATURAL.index(note % 12) if (note % 12) in MINOR_NATURAL else 0
                note = MINOR_NATURAL[(note_index + interval) % len(MINOR_NATURAL)] + 12
                velocity = 75 + random.randint(-15, 20)
                if (bar + n) % 2 == idx % 2 or random.random() < 0.2 * complexity:
                    add_note(tracks[channel], note, velocity, time_cursor + random.randint(-12, 12), BAR_LENGTH // 6, channel)
                time_cursor += BAR_LENGTH // 6
        # --- Pads/Atmosphere ---
        if bar % 4 == 0 and section not in ['intro', 'bridge']:
            pad_notes = [MINOR_NATURAL[0] + 24, MINOR_NATURAL[3] + 24]
            pad_duration = BAR_LENGTH * 2
            for channel in [7, 8]:
                for note in pad_notes:
                    add_note(tracks[channel], note, 50 + random.randint(-10, 10), bar_start, pad_duration, channel)
        # --- FX & Breaks ---
        if random.random() < 0.4:
            for channel in [11, 12]:
                note = random.choice(MINOR_NATURAL) + 36
                add_note(tracks[channel], note, 40 + random.randint(-15, 15), bar_start + random.randint(0, BAR_LENGTH // 2), PPQ // 2, channel)
        # --- Vocal Chops ---
        chop_notes = [60, 62, 64, 65, 67]
        quarter = PPQ
        for channel in [13, 14, 15]:
            for beat in range(4):
                if random.random() < 0.6:
                    note = random.choice(chop_notes)
                    velocity = 80 + random.randint(-15, 15)
                    add_note(tracks[channel], note, velocity, bar_start + beat * quarter + random.randint(-12, 12), quarter // 2, channel)
        # --- Toms, Crash, Ride, Fills ---
        if (bar + 1) % 4 == 0 and section == 'drop':
            tom_notes = [TOM_LOW, TOM_MID, TOM_HIGH]
            step = BAR_LENGTH // (len(tom_notes) * 2)
            tick = bar_start
            for note in tom_notes:
                add_note(tracks[10], note, 90 + random.randint(-15, 15), tick, step, 10)
                tick += step
                add_note(tracks[10], note, 90 + random.randint(-15, 15), tick, step, 10)
                tick += step
        if (bar + 1) in [2, 6] and section == 'drop':
            ride_tick = bar_start
            sixteenth = BAR_LENGTH // 16
            for _ in range(16):
                add_note(tracks[10], RIDE, 60 + random.randint(-15, 15), ride_tick, sixteenth // 2, 10)
                ride_tick += sixteenth
        if (bar + 1) in [1, 5] and section == 'drop':
            add_note(tracks[10], CRASH, 100 + random.randint(-15, 15), bar_start, PPQ, 10)
    # --- Watermark and signature ---
    embed_watermark_in_music(tracks, pattern=watermark_pattern, channels=watermark_channels, velocity=watermark_velocity, duration=watermark_duration)
    add_inaudible_signature(mid, signature=signature)
    try:
        mid.save(filename)
        print(f"🔥 Robust trap beat generated and saved: '{filename}' (Watermarked, signature embedded)")
    except Exception as e:
        print(f"[ERROR] Could not save MIDI: {e}")

# === STEM SPLIT ===
def stem_split(input_file, output_dir, stems=2):
    """
    Split audio into stems using Spleeter.
    stems: 2 (vocals/accompaniment), 4 (vocals, drums, bass, other), or 5 (vocals, drums, bass, piano, other)
    """
    try:
        from spleeter.separator import Separator
    except ImportError:
        print("[ERROR] The 'spleeter' package is required for stem splitting. Install it with: pip install spleeter")
        sys.exit(1)
    if stems not in [2, 4, 5]:
        print("[ERROR] Stems must be 2, 4, or 5.")
        sys.exit(1)
    model = f'spleeter:{stems}stems'
    separator = Separator(model)
    print(f"Splitting '{input_file}' into {stems} stems...")
    separator.separate_to_file(input_file, output_dir)
    print(f"Stems saved to '{output_dir}'")

# === CLI ===
def print_usage():
    print("""
AUDIOTOOLS.py - Unified Audio/Music Processing Toolkit

Usage:
  python AUDIOTOOLS.py watermark-detect <input.mid>
  python AUDIOTOOLS.py remove-vocal-noise <input.wav> <output.wav> [--noise-strength=0.5] [--silence-db=30]
  python AUDIOTOOLS.py slice <input.wav> <segment_length_sec> <output_prefix>
  python AUDIOTOOLS.py normalize <folder>
  python AUDIOTOOLS.py bpm <input.wav>
  python AUDIOTOOLS.py key <input.wav>
  python AUDIOTOOLS.py midi2wav <input.mid> <output.wav> <soundfont.sf2>
  python AUDIOTOOLS.py trapbeat [output.mid]
  python AUDIOTOOLS.py master <input.wav> <output.wav>
  python AUDIOTOOLS.py stem-split <input.wav> <output_dir> [--stems=2|4|5]

Commands:
  watermark-detect   Detect watermark in MIDI file
  remove-vocal-noise Remove background noise and trim silence from vocal WAV
  slice              Slice audio into segments
  normalize          Batch normalize all audio files in a folder
  bpm                Detect BPM of audio file
  key                Detect musical key of audio file
  midi2wav           Convert MIDI to WAV using a SoundFont
  trapbeat           Generate trap beat MIDI with watermark
  master             Master an audio file
  stem-split         Split audio into stems using Spleeter (2, 4, or 5 stems)
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "watermark-detect":
        if len(sys.argv) < 3:
            print("Usage: python AUDIOTOOLS.py watermark-detect <input.mid>")
            sys.exit(1)
        midi_file = sys.argv[2]
        patterns = [WATERMARK_PATTERN, list(reversed(WATERMARK_PATTERN)), [n+1 for n in WATERMARK_PATTERN], [n-1 for n in WATERMARK_PATTERN]]
        detect_watermark_patterns(midi_file, patterns=patterns, velocity_threshold=5, tolerance=1, min_matches=4)
    elif cmd == "remove-vocal-noise":
        if len(sys.argv) < 4:
            print("Usage: python AUDIOTOOLS.py remove-vocal-noise <input.wav> <output.wav> [--noise-strength=0.5] [--silence-db=30]")
            sys.exit(1)
        input_wav = sys.argv[2]
        output_wav = sys.argv[3]
        noise_strength = 0.5
        silence_db = 30
        for i, arg in enumerate(sys.argv):
            if arg.startswith("--noise-strength="):
                noise_strength = float(arg.split("=")[1])
            if arg.startswith("--silence-db="):
                silence_db = float(arg.split("=")[1])
        process_vocal(input_wav, output_wav, noise_strength=noise_strength, silence_db=silence_db)
    elif cmd == "slice":
        if len(sys.argv) < 5:
            print("Usage: python AUDIOTOOLS.py slice <input.wav> <segment_length_sec> <output_prefix>")
            sys.exit(1)
        audio_slice(sys.argv[2], float(sys.argv[3]), sys.argv[4])
    elif cmd == "normalize":
        if len(sys.argv) < 3:
            print("Usage: python AUDIOTOOLS.py normalize <folder>")
            sys.exit(1)
        batch_normalize(sys.argv[2])
    elif cmd == "bpm":
        if len(sys.argv) < 3:
            print("Usage: python AUDIOTOOLS.py bpm <input.wav>")
            sys.exit(1)
        detect_bpm(sys.argv[2])
    elif cmd == "key":
        if len(sys.argv) < 3:
            print("Usage: python AUDIOTOOLS.py key <input.wav>")
            sys.exit(1)
        detect_key(sys.argv[2])
    elif cmd == "midi2wav":
        if len(sys.argv) < 5:
            print("Usage: python AUDIOTOOLS.py midi2wav <input.mid> <output.wav> <soundfont.sf2>")
            sys.exit(1)
        midi_to_wav(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "trapbeat":
        outname = sys.argv[2] if len(sys.argv) > 2 else FILENAME
        generate_full_trap_beat(filename=outname)
    elif cmd == "master":
        if len(sys.argv) < 4:
            print("Usage: python AUDIOTOOLS.py master <input.wav> <output.wav>")
            sys.exit(1)
        master_audio(sys.argv[2], sys.argv[3])
    elif cmd == "stem-split":
        if len(sys.argv) < 4:
            print("Usage: python AUDIOTOOLS.py stem-split <input.wav> <output_dir> [--stems=2|4|5]")
            sys.exit(1)
        input_file = sys.argv[2]
        output_dir = sys.argv[3]
        stems = 2
        for arg in sys.argv[4:]:
            if arg.startswith("--stems="):
                stems = int(arg.split("=")[1])
        stem_split(input_file, output_dir, stems=stems)
    else:
        print_usage()
        sys.exit(1)
