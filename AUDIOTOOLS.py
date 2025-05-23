# === IMPORTS ===
import os
import random
import subprocess
import shutil # For shutil.copy and shutil.which
import librosa
import numpy as np # Ensure numpy is imported
import soundfile as sf
from pydub import AudioSegment, effects
from pydub.silence import split_on_silence
import mido # Keep mido import
from mido import MidiFile, Message, MidiTrack, MetaMessage, bpm2tempo # Explicitly import bpm2tempo
import scipy.signal # Ensure scipy.signal is imported
from scipy.ndimage import median_filter
from pathlib import Path
import argparse

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

def remove_vocal_noise_deepfilternet(input_path, output_path, model_name="DeepFilterNet3"):
    try:
        import deepfilternet as dfn
        # Initialize DeepFilterNet model
        # Adjust attenuation and other parameters as needed
        model = dfn.DeepFilterNet.from_pretrained(model_name)
        # Load audio
        noisy_audio, sr = librosa.load(input_path, sr=model.sr)
        # Enhance audio
        enhanced_audio = model.enhance(noisy_audio)
        # Save enhanced audio
        sf.write(output_path, enhanced_audio, sr)
        print(f"Vocal noise removed using DeepFilterNet. Output: {output_path}")
        return True
    except ImportError:
        print("Error: deepfilternet library is not installed. Please install it to use this feature (e.g., pip install deepfilternet).")
        return False
    except Exception as e:
        print(f"Error during DeepFilterNet processing: {e}")
        return False

def trim_silence(audio, sr, top_db=30):
    """
    Trim leading and trailing silence from audio.
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def process_vocal(input_path, output_path, noise_strength=0.5, silence_db=30, use_deepfilternet=False):
    if use_deepfilternet:
        remove_vocal_noise_deepfilternet(input_path, output_path)
        return
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
BAR_LENGTH = PPQ * 4  # 4/4 time signature, 1 bar = 4 quarter notes
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
    # Clamp note and velocity to valid MIDI range
    note = max(0, min(127, note))
    velocity = max(0, min(127, velocity))
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

def energy_based_eq(audio): # audio is Pydub AudioSegment
    samples_original = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        samples_mono_for_analysis = np.mean(samples_original.reshape(-1, audio.channels), axis=1)
    else:
        samples_mono_for_analysis = samples_original.copy()

    sr = audio.frame_rate
    
    fft_spectrum = np.fft.rfft(samples_mono_for_analysis)
    fft_freq = np.fft.rfftfreq(len(samples_mono_for_analysis), 1.0/sr)
    magnitude = np.abs(fft_spectrum)
    
    bands = {'low': (20, 250), 'mid': (250, 2000), 'high': (2000, min(12000, sr/2.0 -1.0))} 
    energy_profile = {}

    for band_name, (low_freq, high_freq) in bands.items():
        idx = np.where((fft_freq >= low_freq) & (fft_freq < high_freq))[0]
        if len(idx) > 0:
            band_energy = np.mean(magnitude[idx]) 
        else:
            band_energy = 0
        energy_profile[band_name] = band_energy

    total_energy = np.sum(list(energy_profile.values()))
    if total_energy > 1e-9: # Avoid division by zero if silent
        for band_name in energy_profile:
            energy_profile[band_name] /= total_energy
    else: # If silent, no EQ adjustments needed based on energy
        return audio
    
    eq_bands_params = []
    # More sensitive thresholds for relative energy
    if energy_profile.get('low', 0) < 0.20: 
        eq_bands_params.append({'center_freq': 80, 'bandwidth': 120, 'gain': 2.5, 'sample_rate': sr})
    elif energy_profile.get('low', 0) > 0.45: # If low is too dominant
        eq_bands_params.append({'center_freq': 100, 'bandwidth': 150, 'gain': -1.5, 'sample_rate': sr})
        
    if energy_profile.get('mid', 0) < 0.30: 
        eq_bands_params.append({'center_freq': 1000, 'bandwidth': 1500, 'gain': 1.0, 'sample_rate': sr})
    elif energy_profile.get('mid', 0) > 0.55:
        eq_bands_params.append({'center_freq': 800, 'bandwidth': 1000, 'gain': -1.0, 'sample_rate': sr})
        
    if energy_profile.get('high', 0) < 0.15:
        eq_bands_params.append({'center_freq': 7000, 'bandwidth': 5000, 'gain': 2.0, 'sample_rate': sr})
    elif energy_profile.get('high', 0) > 0.40:
        eq_bands_params.append({'center_freq': 5000, 'bandwidth': 4000, 'gain': -1.5, 'sample_rate': sr})

    if not eq_bands_params:
        return audio 

    active_eq_bands = [ParametricEQBand(**params) for params in eq_bands_params]
    
    # The sum of (filtered and gained) bands will be the EQ effect.
    # This effect is then added to the original signal.
    # This is more akin to how parallel EQs or some graphic EQs work when bands are summed.
    sum_of_processed_bands = np.zeros_like(samples_original, dtype=np.float32)
    for band_filter in active_eq_bands:
        # band_filter.process now returns the isolated, gained band effect (or zeros if filter failed)
        sum_of_processed_bands += band_filter.process(samples_original.copy()) 

    # The result is the original signal plus the sum of adjustments from each band.
    # However, the previous logic was: final_samples = processed_samples_effect
    # This implies the sum of bands *is* the new signal, not an addition to original.
    # Let's stick to that model: the output is the sum of the processed bands.
    final_samples = sum_of_processed_bands

    final_samples_clipped = np.clip(final_samples, -32768, 32767).astype(np.int16)
    return audio._spawn(final_samples_clipped.tobytes())

# === PARAMETRIC EQ BAND CLASS ===
class ParametricEQBand:
    def __init__(self, center_freq, bandwidth, gain, sample_rate):
        self.center_freq = float(center_freq)
        self.bandwidth = float(bandwidth)
        self.gain_db = float(gain)
        self.sample_rate = float(sample_rate)
        self.b, self.a = self._create_band_filter()

    def _create_band_filter(self):
        nyquist = self.sample_rate / 2.0
        
        low_cutoff = self.center_freq - self.bandwidth / 2.0
        high_cutoff = self.center_freq + self.bandwidth / 2.0

        low_cutoff = max(1.0, low_cutoff) 
        high_cutoff = min(nyquist - 1.0, high_cutoff)

        if low_cutoff >= high_cutoff:
            # print(f"Warning: Invalid band parameters for ParametricEQBand (f={self.center_freq}, bw={self.bandwidth}). Low cutoff >= high cutoff.")
            return np.array([1.0]), np.array([1.0]) # Return pass-through filter

        wn_low = low_cutoff / nyquist
        wn_high = high_cutoff / nyquist
        
        wn_low = max(1e-6, min(wn_low, 1.0 - 1e-6)) # Clamp to (0, 1) exclusive
        wn_high = max(1e-6, min(wn_high, 1.0 - 1e-6))

        if wn_low >= wn_high:
            # print(f"Warning: Invalid normalized cutoffs for ParametricEQBand (f={self.center_freq}, bw={self.bandwidth}). wn_low >= wn_high.")
            return np.array([1.0]), np.array([1.0]) # Pass-through

        try:
            # For a band-specific gain effect as used in energy_based_eq (summing bands)
            # we use a bandpass filter.
            b, a = scipy.signal.butter(2, [wn_low, wn_high], btype='bandpass')
        except ValueError as e:
            # print(f"Warning: Could not create Butterworth filter for f={self.center_freq}, bw={self.bandwidth}. {e}")
            return np.array([1.0]), np.array([1.0]) # Pass-through
        return b, a

    def process(self, audio_samples): # audio_samples is a numpy array
        if np.array_equal(self.b, [1.0]) and np.array_equal(self.a, [1.0]): # Pass-through filter due to error or invalid params
             return np.zeros_like(audio_samples) # This band contributes nothing to the sum of filtered bands

        filtered_samples = scipy.signal.lfilter(self.b, self.a, audio_samples)
        gain_linear = 10 ** (self.gain_db / 20.0)
        processed_samples = filtered_samples * gain_linear
        return processed_samples

# Replace the old multiband_compress with this new one
def _apply_pydub_compression(segment, params):
    """Helper to apply compression and optional makeup gain to a Pydub AudioSegment."""
    if not segment or len(segment) == 0:
        return segment 
    
    comp_config = {
        'threshold': params.get('threshold', -20.0),
        'ratio': params.get('ratio', 4.0),
        'attack': params.get('attack', 5.0), 
        'release': params.get('release', 50.0) 
    }
    try:
        compressed_segment = segment.compress_dynamic_range(**comp_config)
        current_loudness = audio.dBFS # This is peak, not LUFS. Pydub doesn't have direct LUFS.
        # For actual LUFS, pyloudnorm is used in normalize_loudness.
        # The normalize_loudness function handles this.
        audio_normalized = normalize_loudness(audio.copy(), target_lufs)
        if audio_normalized is not None:
            audio = audio_normalized
        else:
            print("Warning: LUFS normalization returned None. Skipping normalization.")
            # Fallback or error handling if normalize_loudness fails
    except Exception as e:
        print(f"Error during LUFS Normalization: {e}. Skipping normalization.")


    # 6. Soft Limiting / True Peak Limiting
    print(f"Step 6: Applying Limiting to meet True Peak {true_peak} dBFS...")
    try:
        # Pydub's normalize can act as a simple peak limiter.
        # For more precise true peak limiting, dedicated tools are better.
        # This is a basic approach.
        
        # First, ensure we are not clipping excessively before trying to meet true peak.
        # If audio is already heavily compressed/limited, further gain reduction might be needed.
        # This is a simplified limiter. A real true peak limiter is more complex.
        
        # Calculate current peak
        current_peak_dbfs = audio.max_dBFS
        
        # Calculate gain adjustment needed to hit the true_peak target
        gain_to_meet_true_peak = true_peak - current_peak_dbfs
        
        if gain_to_meet_true_peak < 0: # Only apply if current peak exceeds target
            audio = audio.apply_gain(gain_to_meet_true_peak)
            print(f"Applied {gain_to_meet_true_peak:.2f} dB gain to meet true peak (current max: {audio.max_dBFS:.2f} dBFS).")
        else:
            print(f"Audio peak ({current_peak_dbfs:.2f} dBFS) is already at or below target true peak ({true_peak} dBFS). No peak limiting applied.")

    except Exception as e:
        print(f"Error during Peak Limiting: {e}. Skipping limiting.")

    # 7. Dithering (Conceptual - Pydub handles this during export for lower bit depths if needed)
    print("Step 7: Dithering (Handled by Pydub on export to relevant formats if necessary).")

    print("Classic mastering chain processing completed.")
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
def stem_split(input_file, output_dir, num_stems=2):
    """
    Split audio into stems using Demucs.
    stems: 2 (vocals/accompaniment), 4 (vocals, drums, bass, other), or 6 (htdemucs: vocals, drums, bass, guitar, piano, other)
    """
    try:
        # Check if demucs is installed
        import shutil
        if not shutil.which("demucs"):
            print("[ERROR] The 'demucs' command-line tool is required for stem splitting. Install it with: pip install demucs")
            return
    except Exception as e:
        print(f"[ERROR] Could not check for demucs: {e}")
        return
    # Select Demucs model based on stems
    if stems == 2:
        model = "htdemucs"  # htdemucs supports 2-stem with --two-stems
        stem_arg = f"--two-stems=vocals"
    elif stems == 4:
        model = "htdemucs"
        stem_arg = ""
    elif stems == 6:
        model = "htdemucs_6s"
        stem_arg = ""
    else:
        print("[ERROR] Stems must be 2, 4, or 6 for Demucs.")
        return
    try:
        cmd = [
            "demucs",
            "--model", model,
            input_file,
            "-o", output_dir
        ]
        if stem_arg:
            cmd.insert(2, stem_arg)
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[Demucs] Stems saved to '{output_dir}'")
    except Exception as e:
        print(f"[ERROR] Demucs stem splitting failed: {e}")

# === MASTERING (CLASSIC & AI) ===
def ai_master_audio(input_path, output_path, target_lufs, true_peak):
    """
    Placeholder for AI mastering logic.
    This function should ideally call an AI mastering service or local model.
    """
    print("AI mastering function called. This is a placeholder.")
    # Simulate AI processing by copying the input to output
    # In a real scenario, this would involve complex AI-driven audio processing
    try:
        # Example: Using a hypothetical AI mastering library or API call
        # from ai_mastering_library import AIMasteringService
        # service = AIMasteringService(api_key="YOUR_API_KEY")
        # result = service.master_track(input_path, target_lufs=target_lufs, true_peak=true_peak)
        # result.save(output_path)
        
        # For now, let's just copy the file to simulate a process
        import shutil
        shutil.copy(input_path, output_path)
        print(f"Placeholder AI mastering: copied {input_path} to {output_path}")
        # Simulate success/failure
        # return True 
        # To demonstrate fallback, let's return False
        print("AI mastering placeholder returning False to trigger fallback.")
        return False # Simulate failure to test fallback

    except ImportError:
        print("AI mastering library not found. Please install the required library.")
        return False
    except Exception as e:
        print(f"Error in AI mastering placeholder: {e}")
        return False

def master_audio(input_path, output_path, ai_mastering=False, target_lufs=-14.0, true_peak=-1.0, output_format='wav'):
    """
    Master an audio file with advanced chain, LUFS normalization, dithering, and optional AI mastering.
    """
    try:
        if ai_mastering:
            print("Attempting AI mastering...")
            success = ai_master_audio(input_path, output_path, target_lufs, true_peak)
            if success:
                print("AI mastering successful.")
                # Further processing or direct use of AI output
                audio = AudioSegment.from_file(output_path)
            else:
                print("AI mastering failed or is not fully implemented. Falling back to classic mastering.")
                audio = classic_mastering_chain(audio, target_lufs, true_peak)
        else:
            audio = classic_mastering_chain(audio, target_lufs, true_peak)
        # Classic mastering chain
        audio = AudioSegment.from_file(input_path)
        print(f"Loaded {input_path} (channels={audio.channels}, duration={audio.duration_seconds:.2f}s)")
        before_rms = audio.rms
        print(f"Input RMS: {before_rms}")
        mastered = apply_mastering_chain(audio)
        mastered = normalize_lufs(mastered, target=target_lufs)
        if dither:
            # Simple dithering: add low-level noise
            samples = np.array(mastered.get_array_of_samples()).astype(np.float32)
            noise = np.random.uniform(-1, 1, size=samples.shape) * 0.5
            samples = samples + noise
            samples = np.clip(samples, -32768, 32767).astype(np.int16)
            mastered = mastered._spawn(samples.tobytes())
        mastered = mastered.set_frame_rate(audio.frame_rate).set_channels(audio.channels)
        mastered.export(output_path, format=output_format)
        after_rms = mastered.rms
        print(f"Exported mastered file: {output_path}")
        print(f"Output RMS: {after_rms}")
    except Exception as e:
        print(f"[ERROR] Mastering failed: {e}")

# === PARAMETRIC EQ BAND CLASS ===
class ParametricEQBand:
    def __init__(self, center_freq, bandwidth, gain):
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.gain = gain
        # Assuming a default sample rate of 44100 for filter design.
        # This might need to be dynamic if handling audio with different sample rates.
        self.sample_rate = 44100 
        self.b, self.a = self._create_band_filter()

    def _create_band_filter(self):
        nyquist = self.sample_rate / 2
        low = (self.center_freq - self.bandwidth / 2) / nyquist
        high = (self.center_freq + self.bandwidth / 2) / nyquist
        
        # Ensure low and high are within valid range (0, 1)
        low = max(0.01, min(low, 0.99)) # Clamp to avoid issues at nyquist or 0
        high = max(0.01, min(high, 0.99))

        if low >= high: # Avoid issues if bandwidth is too large or center_freq is near extremes
            # Fallback to a simple gain adjustment or skip filtering for this band
            # For simplicity, returning a pass-through filter
            return [1], [1]
            
        # Using a 2nd order Butterworth filter
        b, a = scipy.signal.butter(2, [low, high], btype='bandpass')
        return b, a

    def process(self, audio_samples):
        if np.array_equal(self.b, [1]) and np.array_equal(self.a, [1]):
            # Pass-through filter, just apply gain
            return audio_samples * (10 ** (self.gain / 20))
            
        filtered_samples = scipy.signal.lfilter(self.b, self.a, audio_samples)
        # Apply gain to the filtered band
        processed_samples = filtered_samples * (10 ** (self.gain / 20))
        return processed_samples

# === CLI ===
def print_usage():
    print("""
AUDIOTOOLS.py - Unified Audio/Music Processing Toolkit

Usage:
  python AUDIOTOOLS.py watermark-detect <input.mid>
  python AUDIOTOOLS.py remove-vocal-noise <input.wav> <output.wav> [--noise-strength=0.5] [--silence-db=30] [--deepfilternet]
  python AUDIOTOOLS.py slice <input.wav> <segment_length_sec> <output_prefix>
  python AUDIOTOOLS.py normalize <folder>
  python AUDIOTOOLS.py bpm <input.wav>
  python AUDIOTOOLS.py key <input.wav>
  python AUDIOTOOLS.py midi2wav <input.mid> <output.wav> <soundfont.sf2>
  python AUDIOTOOLS.py trapbeat [output.mid]
  python AUDIOTOOLS.py master <input.wav> <output.wav> [--lufs=-14.0] [--format=wav] [--no-dither] [--ai-mastering]
  python AUDIOTOOLS.py stem-split <input.wav> <output_dir> [--stems=2|4|6]

Commands:
  watermark-detect   Detect watermark in MIDI file
  remove-vocal-noise Remove background noise and trim silence from vocal WAV (add --deepfilternet for DeepFilterNet)
  slice              Slice audio into segments
  normalize          Batch normalize all audio files in a folder
  bpm                Detect BPM of audio file
  key                Detect musical key of audio file
  midi2wav           Convert MIDI to WAV using a SoundFont
  trapbeat           Generate trap beat MIDI with watermark
  master             Master an audio file (advanced chain, LUFS, dither, format, AI option)
  stem-split         Split audio into stems using Demucs (2, 4, or 6 stems)
""")

def check_command_exists(command):
    """Checks if a command exists on the system path."""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=False, text=True) # Most CLI tools support --version
        return True
    except FileNotFoundError:
        try: # Fallback for commands that might not have --version or for minimal checks
            subprocess.run([command], capture_output=True, check=False, text=True, timeout=1)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    except Exception: # Catch any other exception during the check
        return False

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
            print("Usage: python AUDIOTOOLS.py remove-vocal-noise <input.wav> <output.wav> [--noise-strength=0.5] [--silence-db=30] [--deepfilternet]")
            sys.exit(1)
        input_wav = sys.argv[2]
        output_wav = sys.argv[3]
        noise_strength = 0.5
        silence_db = 30
        use_deepfilternet = False
        for i, arg in enumerate(sys.argv):
            if arg.startswith("--noise-strength="):
                noise_strength = float(arg.split("=")[1])
            if arg.startswith("--silence-db="):
                silence_db = float(arg.split("=")[1])
            if arg == "--deepfilternet":
                use_deepfilternet = True
        process_vocal(input_wav, output_wav, noise_strength=noise_strength, silence_db=silence_db, use_deepfilternet=use_deepfilternet)
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
            print("Usage: python AUDIOTOOLS.py master <input.wav> <output.wav> [--lufs=-14.0] [--format=wav] [--no-dither] [--ai-mastering]")
            sys.exit(1)
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        target_lufs = -14.0
        output_format = "wav"
        dither = True
        ai_mastering = False
        for arg in sys.argv[4:]:
            if arg.startswith("--lufs="):
                target_lufs = float(arg.split("=")[1])
            if arg.startswith("--format="):
                output_format = arg.split("=")[1]
            if arg == "--no-dither":
                dither = False
            if arg == "--ai-mastering":
                ai_mastering = True
        master_audio(input_file, output_file, ai_mastering=ai_mastering, target_lufs=target_lufs, output_format=output_format)
    elif cmd == "stem-split":
        if len(sys.argv) < 4:
            print("Usage: python AUDIOTOOLS.py stem-split <input.wav> <output_dir> [--stems=2|4|6]")
            sys.exit(1)
        input_file = sys.argv[2]
        output_dir = sys.argv[3]
        stems = 2
        for arg in sys.argv[4:]:
            if arg.startswith("--stems="):
                stems = int(arg.split("=")[1])
        stem_split(input_file, output_dir, num_stems=stems)
    else:
        print_usage()
        sys.exit(1)
