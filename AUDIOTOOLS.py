# === IMPORTS ===
import os
import random
import subprocess
import shutil # For shutil.copy and shutil.which
import librosa
import numpy as np # Ensure numpy is imported
import soundfile as sf
from pydub import AudioSegment, effects # pydub.effects.normalize is used
from pydub.silence import split_on_silence
from mido import MidiFile, Message, MidiTrack, MetaMessage, bpm2tempo # Explicitly import bpm2tempo
import scipy.signal # Ensure scipy.signal is imported
from scipy.ndimage import median_filter
from pathlib import Path
import argparse
import sys # IMPORT SYS
import mido
import math # ADDED IMPORT
import traceback # ADDED IMPORT

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
BPM = 140 # Default BPM
PPQ = 480 # Pulses Per Quarter note, common for MIDI
# NUM_BARS will be dynamically calculated
BAR_LENGTH = PPQ * 4  # 4/4 time signature, 1 bar = 4 quarter notes
FILENAME = 'trap_beat_generated.mid' # Default output filename
MINOR_NATURAL_SCALE = [0, 2, 3, 5, 7, 8, 10] # Relative intervals for a natural minor scale

# Instrument and Channel Mapping (0-indexed for Mido)
# General MIDI (GM) standard often puts percussion on channel 9 (0-indexed)
# We'll use separate tracks for clarity, and can assign GM program changes if needed.
INSTRUMENT_CHANNELS = {
    'Kick': 0,
    'Snare': 1,
    'Clap': 2,
    'HiHat_Closed': 3,
    'HiHat_Open': 4,
    'Tom_Low': 5,
    'Tom_Mid': 6,
    'Tom_High': 7,
    'Ride_Cymbal': 8, # Example, can be part of a drum kit on channel 9
    'Crash_Cymbal': 9, # Often channel 9 for drums
    'Bass_808': 10,
    'Melody_Lead': 11,
    'Melody_Harmony': 12,
    'Pads': 13,
    'FX': 14,
    'Watermark': 15 # Dedicated channel for watermark
}

# MIDI Note Numbers for common drum sounds (can vary by soundfont/DAW mapping)
# These are illustrative; actual sound depends on the playback device/soundfont.
# For GM standard drum map (channel 9), these are somewhat standard.
DRUM_NOTES = {
    'Kick': 36,       # Acoustic Bass Drum
    'Snare': 38,      # Acoustic Snare
    'Clap': 39,       # Hand Clap
    'HiHat_Closed': 42,# Closed Hi-Hat
    'HiHat_Open': 46, # Open Hi-Hat
    'Tom_Low': 41,    # Low Tom
    'Tom_Mid': 45,    # Mid Tom
    'Tom_High': 48,   # High Tom
    'Ride_Cymbal': 51,# Ride Cymbal 1
    'Crash_Cymbal': 49 # Crash Cymbal 1
}
# For 808, melody, pads, FX, we'll use pitched notes, not fixed drum notes.
# Base note for 808, e.g., C2 (MIDI note 36) or E1 (MIDI note 28)
BASE_808_NOTE = 28 # E1

def add_note(track, note, velocity, start_tick, duration, channel):
    # Ensure time values are integers for Mido, as per error "message time must be int"
    _start_tick = int(start_tick) 
    _duration = int(duration)

    # Clamp note and velocity to valid MIDI range
    note = max(0, min(127, note))
    velocity = max(0, min(127, velocity))
    if len(track) == 0:
        delta = _start_tick
    else:
        # Calculate delta time for the note_on message
        # This is relative to the previous message on this track.
        # Sum of all previous delta times on this track gives the absolute time of the last event
        current_abs_time_on_track = sum(msg.time for msg in track if hasattr(msg, 'time'))
        delta = max(0, _start_tick - current_abs_time_on_track)
    
    track.append(Message('note_on', note=note, velocity=velocity, time=delta, channel=channel))
    # The 'time' for note_off is its duration relative to the note_on.
    track.append(Message('note_off', note=note, velocity=0, time=_duration, channel=channel))

def add_inaudible_signature(mid, signature='TS'):
    # This function needs to be adapted. It currently tries to access mid.tracks[channel]
    # which assumes a fixed channel for the signature track.
    # Better to pass the specific track object for the signature.
    # For now, let's assume it will find/create the watermark track if needed.
    
    # Find or create the watermark track
    watermark_track = None
    watermark_channel_num = INSTRUMENT_CHANNELS['Watermark']

    # Check if a track is already named "Watermark"
    for t in mid.tracks:
        track_name_msg = next((msg for msg in t if msg.is_meta and msg.type == 'track_name'), None)
        if track_name_msg and track_name_msg.name == 'Watermark':
            watermark_track = t
            break
    
    if watermark_track is None: # If not found by name, try to get it by expected channel (less robust)
                                # Or, if we strictly use one track per instrument as set up in generate_full_trap_beat
                                # this function might not be needed if generate_full_trap_beat handles it.
                                # For now, let's assume the track exists and was prepared.
        # This part is tricky if generate_full_trap already created all tracks.
        # The original add_inaudible_signature took mid.tracks[channel]
        # Let's assume the watermark track is the one associated with INSTRUMENT_CHANNELS['Watermark']
        # This requires generate_full_trap_beat to have set up instrument_tracks correctly.
        # This function is now less standalone if it relies on generate_full_trap_beat's setup.
        # A cleaner way: generate_full_trap_beat passes the specific watermark track object.
        print("Warning: add_inaudible_signature couldn't definitively find the watermark track. Skipping.")
        return


    char_to_note = {c: (ord(c) % 32) + 24 for c in signature.upper()} # Shifted to a slightly higher, still low octave
    # channel = INSTRUMENT_CHANNELS['Watermark'] # Use the defined watermark channel
    velocity = 1 # Very low velocity
    duration = 1 # Minimal duration
    spacing = PPQ // 4 # Spacing between signature notes

    # Ensure tempo is set if this track is the first or only one (Mido needs tempo)
    # This is now handled in generate_full_trap_beat
    # if not any(msg.is_meta and msg.type == 'set_tempo' for msg in watermark_track):
    # watermark_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(BPM), time=0)) # Use global BPM as fallback

    current_tick_on_track = sum(msg.time for msg in watermark_track if hasattr(msg, 'time'))
    
    # Add signature notes
    for i, char in enumerate(signature):
        note = char_to_note.get(char, 24) # Default to a low C if char not found
        
        delta_time_for_note = 0
        if i == 0: # First note of signature
            # If track is empty, delta is 0. If not, it's relative to last event.
            # We want to place it at the beginning of the track effectively.
            # This is complex with Mido's delta times.
            # Simplification: assume this is called on a fresh or appropriately prepared track.
            # For now, just append with minimal delta or let add_note handle it.
             pass # Delta will be 0 if track is empty or first message

        # Calculate delta time for Mido
        # This needs to be relative to the *last event on this specific track*
        # The `add_note` function handles this if we use it.
        # Direct append:
        current_track_time = sum(msg.time for msg in watermark_track if hasattr(msg, 'time'))
        # We want to add notes sequentially with `spacing`
        target_abs_time_for_this_note = i * spacing 
        
        delta = max(0, target_abs_time_for_this_note - current_track_time)

        watermark_track.append(Message('note_on', note=note, velocity=velocity, time=delta, channel=INSTRUMENT_CHANNELS['Watermark']))
        watermark_track.append(Message('note_off', note=note, velocity=0, time=duration, channel=INSTRUMENT_CHANNELS['Watermark']))
    print(f"Inaudible signature '{signature}' added to track: {getattr(watermark_track, 'name', 'Unnamed Watermark Track')}")


def embed_watermark_in_music(track, num_bars, ppq_val, pattern=[60, 62, 64, 65, 67], velocity=2):
    """Embeds a watermark pattern into a specific MIDI track."""
    # Ensure track is a Mido Track object
    if not isinstance(track, MidiTrack):
        print("Error: embed_watermark_in_music expects a MidiTrack object.")
        return

    bar_len_ticks = ppq_val * 4
    duration_ticks = ppq_val // 8 # Short duration for watermark notes

    for i, note_val in enumerate(pattern):
        # Embed on a random beat within a random bar
        target_bar = random.randint(0, num_bars - 1)
        # Place on one of the 4 beats in the bar, with slight random offset
        beat_offset = random.choice([0, ppq_val, ppq_val*2, ppq_val*3])
        random_tick_offset = random.randint(-ppq_val // 8, ppq_val // 8)
        
        start_tick = (target_bar * bar_len_ticks) + beat_offset + random_tick_offset
        start_tick = max(0, start_tick) # Ensure no negative ticks

        # Add note using the existing add_note logic, but simplified for direct append
        # Assuming channel is already set for the track or will be handled by add_note if it takes channel
        # For watermark, channel is fixed (e.g. 15)
        
        # Get current time of the track to calculate delta
        current_track_time = sum(msg.time for msg in track if hasattr(msg, 'time'))
        delta_time = max(0, start_tick - current_track_time)

        track.append(Message('note_on', note=note_val, velocity=velocity, time=delta_time, channel=INSTRUMENT_CHANNELS['Watermark']))
        track.append(Message('note_off', note=note_val, velocity=0, time=duration_ticks, channel=INSTRUMENT_CHANNELS['Watermark']))
    print(f"Watermark embedded in track: {getattr(track, 'name', 'Unnamed Track')}")


def generate_full_trap_beat(filename=FILENAME, bpm=BPM, key_root_note=60): # key_root_note C4
    """
    Generates a more complete trap beat MIDI file with specified duration, BPM, key,
    instrument track names, and watermarking.
    """
    print(f"Generating trap beat: {filename} at {bpm} BPM...")

    mid = MidiFile(ticks_per_beat=PPQ)

    # --- Duration Calculation ---
    min_duration_seconds = 1*60 + 50 # 1 min 50 sec
    max_duration_seconds = 3*60 + 50 # 3 min 50 sec

    seconds_per_beat = 60.0 / bpm
    ticks_per_second = PPQ / seconds_per_beat
    ticks_per_bar = PPQ * 4 # Assuming 4/4

    min_total_ticks = min_duration_seconds * ticks_per_second
    max_total_ticks = max_duration_seconds * ticks_per_second

    min_bars = math.ceil(min_total_ticks / ticks_per_bar)
    max_bars = math.floor(max_total_ticks / ticks_per_bar)

    if max_bars < min_bars:
        max_bars = min_bars # Ensure at least min_bars
    if min_bars < 4: # Ensure a minimum reasonable length for structure
        min_bars = 4
        if max_bars < min_bars: max_bars = min_bars
        
    actual_num_bars = random.randint(min_bars, max_bars)
    total_ticks = actual_num_bars * ticks_per_bar
    total_duration_seconds = total_ticks / ticks_per_second
    print(f"Target duration: {min_duration_seconds}-{max_duration_seconds}s. Actual: {total_duration_seconds:.2f}s ({actual_num_bars} bars)")

    # --- Track Setup ---
    instrument_tracks = {}
    track_instrument_names = {} # To store names for final output

    for name, channel_num in INSTRUMENT_CHANNELS.items():
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('track_name', name=name, time=0))
        # Optional: Add program change for GM compatibility if desired
        # if name not in ['Kick', 'Snare', 'Clap', 'HiHat_Closed', 'HiHat_Open', 'Tom_Low', 'Tom_Mid', 'Tom_High', 'Ride_Cymbal', 'Crash_Cymbal', 'Watermark']:
        #     # Example: Acoustic Grand Piano for Melody_Lead
        #     program = 0 # Default to piano, adjust as needed
        #     if name == 'Bass_808': program = 33 # Acoustic Bass or 34 Fretless Bass
        #     elif name == 'Melody_Lead': program = 81 # Lead 2 (sawtooth)
        #     elif name == 'Pads': program = 89 # Pad 1 (new age)
        #     track.append(Message('program_change', program=program, time=0, channel=channel_num))
        instrument_tracks[name] = track
        track_instrument_names[channel_num] = name


    # Set tempo (once, in the first track or a dedicated meta track)
    # Mido handles tempo globally if it's in any track before notes.
    # Let's put it in the first created track (Kick)
    instrument_tracks['Kick'].append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm), time=0))
    instrument_tracks['Kick'].append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))


    # --- Music Generation Logic ---
    # This is a simplified example. A real generator would be more complex.
    
    # Scale for melody and bass (e.g., C minor)
    current_scale = [key_root_note + interval for interval in MINOR_NATURAL_SCALE]
    # Extend scale to cover more octaves
    full_scale = []
    for octave_offset in [-12, 0, 12]:
        full_scale.extend([note + octave_offset for note in current_scale])
    full_scale = sorted(list(set(n for n in full_scale if 0 <= n <= 127)))


    for bar in range(actual_num_bars):
        bar_start_tick = bar * ticks_per_bar

        # ** Kick ** (Channel INSTRUMENT_CHANNELS['Kick'])
        # Common pattern: 1, (optional 1.5), and 3
        add_note(instrument_tracks['Kick'], DRUM_NOTES['Kick'], 120, bar_start_tick, PPQ // 2, INSTRUMENT_CHANNELS['Kick'])
        if bar % 2 == 0: # Add some variation
             add_note(instrument_tracks['Kick'], DRUM_NOTES['Kick'], 110, bar_start_tick + PPQ * 1.5, PPQ // 4, INSTRUMENT_CHANNELS['Kick'])
        add_note(instrument_tracks['Kick'], DRUM_NOTES['Kick'], 115, bar_start_tick + PPQ * 2, PPQ // 2, INSTRUMENT_CHANNELS['Kick'])
        if random.random() < 0.3: # Occasional kick on 4th beat
            add_note(instrument_tracks['Kick'], DRUM_NOTES['Kick'], 100, bar_start_tick + PPQ * 3.5, PPQ // 4, INSTRUMENT_CHANNELS['Kick'])

        # ** Snare / Clap ** (Channels 1, 2) - usually on 2 and 4
        snare_vel = random.randint(100, 127)
        add_note(instrument_tracks['Snare'], DRUM_NOTES['Snare'], snare_vel, bar_start_tick + PPQ, PPQ // 2, INSTRUMENT_CHANNELS['Snare'])
        add_note(instrument_tracks['Clap'], DRUM_NOTES['Clap'], random.randint(90,110), bar_start_tick + PPQ, PPQ //2, INSTRUMENT_CHANNELS['Clap']) # Layer clap

        snare_vel_2 = random.randint(100, 127)
        add_note(instrument_tracks['Snare'], DRUM_NOTES['Snare'], snare_vel_2, bar_start_tick + PPQ * 3, PPQ // 2, INSTRUMENT_CHANNELS['Snare'])
        if random.random() < 0.7: # Layer clap on 4th beat too
            add_note(instrument_tracks['Clap'], DRUM_NOTES['Clap'], random.randint(90,110), bar_start_tick + PPQ * 3, PPQ //2, INSTRUMENT_CHANNELS['Clap'])


        # ** Hi-Hats ** (Channel 3) - 8th notes with swing and rolls
        swing_factor = 0.0 # 0 for no swing, ~0.1-0.2 for light swing on 16ths
        eighth_note_len = PPQ // 2
        for i in range(8): # 8 eighth notes per bar
            hat_start_tick = bar_start_tick + i * eighth_note_len
            hat_vel = random.randint(70, 100)
            hat_note = DRUM_NOTES['HiHat_Closed']
            hat_duration = eighth_note_len // 2 # Default short

            # Apply swing: delay every other 8th note's 16th position
            if i % 2 == 1 : # Apply to 2nd, 4th, 6th, 8th eighth notes
                 hat_start_tick += int(eighth_note_len * swing_factor)
            
            # Rolls and variations
            if random.random() < 0.15 and i < 7 : # Chance of a 16th note roll
                add_note(instrument_tracks['HiHat_Closed'], hat_note, hat_vel - 10, hat_start_tick, eighth_note_len // 4, INSTRUMENT_CHANNELS['HiHat_Closed'])
                add_note(instrument_tracks['HiHat_Closed'], hat_note, hat_vel - 15, hat_start_tick + eighth_note_len // 4, eighth_note_len // 4, INSTRUMENT_CHANNELS['HiHat_Closed'])
                if random.random() < 0.5: # even faster roll part
                    add_note(instrument_tracks['HiHat_Closed'], hat_note, hat_vel - 20, hat_start_tick + eighth_note_len // 2, eighth_note_len // 8, INSTRUMENT_CHANNELS['HiHat_Closed'])
                    add_note(instrument_tracks['HiHat_Closed'], hat_note, hat_vel - 25, hat_start_tick + eighth_note_len // 2 + eighth_note_len // 8, eighth_note_len // 8, INSTRUMENT_CHANNELS['HiHat_Closed'])
                i +=1 # Skip next 8th note position due to roll
            elif random.random() < 0.1: # Chance of an open hat
                add_note(instrument_tracks['HiHat_Open'], DRUM_NOTES['HiHat_Open'], hat_vel + 10, hat_start_tick, eighth_note_len, INSTRUMENT_CHANNELS['HiHat_Open'])
            else: # Standard closed hat
                add_note(instrument_tracks['HiHat_Closed'], hat_note, hat_vel, hat_start_tick, hat_duration, INSTRUMENT_CHANNELS['HiHat_Closed'])
        
        # ** 808 Bass ** (Channel 10)
        # Follows kick roughly, plays notes from the scale
        if bar % 4 == 0 or random.random() < 0.7: # Change 808 note every 4 bars or so
            bass_note = random.choice([n for n in full_scale if 20 <= n <= 50]) # Low notes
        
        # Pattern 1: on kick
        add_note(instrument_tracks['Bass_808'], bass_note, 100, bar_start_tick, PPQ, INSTRUMENT_CHANNELS['Bass_808'])
        if bar % 2 == 0:
            add_note(instrument_tracks['Bass_808'], bass_note, 90, bar_start_tick + PPQ * 1.5, PPQ * 0.5, INSTRUMENT_CHANNELS['Bass_808'])
        add_note(instrument_tracks['Bass_808'], bass_note, 95, bar_start_tick + PPQ * 2, PPQ, INSTRUMENT_CHANNELS['Bass_808'])
        
        # ** Simple Melody ** (Channel 11) - very basic
        if bar % 2 == 0: # Melody phrase every 2 bars
            for i in range(random.randint(1,3)): # 1 to 3 notes in the phrase
                melody_note = random.choice([n for n in full_scale if 55 <= n <= 80])
                melody_start = bar_start_tick + i * (PPQ * random.choice([0.5, 1, 1.5])) + random.randint(0, PPQ//4)
                melody_duration = PPQ * random.choice([0.5, 1, 1.5, 2])
                melody_vel = random.randint(70,90)
                add_note(instrument_tracks['Melody_Lead'], melody_note, melody_vel, melody_start, melody_duration, INSTRUMENT_CHANNELS['Melody_Lead'])

    # --- Watermarking ---
    # Add inaudible signature to the dedicated watermark track
    # add_inaudible_signature(mid, signature='AUDIOTOOLS') # mid object is passed
    # The add_inaudible_signature function needs to be adapted or called differently if it expects tracks directly

    # Embed audible low-velocity watermark pattern
    # This function expects a track object, num_bars, and ppq
    embed_watermark_in_music(instrument_tracks['Watermark'], actual_num_bars, PPQ, pattern=WATERMARK_PATTERN, velocity=1)


    # --- Save MIDI File ---
    try:
        mid.save(filename)
        print(f"Successfully generated MIDI file: {filename}")
        print("Track Instrument Assignments:")
        for i, track in enumerate(mid.tracks):
            # Attempt to get track name from MetaMessage
            track_name_msg = next((msg for msg in track if msg.is_meta and msg.type == 'track_name'), None)
            name_to_print = track_name_msg.name if track_name_msg else f"Track {i} (Unnamed)"
            
            # Try to find which instrument this track corresponds to via our INSTRUMENT_CHANNELS
            # This is a bit indirect; a better way is to store this mapping when creating tracks.
            assigned_instrument = "Unknown"
            for inst_name, ch_num in INSTRUMENT_CHANNELS.items():
                # Heuristic: if the track name meta message matches our instrument name
                if track_name_msg and track_name_msg.name == inst_name:
                    assigned_instrument = inst_name
                    break
                # Fallback: check if notes are on the expected channel (less reliable if channels are reused)
                # For this generator, we assume one instrument per track/channel as defined in INSTRUMENT_CHANNELS
            
            print(f"  MIDI Track {i}: {name_to_print}")

    except Exception as e:
        print(f"Error saving MIDI file {filename}: {e}")

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
    """Checks if a command exists on the system."""
    # Ensure shutil is imported (it's imported at the top of the provided file)
    return shutil.which(command) is not None

# Placeholder for ai_master_audio
def ai_master_audio(input_path, output_path, target_lufs=-14.0, true_peak=-1.0):
    print(f"Simulating AI mastering for {input_path} to {output_path}...")
    print(f"Target LUFS: {target_lufs}, True Peak: {true_peak}")
    try:
        audio = AudioSegment.from_file(input_path)
        processed_audio = effects.normalize(audio) # Basic normalization as a stand-in

        # Crude LUFS and peak simulation (not accurate, for placeholder purposes)
        current_lufs_approx = processed_audio.dBFS 
        gain_to_target_lufs = target_lufs - current_lufs_approx
        if np.isfinite(gain_to_target_lufs): # Check for NaN/inf if dBFS is problematic (e.g. for silence)
             processed_audio = processed_audio.apply_gain(gain_to_target_lufs)
        else:
            print(f"Warning: Could not calculate gain for LUFS adjustment (current_lufs_approx: {current_lufs_approx}). Skipping LUFS gain.")

        peak_headroom = abs(true_peak) if true_peak <= 0 else 0.0
        processed_audio = effects.normalize(processed_audio, headroom=peak_headroom)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        processed_audio.export(output_path, format=Path(output_path).suffix[1:])
        print(f"AI mastering simulation complete. Output: {output_path}")
        return True
    except Exception as e:
        print(f"Error during AI mastering simulation: {e}")
        try:
            # Fallback: copy original if AI fails
            Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Ensure dir for copy
            shutil.copy(input_path, output_path)
            print(f"AI mastering simulation failed, copied original to {output_path} as fallback.")
            return True # Reporting success for the operation of copying as fallback
        except Exception as copy_e:
            print(f"Error copying original file during AI fallback: {copy_e}")
            return False

# stem_split function using Demucs
def stem_split(input_path, output_dir, num_stems=4):
    print(f"Starting stem separation for {input_path} into {output_dir} ({num_stems} stems)...")
    if not check_command_exists("demucs"):
        print("Error: demucs command not found. Please ensure Demucs is installed and in your PATH.")
        print("Installation: pip install -U demucs")
        return False

    # Ensure output_dir is a Path object and exists
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Determine Demucs model/command based on num_stems
    # Defaulting to htdemucs for 4 stems if not 2 or 6.
    if num_stems == 2:
        model_flag = "--two-stems" # Uses a 2-stem model (e.g., mdx_extra_q with recent demucs CLI)
    elif num_stems == 6:
        model_flag = "--htdemucs_6s" # 6-stem htdemucs model
    elif num_stems == 4:
        model_flag = "--htdemucs" # 4-stem htdemucs model
    else:
        print(f"Warning: Invalid number of stems '{num_stems}'. Defaulting to 4 stems (htdemucs model).")
        model_flag = "--htdemucs"
        num_stems = 4 # Update for message consistency

    # Demucs expects string paths for command line arguments
    cmd = ["demucs", model_flag, "-o", str(output_dir_path), str(Path(input_path).resolve())]

    print(f"Using Demucs for {num_stems} stems. Command: {' '.join(cmd)}")

    try:
        # Using text=True for automatic decoding, errors='replace' for robustness
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors='replace')
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"Demucs processing successful. Stems saved in subdirectories within '{output_dir_path}'.")
            # Demucs creates a subdirectory named after the model (e.g., htdemucs),
            # then another subdirectory named after the input file, where stems are saved.
            print("Please check the specified output directory for the separated stems.")
            if stdout: print(f"STDOUT:\\n{stdout}")
            # Demucs often prints progress and info to stderr, even on success
            if stderr: print(f"STDERR:\\n{stderr}")
            return True
        else:
            print(f"Error during Demucs processing. Return code: {process.returncode}")
            if stdout: print(f"STDOUT:\\n{stdout}")
            if stderr: print(f"STDERR:\\n{stderr}")
            return False
    except FileNotFoundError:
        print("Error: demucs command not found. Is it installed and in your PATH? Try: pip install -U demucs")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during stem splitting: {e}")
        import traceback
        traceback.print_exc()
        return False

# KEY_MIDI_OFFSETS for trap beat generator key parsing
KEY_MIDI_OFFSETS = {
    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3, 'E': 4, 'F': 5,
    'F#': 6, 'GB': 6, 'G': 7, 'G#': 8, 'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11
}

# === MASTERING CHAIN ===

class ParametricEQBand:
    def __init__(self, center_freq, q, gain_db, band_type='peak', sample_rate=44100):
        self.center_freq = center_freq
        self.q = q
        self.gain_db = gain_db
        self.band_type = band_type.lower()
        self.sample_rate = sample_rate
        # Ensure scipy.signal is available
        try:
            from scipy import signal
            self.signal = signal
        except ImportError:
            self.signal = None
            print("Warning: scipy.signal not found. ParametricEQBand will not function.")
            # Raise an error or handle gracefully if scipy.signal is critical
            # For now, proceed and let apply fail if signal is None

        if self.signal:
            self.b, self.a = self._design_filter()
        else:
            self.b, self.a = np.array([1]), np.array([1]) # Pass-through if no scipy

    def _design_filter(self):
        if not self.signal:
            return np.array([1]), np.array([1]) # Pass-through

        # RBJ Audio EQ Cookbook formulas
        A = 10**(self.gain_db / 20.0)
        w0 = 2 * np.pi * self.center_freq / self.sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * self.q)

        if self.band_type == 'peak':
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
        elif self.band_type == 'low_shelf':
            # sqrt(A) terms for shelf filters
            sqrt_A_alpha_x2 = 2 * np.sqrt(A) * alpha
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + sqrt_A_alpha_x2)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - sqrt_A_alpha_x2)
            a0 = (A + 1) + (A - 1) * cos_w0 + sqrt_A_alpha_x2
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - sqrt_A_alpha_x2
        elif self.band_type == 'high_shelf':
            sqrt_A_alpha_x2 = 2 * np.sqrt(A) * alpha
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + sqrt_A_alpha_x2)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - sqrt_A_alpha_x2)
            a0 = (A + 1) - (A - 1) * cos_w0 + sqrt_A_alpha_x2
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0) # Note: Sign difference from low_shelf's a1
            a2 = (A + 1) - (A - 1) * cos_w0 - sqrt_A_alpha_x2
        else: # Default to peak
            print(f"Warning: Unknown band_type '{self.band_type}'. Defaulting to 'peak'.")
            b0 = 1 + alpha * A; b1 = -2 * cos_w0; b2 = 1 - alpha * A
            a0 = 1 + alpha / A; a1 = -2 * cos_w0; a2 = 1 - alpha / A
        
        if a0 == 0: # Avoid division by zero if filter design leads to it (e.g. extreme params)
            print(f"Warning: a0 is zero in filter design for f={self.center_freq}, Q={self.q}, gain={self.gain_db}. Using pass-through.")
            return np.array([1.0]), np.array([1.0])
            
        return np.array([b0/a0, b1/a0, b2/a0]), np.array([1, a1/a0, a2/a0])

    def apply(self, audio_data):
        if not self.signal:
            print("  EQ Band: SciPy signal not available, skipping filter application.")
            return audio_data
        if np.array_equal(self.b, [1.0]) and np.array_equal(self.a, [1.0]): # Pass-through filter
             return audio_data

        if audio_data.ndim == 1: # Mono
            return self.signal.lfilter(self.b, self.a, audio_data)
        elif audio_data.ndim == 2: # Stereo (assuming channels first: (C, N))
            processed_channels = []
            for i in range(audio_data.shape[0]):
                processed_channels.append(self.signal.lfilter(self.b, self.a, audio_data[i, :]))
            return np.array(processed_channels)
        else:
            raise ValueError("Audio data must be 1D (mono) or 2D (stereo, channels first).")

def multiband_compress(audio_segment, sample_rate, bands=None, threshold_db=-20.0, ratio=4.0, attack_ms=5, release_ms=100):
    from pydub.effects import compress_dynamic_range
    if bands is None: 
        bands = [
            (0, 250), (250, 1000), (1000, 4000), (4000, min(20000, sample_rate / 2 -1))
        ]
    
    print(f"Applying multiband compression with {len(bands)} bands.")
    original_channels = audio_segment.split_to_mono()
    processed_mono_channels = []

    for channel_idx, channel_audio in enumerate(original_channels):
        processed_bands_for_channel = []
        for i, (low_freq, high_freq) in enumerate(bands):
            # Ensure high_freq does not exceed Nyquist
            high_freq = min(high_freq, sample_rate / 2 - 1) 
            if low_freq >= high_freq: # Skip invalid bands
                print(f"    Skipping invalid band {i+1} for channel {channel_idx}: {low_freq} Hz - {high_freq} Hz")
                continue
            
            print(f"  Channel {channel_idx}, Band {i+1}: {low_freq} Hz - {high_freq} Hz")
            
            band_audio = channel_audio
            if low_freq > 0 and high_freq < (sample_rate / 2 -1) : # Band-pass
                 band_audio = channel_audio.high_pass_filter(low_freq).low_pass_filter(high_freq)
            elif low_freq == 0 and high_freq < (sample_rate / 2 -1): # Low-pass
                band_audio = channel_audio.low_pass_filter(high_freq)
            elif low_freq > 0 and high_freq >= (sample_rate / 2 -1): # High-pass
                band_audio = channel_audio.high_pass_filter(low_freq)
            # else: full band, no filtering (should not happen with typical band definitions)

            if len(band_audio) > 0:
                compressed_band = compress_dynamic_range(
                    band_audio, threshold=threshold_db, ratio=ratio,
                    attack=attack_ms, release=release_ms
                )
                processed_bands_for_channel.append(compressed_band)
            else:
                print(f"    Warning: Band {i+1} for channel {channel_idx} is empty after filtering.")

        if not processed_bands_for_channel:
            print(f"    Warning: No valid bands processed for channel {channel_idx}. Using original channel audio.")
            processed_mono_channels.append(channel_audio)
            continue

        final_channel_audio = processed_bands_for_channel[0]
        for k in range(1, len(processed_bands_for_channel)):
            # Ensure overlay is possible (e.g. if one band became shorter, pad it)
            # Pydub's overlay handles duration differences by taking length of the first.
            final_channel_audio = final_channel_audio.overlay(processed_bands_for_channel[k])
        
        processed_mono_channels.append(final_channel_audio)

    if not processed_mono_channels: return audio_segment 

    if len(processed_mono_channels) == 2:
        final_audio = AudioSegment.from_mono_audiosegments(processed_mono_channels[0], processed_mono_channels[1])
    elif len(processed_mono_channels) == 1:
        final_audio = processed_mono_channels[0]
    else: 
        print("Warning: Multiband compression resulted in unexpected channel count. Returning original.")
        return audio_segment
        
    print("Multiband compression applied.")
    return final_audio

def energy_based_eq(audio_data_np, sample_rate, n_bands=5, target_slope_db_oct=-2.0):
    # audio_data_np is expected as (channels, samples) or (samples,)
    print(f"Applying energy-based EQ: {n_bands} bands, target slope {target_slope_db_oct} dB/oct.")
    
    y_mono_for_analysis = librosa.to_mono(audio_data_np) if audio_data_np.ndim == 2 else audio_data_np
        
    D = librosa.stft(y_mono_for_analysis)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sample_rate)
    
    # Define frequency band edges (logarithmic-like spacing)
    if n_bands == 3:
        band_edges_hz = [20, 250, 4000, min(20000, sample_rate / 2.0 - 1)]
    elif n_bands == 5:
        band_edges_hz = [20, 120, 500, 2000, 8000, min(20000, sample_rate / 2.0 - 1)]
    else:
        print(f"Unsupported n_bands={n_bands}, defaulting to 5 bands for EQ.")
        band_edges_hz = [20, 120, 500, 2000, 8000, min(20000, sample_rate / 2.0 - 1)]

    band_energies_db = []
    band_center_freqs = []

    for i in range(len(band_edges_hz) - 1):
        low_hz, high_hz = band_edges_hz[i], band_edges_hz[i+1]
        idx = (freqs >= low_hz) & (freqs < high_hz)
        if np.any(idx):
            band_mean_db = np.mean(S_db[idx, :]) # Mean energy in dB over time for frequencies in band
            band_energies_db.append(band_mean_db)
            center_f = np.sqrt(low_hz * high_hz) if low_hz > 0 else high_hz / np.sqrt(2)
            band_center_freqs.append(center_f)
        else: # Should not happen with good band_edges_hz and nyquist limit
            band_energies_db.append(-120) 
            band_center_freqs.append((low_hz + high_hz) / 2.0)

    if len(band_center_freqs) < 2:
        print("  Warning: Not enough frequency bands for slope calculation. Skipping EQ.")
        return audio_data_np

    log_freqs = np.log2(np.maximum(1e-6, band_center_freqs)) # Avoid log(0) or negative
    
    # Target energies: E_target[i] = E_actual[0] + slope * (log_freqs[i] - log_freqs[0])
    target_energies_db = [band_energies_db[0] + target_slope_db_oct * (log_freqs[i] - log_freqs[0]) for i in range(len(band_energies_db))]
    gain_adjustments_db = [np.clip(t - a, -12, 12) for t, a in zip(target_energies_db, band_energies_db)] # Clip adjustments
    
    print(f"  EQ Band Centers (Hz): {[round(f,0) for f in band_center_freqs]}")
    print(f"  EQ Gain Adjustments (dB): {[round(g,1) for g in gain_adjustments_db]}")

    processed_audio = audio_data_np.copy()
    default_q = 1.414 # Q for general tone shaping, sqrt(2)

    for i in range(len(band_center_freqs)):
        center_freq = band_center_freqs[i]
        gain_db = gain_adjustments_db[i]
        
        if abs(gain_db) < 0.2 or center_freq <= 20 or center_freq >= (sample_rate / 2.0 -1):
            continue # Skip negligible or out-of-range EQs

        eq_band = ParametricEQBand(center_freq=center_freq, q=default_q, gain_db=gain_db, band_type='peak', sample_rate=sample_rate)
        processed_audio = eq_band.apply(processed_audio) # apply handles mono/stereo
        
    print("Energy-based EQ applied.")
    return processed_audio

def normalize_loudness(audio_segment, target_lufs=-14.0):
    try:
        from pyloudnorm import Meter
    except ImportError:
        print("  Warning: pyloudnorm library not found. Skipping LUFS normalization. Install with: pip install pyloudnorm")
        return audio_segment

    print(f"Normalizing loudness to {target_lufs} LUFS.")
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    
    # Convert to float range for pyloudnorm if not already
    if audio_segment.sample_width == 2: samples /= 32768.0 
    elif audio_segment.sample_width == 1: samples = (samples - 128.0) / 128.0
    elif audio_segment.sample_width == 4: samples /= 2147483648.0 # Assuming int32
    # If already float, it might be fine, but pyloudnorm expects data in range [-1, 1]

    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2)) # (N, C) for pyloudnorm
    
    meter = Meter(rate=audio_segment.frame_rate)
    try:
        # Pyloudnorm expects data as (samples, channels) or (samples,) for mono
        loudness_lufs = meter.integrated_loudness(samples) 
    except Exception as e:
        print(f"  Warning: Could not measure loudness with pyloudnorm: {e}. Skipping LUFS normalization.")
        return audio_segment

    if not np.isfinite(loudness_lufs) or loudness_lufs < -70: # Check for silence or very low levels
        print(f"  Warning: Measured loudness is very low or non-finite ({loudness_lufs:.2f} LUFS). Skipping LUFS normalization to avoid extreme gain.")
        return audio_segment

    print(f"  Measured loudness: {loudness_lufs:.2f} LUFS")
    gain_db = target_lufs - loudness_lufs
    
    if abs(gain_db) < 0.1:
        print("  Loudness already close to target. No adjustment made.")
        return audio_segment

    print(f"  Applying gain: {gain_db:.2f} dB")
    return audio_segment.apply_gain(gain_db)

def apply_true_peak_limiting(audio_segment, true_peak_db=-1.0):
    from pydub.effects import normalize as pydub_normalize
    print(f"Applying (digital) peak limiting to approx {true_peak_db} dBFS.")
    headroom = abs(true_peak_db) if true_peak_db <= 0 else 0.0
    limited_audio = pydub_normalize(audio_segment, headroom=headroom)
    print(f"  Peak limiting applied. New max dBFS: {limited_audio.max_dBFS:.2f}")
    return limited_audio

def apply_dithering(audio_segment, bit_depth=16):
    print(f"Dithering step (conceptual for {bit_depth}-bit output). Pydub handles bit depth on export.")
    # Actual dither is complex. Pydub's export might do some form of quantization.
    # For high quality, an external tool or dedicated library function would be needed.
    return audio_segment 

def classic_mastering_chain(input_path, output_path, target_lufs=-14.0, true_peak=-1.0, output_format="wav", apply_dither_flag=True):
    print(f"Starting classic mastering chain for {input_path}...")
    try:
        audio = AudioSegment.from_file(input_path)
        print(f"  Loaded: {len(audio)/1000:.2f}s, {audio.frame_rate}Hz, {audio.channels}ch, {audio.sample_width*8}-bit")
        
        # Convert to NumPy float array (channels, samples) for SciPy effects
        # Normalize to [-1.0, 1.0]
        samples_float = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.sample_width == 1: samples_float = (samples_float - 128.) / 128.
        elif audio.sample_width == 2: samples_float /= 32768.
        elif audio.sample_width == 4: samples_float /= 2147483648. # Assuming int32, not float32 from file
        # Add other sample widths if necessary

        if audio.channels == 1:
            audio_data_np = samples_float.reshape(1, -1) # (1, N)
        else: # Assuming stereo if not mono
            audio_data_np = samples_float.reshape(-1, audio.channels).T # (C, N)
        
        sample_rate = audio.frame_rate

        # 1. Energy-Based EQ
        audio_data_eq = energy_based_eq(audio_data_np, sample_rate, n_bands=5, target_slope_db_oct=-1.5)
        
        # Convert back to Pydub AudioSegment for Pydub-based effects
        # Rescale to 16-bit int for pydub processing (common intermediate)
        processed_samples_int16 = (np.clip(audio_data_eq, -1.0, 1.0) * 32767).astype(np.int16)
        
        # Pydub expects samples in (sample, sample, sample...) for mono
        # or (L, R, L, R, ...) for stereo.
        # Our processed_samples_int16 is (C, N). Need to interleave if stereo.
        if audio.channels == 1:
            data_for_pydub = processed_samples_int16.tobytes()
        else: # Stereo
            data_for_pydub = processed_samples_int16.T.tobytes() # Transpose to (N,C) then flatten

        audio_after_eq = AudioSegment(
            data_for_pydub, frame_rate=sample_rate,
            sample_width=2, # 16-bit
            channels=audio.channels
        )

        # 2. Multiband Compression
        audio_after_mb_comp = multiband_compress(audio_after_eq, sample_rate, 
                                                 threshold_db=-20.0, ratio=3.5, attack_ms=8, release_ms=120)
        
        # 3. LUFS Normalization
        audio_normalized_lufs = normalize_loudness(audio_after_mb_comp, target_lufs=target_lufs)

        # 4. True Peak Limiting (Digital Peak Limiting with Pydub)
        audio_limited = apply_true_peak_limiting(audio_normalized_lufs, true_peak_db=true_peak)

        # 5. Dithering (Conceptual)
        audio_final = apply_dithering(audio_limited) if apply_dither_flag else audio_limited
        if not apply_dither_flag: print("  Skipping dithering.")

        # 6. Export
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        export_params = {"format": output_format}
        if output_format == "mp3": export_params["bitrate"] = "320k"
        
        audio_final.export(output_path, **export_params)
        print(f"Classic mastering complete. Output: {output_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Classic mastering failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(input_path, output_path)
            print(f"Classic mastering failed, copied original to {output_path} as fallback.")
            return True 
        except Exception as copy_e:
            print(f"Error copying original file during classic mastering fallback: {copy_e}")
            return False

def master_audio(input_path, output_path, target_lufs=-14.0, true_peak=-1.0, output_format="wav", no_dither=False, ai_mastering_flag=False):
    if ai_mastering_flag:
        print("AI Mastering selected.")
        # Check for ffmpeg (pydub might need it)
        if not check_command_exists("ffmpeg"):
            print("Warning: ffmpeg not found. AI mastering might have issues with some audio formats.")
        # Check for pyloudnorm (ai_master_audio simulation uses it)
        try:
            import pyloudnorm # Check if pyloudnorm is available
        except ImportError:
            print("Warning: pyloudnorm not found. AI mastering LUFS features may be limited. (pip install pyloudnorm)")
        return ai_master_audio(input_path, output_path, target_lufs=target_lufs, true_peak=true_peak) # ai_master_audio is already defined
    else:
        print("Classic Mastering selected.")
        missing_deps = []
        if not check_command_exists("ffmpeg"): missing_deps.append("ffmpeg (for pydub format handling)")
        try:
            import pyloudnorm # Check if pyloudnorm is available
        except ImportError:
            missing_deps.append("pyloudnorm (for LUFS measurement; pip install pyloudnorm)")
        try:
            from scipy import signal # Check if scipy.signal is available
        except ImportError:
            missing_deps.append("scipy (for EQ filters; pip install scipy)")

        if missing_deps:
            print(f"Error: Missing dependencies for classic mastering: {', '.join(missing_deps)}.")
            print("Attempting to copy original file as fallback.")
            try:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(input_path, output_path)
                print(f"Copied original to {output_path}.")
                return True 
            except Exception as copy_e:
                print(f"Error copying original file: {copy_e}")
                return False
        
        return classic_mastering_chain(input_path, output_path, 
                                     target_lufs=target_lufs, true_peak=true_peak, 
                                     output_format=output_format, apply_dither_flag=not no_dither)

# === MAIN FUNCTION AND CLI PARSING ===
def main():
    parser = argparse.ArgumentParser(description="AUDIOTOOLS.py - Unified Audio/Music Processing Toolkit", usage="python AUDIOTOOLS.py <command> [<args>]")
    parser.add_argument("command", help="Subcommand to run (e.g., 'watermark-detect', 'master')")

    # Print usage if no command is provided or if it's an unknown command early
    if len(sys.argv) == 1:
        print_usage()
        sys.exit(1)

    # Check for valid command before parsing further to provide better help
    # This is a simple check; subparsers are the robust way to handle this.
    known_commands = ['watermark-detect', 'remove-vocal-noise', 'slice', 'normalize', 'bpm', 'key', 'midi2wav', 'trapbeat', 'master', 'stem-split']
    if sys.argv[1] not in known_commands:
        print(f"Unknown command: {sys.argv[1]}")
        print_usage()
        sys.exit(1)

    args = parser.parse_args(sys.argv[1:2]) # Parse only the command first

    # Input file existence check utility
    def check_input_file_exists(filepath):
        if not Path(filepath).is_file():
            print(f"Error: Input file not found: {filepath}")
            sys.exit(1)

    if args.command == 'watermark-detect':
        parser_wd = argparse.ArgumentParser(description='Detect watermark in MIDI file.')
        parser_wd.add_argument("input_mid", help="Path to the input MIDI file.")
        sub_args = parser_wd.parse_args(sys.argv[2:])
        check_input_file_exists(sub_args.input_mid)
        detect_watermark_patterns(sub_args.input_mid)

    elif args.command == 'remove-vocal-noise':
        parser_rvn = argparse.ArgumentParser(description='Remove noise from vocal audio.')
        parser_rvn.add_argument("input_wav", help="Path to the input WAV file.")
        parser_rvn.add_argument("output_wav", help="Path to the output WAV file.")
        parser_rvn.add_argument("--noise-strength", type=float, default=0.5, help="Strength of noise reduction (0.0-1.0).")
        parser_rvn.add_argument("--silence-db", type=int, default=30, help="Top dB for silence trimming.")
        parser_rvn.add_argument("--deepfilternet", action='store_true', help="Use DeepFilterNet for noise removal.")
        sub_args = parser_rvn.parse_args(sys.argv[2:])
        check_input_file_exists(sub_args.input_wav)
        process_vocal(sub_args.input_wav, sub_args.output_wav, 
                        noise_strength=sub_args.noise_strength, 
                        silence_db=sub_args.silence_db, 
                        use_deepfilternet=sub_args.deepfilternet)

    elif args.command == 'slice':
        parser_slice = argparse.ArgumentParser(description='Slice audio into segments.')
        parser_slice.add_argument("input_wav", help="Path to the input WAV file.")
        parser_slice.add_argument("segment_length", type=float, help="Length of each segment in seconds.")
        parser_slice.add_argument("output_prefix", help="Prefix for output sliced files.")
        sub_args = parser_slice.parse_args(sys.argv[2:])
        check_input_file_exists(sub_args.input_wav)
        audio_slice(sub_args.input_wav, sub_args.segment_length, sub_args.output_prefix)

    elif args.command == 'normalize':
        parser_norm = argparse.ArgumentParser(description='Batch normalize audio files in a folder.')
        parser_norm.add_argument("folder", help="Path to the folder containing audio files.")
        sub_args = parser_norm.parse_args(sys.argv[2:])
        if not Path(sub_args.folder).is_dir():
            print(f"Error: Input folder not found: {sub_args.folder}")
            sys.exit(1)
        batch_normalize(sub_args.folder)

    elif args.command == 'bpm':
        parser_bpm = argparse.ArgumentParser(description='Detect BPM of an audio file.')
        parser_bpm.add_argument("input_wav", help="Path to the input audio file.")
        sub_args = parser_bpm.parse_args(sys.argv[2:])
        check_input_file_exists(sub_args.input_wav)
        detect_bpm(sub_args.input_wav)

    elif args.command == 'key':
        parser_key = argparse.ArgumentParser(description='Detect musical key of an audio file.')
        parser_key.add_argument("input_wav", help="Path to the input audio file.")
        sub_args = parser_key.parse_args(sys.argv[2:])
        check_input_file_exists(sub_args.input_wav)
        detect_key(sub_args.input_wav)

    elif args.command == 'midi2wav':
        parser_m2w = argparse.ArgumentParser(description='Convert MIDI to WAV using a SoundFont.')
        parser_m2w.add_argument("input_mid", help="Path to the input MIDI file.")
        parser_m2w.add_argument("output_wav", help="Path to the output WAV file.")
        parser_m2w.add_argument("soundfont", help="Path to the SoundFont file (.sf2).")
        sub_args = parser_m2w.parse_args(sys.argv[2:])
        check_input_file_exists(sub_args.input_mid)
        check_input_file_exists(sub_args.soundfont)
        if not check_command_exists("fluidsynth"):
            print("Error: fluidsynth command not found. Please install FluidSynth.")
            sys.exit(1)
        midi_to_wav(sub_args.input_mid, sub_args.output_wav, sub_args.soundfont)

    elif args.command == 'trapbeat':
        parser_tb = argparse.ArgumentParser(description='Generate a trap beat MIDI file.')
        parser_tb.add_argument("output_mid", nargs='?', default=FILENAME, help=f"Output MIDI filename (default: {FILENAME})")
        parser_tb.add_argument("--bpm", type=int, default=BPM, help=f"Beats per minute (default: {BPM})")
        parser_tb.add_argument("--key", type=str, default='C', help="Musical key (e.g., C, G#, Eb). Default: C")
        sub_args = parser_tb.parse_args(sys.argv[2:])
        
        key_name = sub_args.key.upper().replace('SHARP', '#').replace('FLAT', 'B')
        if key_name not in KEY_MIDI_OFFSETS:
            print(f"Error: Invalid key '{sub_args.key}'. Supported keys: {list(KEY_MIDI_OFFSETS.keys())}")
            sys.exit(1)
        
        # Default octave for root note (e.g., C4)
        # MIDI note 60 is C4. KEY_MIDI_OFFSETS gives offset from C.
        # So, C -> 0, C# -> 1, etc.  key_root_note = 60 (C4) + offset
        base_midi_c4 = 60 
        key_root_note = base_midi_c4 + KEY_MIDI_OFFSETS[key_name]
        
        generate_full_trap_beat(filename=sub_args.output_mid, bpm=sub_args.bpm, key_root_note=key_root_note)

    elif args.command == 'master':
        parser_master = argparse.ArgumentParser(description='Master an audio file.')
        parser_master.add_argument("input_wav", help="Path to the input audio file.")
        parser_master.add_argument("output_wav", help="Path to the output mastered audio file.")
        parser_master.add_argument("--lufs", type=float, default=-14.0, help="Target LUFS for mastering (default: -14.0).")
        parser_master.add_argument("--true-peak", type=float, default=-1.0, help="Target true peak in dBFS (default: -1.0).")
        parser_master.add_argument("--format", type=str, default="wav", choices=['wav', 'mp3', 'flac', 'ogg'], help="Output format (default: wav).")
        parser_master.add_argument("--no-dither", action='store_true', help="Disable dithering.")
        parser_master.add_argument("--ai-mastering", action='store_true', help="Use AI mastering simulation.")
        sub_args = parser_master.parse_args(sys.argv[2:])
        check_input_file_exists(sub_args.input_wav)
        master_audio(sub_args.input_wav, sub_args.output_wav,
                       target_lufs=sub_args.lufs, true_peak=sub_args.true_peak,
                       output_format=sub_args.format, no_dither=sub_args.no_dither,
                       ai_mastering_flag=sub_args.ai_mastering)

    elif args.command == 'stem-split':
        parser_ss = argparse.ArgumentParser(description='Split audio into stems using Demucs.')
        parser_ss.add_argument("input_wav", help="Path to the input audio file.")
        parser_ss.add_argument("output_dir", help="Directory to save the output stems.")
        parser_ss.add_argument("--stems", type=int, default=4, choices=[2, 4, 6], help="Number of stems to split into (2, 4, or 6. Default: 4).")
        sub_args = parser_ss.parse_args(sys.argv[2:])
        check_input_file_exists(sub_args.input_wav)
        stem_split(sub_args.input_wav, sub_args.output_dir, num_stems=sub_args.stems)

    else:
        print(f"Unknown command: {args.command}")
        print_usage()
        sys.exit(1)

if __name__ == "__main__":
    main()
