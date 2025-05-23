import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import random

BPM = 140
PPQ = 480
NUM_BARS = 8
BAR_LENGTH = PPQ * 4
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

mid = MidiFile(ticks_per_beat=PPQ)

tracks = []
for i in range(16):
    track = MidiTrack()
    mid.tracks.append(track)
    tracks.append(track)

tempo = mido.bpm2tempo(BPM)
for track in tracks:
    track.append(MetaMessage('set_tempo', tempo=tempo))

def add_note(track, note, velocity, start_tick, duration, channel):
    if len(track) == 0:
        delta = start_tick
    else:
        delta = max(0, start_tick - sum(msg.time for msg in track if hasattr(msg, 'time')))
    track.append(Message('note_on', note=note, velocity=velocity, time=delta, channel=channel))
    track.append(Message('note_off', note=note, velocity=0, time=duration, channel=channel))

def generate_drum_pattern(bar_start):
    kick_pos = [0, int(BAR_LENGTH * 0.75), int(BAR_LENGTH * 3.5)]
    for pos in kick_pos:
        add_note(tracks[9], KICK, 120, bar_start + pos, PPQ // 8, 9)
    snare_pos = [PPQ, PPQ * 3]
    for pos in snare_pos:
        add_note(tracks[9], SNARE, 110, bar_start + pos, PPQ // 6, 9)
        add_note(tracks[9], CLAP, 90, bar_start + pos + PPQ // 16, PPQ // 8, 9)
    sixteenth = BAR_LENGTH // 16
    tick = bar_start
    for i in range(16):
        vel = 75 + (15 if i % 4 == 0 else 0)
        add_note(tracks[9], HIHAT_CLOSED, vel, tick, sixteenth // 2, 9)
        tick += sixteenth

def generate_toms_crash_ride(bar_start):
    if (bar_start // BAR_LENGTH + 1) % 4 == 0:
        tom_notes = [TOM_LOW, TOM_MID, TOM_HIGH]
        step = BAR_LENGTH // (len(tom_notes) * 2)
        tick = bar_start
        for note in tom_notes:
            add_note(tracks[10], note, 90, tick, step, 10)
            tick += step
            add_note(tracks[10], note, 90, tick, step, 10)
            tick += step
    if (bar_start // BAR_LENGTH + 1) in [2, 6]:
        ride_tick = bar_start
        sixteenth = BAR_LENGTH // 16
        for _ in range(16):
            add_note(tracks[10], RIDE, 60, ride_tick, sixteenth // 2, 10)
            ride_tick += sixteenth
    if (bar_start // BAR_LENGTH + 1) in [1, 5]:
        add_note(tracks[10], CRASH, 100, bar_start, PPQ, 10)

def generate_808_bass_variations(bar_start):
    patterns = [
        [0, 7, 8, 7],
        [0, 5, 7, 10],
    ]
    quarter = PPQ
    for i, channel in enumerate([3,4]):
        pattern = patterns[i]
        for beat in range(4):
            note = BASE_808 + pattern[beat]
            velocity = 120 - beat * 5
            add_note(tracks[channel], note, velocity, bar_start + beat * quarter, quarter, channel)

def generate_leads_and_melodies(bar_start):
    phrases = [
        [0, 2, 3, 5, 3, 2],
        [5, 7, 8, 7, 5, 3],
        [3, 5, 6, 5, 3, 2],
        [0, 3, 5, 7, 8, 10],
    ]
    note_length = BAR_LENGTH // 6
    for i, channel in enumerate([1, 2, 5, 6]):
        phrase = phrases[i]
        time_cursor = bar_start
        for interval in phrase:
            note_index = interval if interval < len(MINOR_NATURAL) else len(MINOR_NATURAL) - 1
            note = MINOR_NATURAL[note_index] + 12
            velocity = 70 + random.randint(-5, 10)
            add_note(tracks[channel], note, velocity, time_cursor, note_length, channel)
            time_cursor += note_length

def generate_pads_and_fx(bar_start):
    pad_notes = [MINOR_NATURAL[0] + 24, MINOR_NATURAL[3] + 24]
    pad_duration = BAR_LENGTH * 2
    if (bar_start // BAR_LENGTH) % 4 == 0:
        for channel in [7,8]:
            for note in pad_notes:
                add_note(tracks[channel], note, 50, bar_start, pad_duration, channel)
    if random.random() < 0.3:
        for channel in [11,12]:
            note = random.choice(MINOR_NATURAL) + 36
            add_note(tracks[channel], note, 40, bar_start + random.randint(0, BAR_LENGTH // 2), PPQ//2, channel)

def generate_vocal_chop_placeholders(bar_start):
    chop_notes = [60, 62, 64, 65, 67]
    quarter = PPQ
    for channel in [13,14,15]:
        for beat in range(4):
            note = random.choice(chop_notes)
            velocity = 80 + random.randint(-10, 10)
            add_note(tracks[channel], note, velocity, bar_start + beat * quarter, quarter // 2, channel)

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

for i, track in enumerate(tracks):
    # Set program/instrument for each channel
    if i == 9:
        # Channel 10: Drums (program ignored)
        pass
    elif i in [3, 4]:
        track.append(Message('program_change', program=38, time=0, channel=i))  # Synth Bass 1
    elif i in [1, 2, 5, 6]:
        track.append(Message('program_change', program=81, time=0, channel=i))  # Lead 1 (square)
    elif i in [7, 8]:
        track.append(Message('program_change', program=89, time=0, channel=i))  # Pad 2 (warm)
    elif i in [11, 12]:
        track.append(Message('program_change', program=95, time=0, channel=i))  # FX 1 (rain)
    elif i in [13, 14, 15]:
        track.append(Message('program_change', program=54, time=0, channel=i))  # Voice Oohs
    else:
        track.append(Message('program_change', program=0, time=0, channel=i))  # Default to Acoustic Grand Piano

for bar in range(NUM_BARS):
    bar_start = bar * BAR_LENGTH
    generate_drum_pattern(bar_start)
    generate_toms_crash_ride(bar_start)
    generate_808_bass_variations(bar_start)
    generate_leads_and_melodies(bar_start)
    generate_pads_and_fx(bar_start)
    generate_vocal_chop_placeholders(bar_start)

add_inaudible_signature(mid, signature='TS')
mid.save(FILENAME)
print(f"🔥 Max-channel trap gold with watermark saved: '{FILENAME}'")
