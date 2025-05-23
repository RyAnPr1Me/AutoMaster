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

# --- FULL TRAP BEAT GENERATOR ---
def generate_full_trap_beat(
    num_bars=NUM_BARS,
    bpm=BPM,
    complexity=0.7,
    random_seed=None,
    watermark_pattern=[60, 62, 64, 65, 67],
    watermark_channels=[1,2,5,6,7,8],
    watermark_velocity=2,
    watermark_duration=PPQ//8,
    signature='TS',
    filename=FILENAME
):
    # Use local versions of mid/tracks so global code doesn't interfere
    mid = MidiFile(ticks_per_beat=PPQ)
    tracks = []
    for i in range(16):
        track = MidiTrack()
        mid.tracks.append(track)
        tracks.append(track)
    tempo = mido.bpm2tempo(bpm)
    for track in tracks:
        track.append(MetaMessage('set_tempo', tempo=tempo))
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
            track.append(Message('program_change', program=54, time=0, channel=i))
        else:
            track.append(Message('program_change', program=0, time=0, channel=i))
    # --- Arrangement: intro, drop, bridge, outro ---
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
            tick = bar_start + step * (BAR_LENGTH // 16) + random.randint(-8, 8)
            # KICK
            if (section != 'intro' and step in [0, 6, 10, 14]) or (section == 'intro' and step == 0) or (random.random() < 0.08 * complexity):
                add_note(tracks[9], KICK, 120 + random.randint(-5, 5), tick, PPQ // 8, 9)
            # SNARE/CLAP
            if step in [4, 12]:
                add_note(tracks[9], SNARE, 110 + random.randint(-10, 10), tick, PPQ // 8, 9)
                if random.random() < 0.5:
                    add_note(tracks[9], CLAP, 90 + random.randint(-10, 10), tick + PPQ // 32, PPQ // 8, 9)
            elif step in [6, 14] and random.random() < 0.2 * complexity:
                add_note(tracks[9], SNARE, 60 + random.randint(-10, 10), tick, PPQ // 16, 9)
            # HIHAT
            if section != 'outro' or step % 4 == 0:
                vel = 70 + int(25 * (1 if step % 4 == 0 else random.random())) + random.randint(-5, 5)
                add_note(tracks[9], HIHAT_CLOSED, vel, tick, PPQ // 16, 9)
                if random.random() < 0.12 * complexity and step % 4 == 2:
                    for roll in range(2):
                        roll_tick = tick + roll * (PPQ // 32) + random.randint(-2, 2)
                        add_note(tracks[9], HIHAT_CLOSED, vel - 10, roll_tick, PPQ // 32, 9)
                if step % 4 == 2 and random.random() < 0.25 * complexity:
                    add_note(tracks[9], HIHAT_OPEN, 80 + random.randint(-10, 10), tick, PPQ // 16, 9)
        # --- 808 Bass ---
        base_pattern = [0, 0, 3, 5, 0, 7, 5, 0]
        for i, channel in enumerate([3, 4]):
            last_note = None
            for step in range(8):
                note = BASE_808 + base_pattern[step] + (random.choice([-2, 0, 2]) if random.random() < 0.15 * complexity else 0)
                velocity = 115 - step * 2 + int(10 * random.random())
                start = bar_start + step * (BAR_LENGTH // 8) + random.randint(-8, 8)
                dur = PPQ // 2
                if random.random() < 0.15 * complexity and last_note is not None:
                    add_note(tracks[channel], last_note, velocity - 10, start, dur // 2, channel)
                add_note(tracks[channel], note, velocity, start, dur, channel)
                last_note = note
        # --- Melodic Leads ---
        for idx, channel in enumerate([1, 2, 5, 6]):
            note = random.choice(MINOR_NATURAL) + 12
            time_cursor = bar_start
            for n in range(6):
                interval = random.choice([-2, -1, 0, 1, 2])
                note_index = MINOR_NATURAL.index(note % 12) if (note % 12) in MINOR_NATURAL else 0
                note = MINOR_NATURAL[(note_index + interval) % len(MINOR_NATURAL)] + 12
                velocity = 75 + random.randint(-10, 15)
                if (bar + n) % 2 == idx % 2:
                    add_note(tracks[channel], note, velocity, time_cursor + random.randint(-10, 10), BAR_LENGTH // 6, channel)
                time_cursor += BAR_LENGTH // 6
        # --- Pads/Atmosphere ---
        if bar % 4 == 0 and section not in ['intro', 'bridge']:
            pad_notes = [MINOR_NATURAL[0] + 24, MINOR_NATURAL[3] + 24]
            pad_duration = BAR_LENGTH * 2
            for channel in [7, 8]:
                for note in pad_notes:
                    add_note(tracks[channel], note, 50 + random.randint(-5, 5), bar_start, pad_duration, channel)
        # --- FX & Breaks ---
        if random.random() < 0.3:
            for channel in [11, 12]:
                note = random.choice(MINOR_NATURAL) + 36
                add_note(tracks[channel], note, 40 + random.randint(-10, 10), bar_start + random.randint(0, BAR_LENGTH // 2), PPQ // 2, channel)
        # --- Vocal Chops ---
        chop_notes = [60, 62, 64, 65, 67]
        quarter = PPQ
        for channel in [13, 14, 15]:
            for beat in range(4):
                if random.random() < 0.5:
                    note = random.choice(chop_notes)
                    velocity = 80 + random.randint(-10, 10)
                    add_note(tracks[channel], note, velocity, bar_start + beat * quarter + random.randint(-8, 8), quarter // 2, channel)
        # --- Toms, Crash, Ride ---
        if (bar + 1) % 4 == 0 and section == 'drop':
            tom_notes = [TOM_LOW, TOM_MID, TOM_HIGH]
            step = BAR_LENGTH // (len(tom_notes) * 2)
            tick = bar_start
            for note in tom_notes:
                add_note(tracks[10], note, 90 + random.randint(-10, 10), tick, step, 10)
                tick += step
                add_note(tracks[10], note, 90 + random.randint(-10, 10), tick, step, 10)
                tick += step
        if (bar + 1) in [2, 6] and section == 'drop':
            ride_tick = bar_start
            sixteenth = BAR_LENGTH // 16
            for _ in range(16):
                add_note(tracks[10], RIDE, 60 + random.randint(-10, 10), ride_tick, sixteenth // 2, 10)
                ride_tick += sixteenth
        if (bar + 1) in [1, 5] and section == 'drop':
            add_note(tracks[10], CRASH, 100 + random.randint(-10, 10), bar_start, PPQ, 10)
    # --- Watermark ---
    embed_watermark_in_music(tracks, pattern=watermark_pattern, channels=watermark_channels, velocity=watermark_velocity, duration=watermark_duration)
    add_inaudible_signature(mid, signature=signature)
    mid.save(filename)
    print(f"🔥 Full trap beat generated and saved: '{filename}' (Full generator, embedded watermark)")

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

def add_watermark_pattern(mid, pattern=[60, 62, 64, 65, 67], channel=15, velocity=1, duration=2, spacing=PPQ//8):
    """Add a short, inaudible note pattern as a watermark on a high channel."""
    track = mid.tracks[channel]
    start_tick = 0
    for note in pattern:
        track.append(Message('note_on', note=note, velocity=velocity, time=start_tick, channel=channel))
        track.append(Message('note_off', note=note, velocity=0, time=duration, channel=channel))
        start_tick = spacing

def embed_watermark_in_music(tracks, pattern=[60, 62, 64, 65, 67], channels=[1,2,5,6,7,8], velocity=2, duration=PPQ//8):
    """Embed the watermark pattern into musical channels at low velocity."""
    import random
    # Choose a random bar and channel for each note in the pattern
    for i, note in enumerate(pattern):
        channel = random.choice(channels)
        # Find a plausible tick (e.g., start of bar or random within first half)
        bar = random.randint(0, NUM_BARS-1)
        bar_start = bar * BAR_LENGTH
        tick = bar_start + random.randint(0, BAR_LENGTH//2)
        add_note(tracks[channel], note, velocity, tick, duration, channel)

# --- USE MODERN GENERATOR ---
# generate_trap_beat_modern(num_bars=NUM_BARS, complexity=0.7, random_seed=42)

# Embed watermark pattern in musical channels (not a separate channel)
# embed_watermark_in_music(tracks, pattern=[60, 62, 64, 65, 67], channels=[1,2,5,6,7,8], velocity=2, duration=PPQ//8)

# add_inaudible_signature(mid, signature='TS')

# mid.save(FILENAME)
# print(f"🔥 Max-channel trap gold with watermark saved: '{FILENAME}' (Modern AI version, embedded watermark)")

# --- ENHANCED AI TRAP BEAT GENERATOR ---
def generate_trap_beat_enhanced(num_bars=NUM_BARS, complexity=0.7, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    for bar in range(num_bars):
        bar_start = bar * BAR_LENGTH
        # --- Arrangement: intro, drop, outro ---
        if bar < 2:
            section = 'intro'
        elif bar > num_bars - 3:
            section = 'outro'
        else:
            section = 'drop'
        # --- Modern Drum Pattern with Hi-hat Rolls & Fills ---
        for step in range(16):
            tick = bar_start + step * (BAR_LENGTH // 16)
            # Humanize timing
            tick += random.randint(-8, 8)
            # KICK: 1, 7, 11, 15, with groove
            if step in [0, 6, 10, 14] or (random.random() < 0.08 * complexity):
                if section != 'intro' or step == 0:
                    add_note(tracks[9], KICK, 120 + random.randint(-5, 5), tick, PPQ // 8, 9)
            # SNARE: 5, 13, with ghost notes
            if step in [4, 12]:
                add_note(tracks[9], SNARE, 110 + random.randint(-10, 10), tick, PPQ // 8, 9)
                if random.random() < 0.5:
                    add_note(tracks[9], CLAP, 90 + random.randint(-10, 10), tick + PPQ // 32, PPQ // 8, 9)
            elif step in [6, 14] and random.random() < 0.2 * complexity:
                add_note(tracks[9], SNARE, 60 + random.randint(-10, 10), tick, PPQ // 16, 9)
            # HIHAT: 16th grid, velocity groove, open hats on offbeats, rolls
            if section != 'outro' or step % 4 == 0:
                vel = 70 + int(25 * (1 if step % 4 == 0 else random.random())) + random.randint(-5, 5)
                add_note(tracks[9], HIHAT_CLOSED, vel, tick, PPQ // 16, 9)
                # Hi-hat rolls
                if random.random() < 0.12 * complexity and step % 4 == 2:
                    for roll in range(2):
                        roll_tick = tick + roll * (PPQ // 32) + random.randint(-2, 2)
                        add_note(tracks[9], HIHAT_CLOSED, vel - 10, roll_tick, PPQ // 32, 9)
                if step % 4 == 2 and random.random() < 0.25 * complexity:
                    add_note(tracks[9], HIHAT_OPEN, 80 + random.randint(-10, 10), tick, PPQ // 16, 9)
        # --- Modern 808 Bass with occasional slides ---
        base_pattern = [0, 0, 3, 5, 0, 7, 5, 0]
        for i, channel in enumerate([3, 4]):
            last_note = None
            for step in range(8):
                note = BASE_808 + base_pattern[step] + (random.choice([-2, 0, 2]) if random.random() < 0.15 * complexity else 0)
                velocity = 115 - step * 2 + int(10 * random.random())
                start = bar_start + step * (BAR_LENGTH // 8) + random.randint(-8, 8)
                dur = PPQ // 2
                # Occasionally overlap for slide
                if random.random() < 0.15 * complexity and last_note is not None:
                    add_note(tracks[channel], last_note, velocity - 10, start, dur // 2, channel)
                add_note(tracks[channel], note, velocity, start, dur, channel)
                last_note = note
        # --- Melodic Variation: Call-and-response, humanization ---
        for idx, channel in enumerate([1, 2, 5, 6]):
            note = random.choice(MINOR_NATURAL) + 12
            time_cursor = bar_start
            for n in range(6):
                interval = random.choice([-2, -1, 0, 1, 2])
                note_index = MINOR_NATURAL.index(note % 12) if (note % 12) in MINOR_NATURAL else 0
                note = MINOR_NATURAL[(note_index + interval) % len(MINOR_NATURAL)] + 12
                velocity = 75 + random.randint(-10, 15)
                # Call-and-response: alternate channels
                if (bar + n) % 2 == idx % 2:
                    add_note(tracks[channel], note, velocity, time_cursor + random.randint(-10, 10), BAR_LENGTH // 6, channel)
                time_cursor += BAR_LENGTH // 6
        # --- Pads/Atmosphere ---
        if bar % 4 == 0 and section != 'intro':
            pad_notes = [MINOR_NATURAL[0] + 24, MINOR_NATURAL[3] + 24]
            pad_duration = BAR_LENGTH * 2
            for channel in [7, 8]:
                for note in pad_notes:
                    add_note(tracks[channel], note, 50 + random.randint(-5, 5), bar_start, pad_duration, channel)
        # --- FX & Breaks ---
        if random.random() < 0.3:
            for channel in [11, 12]:
                note = random.choice(MINOR_NATURAL) + 36
                add_note(tracks[channel], note, 40 + random.randint(-10, 10), bar_start + random.randint(0, BAR_LENGTH // 2), PPQ // 2, channel)
        # --- Vocal Chops ---
        chop_notes = [60, 62, 64, 65, 67]
        quarter = PPQ
        for channel in [13, 14, 15]:
            for beat in range(4):
                if random.random() < 0.5:
                    note = random.choice(chop_notes)
                    velocity = 80 + random.randint(-10, 10)
                    add_note(tracks[channel], note, velocity, bar_start + beat * quarter + random.randint(-8, 8), quarter // 2, channel)
        # --- Toms, Crash, Ride ---
        if (bar + 1) % 4 == 0 and section == 'drop':
            tom_notes = [TOM_LOW, TOM_MID, TOM_HIGH]
            step = BAR_LENGTH // (len(tom_notes) * 2)
            tick = bar_start
            for note in tom_notes:
                add_note(tracks[10], note, 90 + random.randint(-10, 10), tick, step, 10)
                tick += step
                add_note(tracks[10], note, 90 + random.randint(-10, 10), tick, step, 10)
                tick += step
        if (bar + 1) in [2, 6] and section == 'drop':
            ride_tick = bar_start
            sixteenth = BAR_LENGTH // 16
            for _ in range(16):
                add_note(tracks[10], RIDE, 60 + random.randint(-10, 10), ride_tick, sixteenth // 2, 10)
                ride_tick += sixteenth
        if (bar + 1) in [1, 5] and section == 'drop':
            add_note(tracks[10], CRASH, 100 + random.randint(-10, 10), bar_start, PPQ, 10)

# --- USE ENHANCED GENERATOR ---
# generate_trap_beat_enhanced(num_bars=NUM_BARS, complexity=0.7, random_seed=42)

# Embed watermark pattern in musical channels (not a separate channel)
# embed_watermark_in_music(tracks, pattern=[60, 62, 64, 65, 67], channels=[1,2,5,6,7,8], velocity=2, duration=PPQ//8)

# add_inaudible_signature(mid, signature='TS')

# mid.save(FILENAME)
# print(f"🔥 Max-channel trap gold with watermark saved: '{FILENAME}' (Enhanced AI version, embedded watermark)")

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    generate_full_trap_beat(num_bars=NUM_BARS, bpm=BPM, complexity=0.7, random_seed=42)
