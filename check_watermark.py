import mido

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
    mid = mido.MidiFile(midi_path)
    found_any = False
    for track_idx, track in enumerate(mid.tracks):
        channel_notes = {}
        channel_velocities = {}
        channel_times = {}
        abs_time = 0
        for msg in track:
            if msg.type == 'note_on':
                ch = msg.channel
                if ch not in channel_notes:
                    channel_notes[ch] = []
                    channel_velocities[ch] = []
                    channel_times[ch] = []
                channel_notes[ch].append(msg.note)
                channel_velocities[ch].append(msg.velocity)
                abs_time += msg.time
                channel_times[ch].append(abs_time)
        for ch in channel_notes:
            notes = channel_notes[ch]
            velocities = channel_velocities[ch]
            times = channel_times[ch]
            for pattern in patterns:
                for i in range(len(notes) - len(pattern) + 1):
                    match = True
                    mismatches = 0
                    low_vel_count = 0
                    # --- ENHANCED: Accept watermark if at least min_matches notes are low velocity and match pattern order ---
                    for j, pnote in enumerate(pattern):
                        # Accept watermark if note is within tolerance and velocity is low
                        if abs(notes[i + j] - pnote) <= tolerance and velocities[i + j] <= velocity_threshold:
                            low_vel_count += 1
                        elif abs(notes[i + j] - pnote) > tolerance:
                            mismatches += 1
                        if mismatches > (len(pattern) - min_matches):
                            match = False
                            break
                    if match and low_vel_count >= min_matches:
                        print(f"Watermark pattern detected in track {track_idx}, channel {ch}, index {i}.")
                        print(f"Notes: {notes[i:i+len(pattern)]}")
                        print(f"Velocities: {velocities[i:i+len(pattern)]}")
                        print(f"Times: {times[i:i+len(pattern)]}")
                        found_any = True
    # --- ENHANCED: Try to find watermark notes even if not in strict sequence ---
    if not found_any:
        all_low_vel_notes = set()
        all_low_vel_times = []
        for track in mid.tracks:
            abs_time = 0
            for msg in track:
                if msg.type == 'note_on':
                    abs_time += msg.time
                    if msg.velocity <= velocity_threshold:
                        all_low_vel_notes.add(msg.note)
                        all_low_vel_times.append(abs_time)
        for pattern in patterns:
            matches = len(set(pattern) & all_low_vel_notes)
            if matches >= min_matches:
                print(f"Watermark notes present (not in sequence): {set(pattern) & all_low_vel_notes}")
                found_any = True
    # --- ADVANCED: Try more transpositions (octave, fifth, etc) ---
    if not found_any:
        transpositions = [2, 3, 5, 7, 12, -2, -3, -5, -7, -12]
        for t in transpositions:
            for pattern in patterns:
                transposed = [n + t for n in pattern]
                matches = len(set(transposed) & all_low_vel_notes)
                if matches >= min_matches:
                    print(f"Transposed watermark notes present (t={t}): {set(transposed) & all_low_vel_notes}")
                    found_any = True
    # --- ADVANCED: Rhythmic fingerprinting (relative time deltas) ---
    # Only run if we have enough low-velocity notes
    if not found_any and len(all_low_vel_times) >= min_matches:
        deltas = [all_low_vel_times[i+1] - all_low_vel_times[i] for i in range(len(all_low_vel_times)-1)]
        # Check for repeated short/long patterns (e.g. watermark notes close together)
        if any(d < 100 for d in deltas):
            print(f"Possible rhythmic watermark: short time deltas detected in low-velocity notes: {deltas}")
            found_any = True
    # --- LOGGING ---
    if found_any:
        with open('watermark_detection_report.txt', 'a') as f:
            f.write(f"Watermark detected in {midi_path} on {__import__('datetime').datetime.now()}\n")
    return found_any

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_watermark.py <midi_file>")
        sys.exit(1)
    midi_file = sys.argv[1]
    # Try multiple watermark patterns (original, reversed, transposed)
    patterns = [WATERMARK_PATTERN]
    # Add reversed
    patterns.append(list(reversed(WATERMARK_PATTERN)))
    # Add transposed up/down 1 semitone
    patterns.append([n+1 for n in WATERMARK_PATTERN])
    patterns.append([n-1 for n in WATERMARK_PATTERN])
    found = detect_watermark_patterns(midi_file, patterns=patterns, velocity_threshold=5, tolerance=1, min_matches=4)
    if found:
        print("✅ Watermark pattern detected (robust, multi-method check)!")
    else:
        print("❌ Watermark pattern NOT found. Try adjusting tolerance or check for heavy editing/distortion.")
