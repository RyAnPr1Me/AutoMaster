#!/usr/bin/env python3
import mido

try:
    mid = mido.MidiFile('swerve_kick.mid')
    print('=== MIDI FILE INFO ===')
    print(f'Tracks: {len(mid.tracks)}')
    print(f'Ticks per beat: {mid.ticks_per_beat}')
    print(f'Type: {mid.type}')

    track = mid.tracks[0]
    print(f'\n=== TRACK 0 INFO ===')
    print(f'Messages: {len(track)}')

    print('\n=== FIRST 10 MESSAGES ===')
    for i, msg in enumerate(track[:10]):
        print(f'{i:2d}: {msg}')

    note_events = [msg for msg in track if msg.type in ['note_on', 'note_off']]
    print(f'\n=== NOTE EVENTS SUMMARY ===')
    print(f'Total note events: {len(note_events)}')
    print(f'Note on events: {len([msg for msg in track if msg.type == "note_on"])}')
    print(f'Note off events: {len([msg for msg in track if msg.type == "note_off"])}')
    
    # Check timing
    print(f'\n=== TIMING CHECK ===')
    total_time = 0
    for msg in track:
        total_time += msg.time
    print(f'Total ticks: {total_time}')
    print(f'Total beats: {total_time / mid.ticks_per_beat:.2f}')
    print(f'Total seconds at 140 BPM: {(total_time / mid.ticks_per_beat) * (60/140):.2f}')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
