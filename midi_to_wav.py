import sys
import subprocess

# Usage: python midi_to_wav.py input.mid output.wav soundfont.sf2
if len(sys.argv) < 4:
    print("Usage: python midi_to_wav.py <input.mid> <output.wav> <soundfont.sf2>")
    sys.exit(1)

midi_file = sys.argv[1]
wav_file = sys.argv[2]
sf2_file = sys.argv[3]

cmd = [
    "fluidsynth", "-ni", sf2_file, midi_file, "-F", wav_file, "-r", "44100"
]
print(f"Running: {' '.join(cmd)}")
subprocess.run(cmd, check=True)
print(f"Exported {wav_file}")
