import sys
import librosa

if len(sys.argv) < 2:
    print("Usage: python bpm_detect.py <audiofile>")
    sys.exit(1)

filename = sys.argv[1]
y, sr = librosa.load(filename)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f"Estimated BPM: {tempo:.2f}")
