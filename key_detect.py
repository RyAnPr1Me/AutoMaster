import sys
import librosa
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python key_detect.py <audiofile>")
    sys.exit(1)

filename = sys.argv[1]
y, sr = librosa.load(filename)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
chroma_mean = np.mean(chroma, axis=1)
keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
key_index = np.argmax(chroma_mean)
print(f"Estimated key: {keys[key_index]}")
