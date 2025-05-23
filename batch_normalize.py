import sys
import os
from pydub import AudioSegment, effects

if len(sys.argv) < 2:
    print("Usage: python batch_normalize.py <folder>")
    sys.exit(1)

folder = sys.argv[1]
for fname in os.listdir(folder):
    if fname.lower().endswith(('.wav', '.mp3')):
        path = os.path.join(folder, fname)
        audio = AudioSegment.from_file(path)
        normalized = effects.normalize(audio)
        outname = os.path.join(folder, f"normalized_{fname}")
        normalized.export(outname, format="wav")
        print(f"Exported {outname}")
