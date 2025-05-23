import sys
from pydub import AudioSegment

if len(sys.argv) < 4:
    print("Usage: python audio_slicer.py <input.wav> <segment_length_sec> <output_prefix>")
    sys.exit(1)

input_file = sys.argv[1]
segment_length = float(sys.argv[2])
prefix = sys.argv[3]
audio = AudioSegment.from_file(input_file)
segment_ms = int(segment_length * 1000)
for i, start in enumerate(range(0, len(audio), segment_ms)):
    segment = audio[start:start+segment_ms]
    outname = f"{prefix}_slice_{i+1}.wav"
    segment.export(outname, format="wav")
    print(f"Exported {outname}")
