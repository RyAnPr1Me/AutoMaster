# AUDIOTOOLS.py

**AUDIOTOOLS.py** is a unified, fully standalone command-line toolkit for advanced audio and music processing. It merges and enhances a wide range of professional features for musicians, producers, and audio engineers, including watermark detection, trap beat generation, mastering (classic & AI), batch normalization, audio slicing, BPM/key detection, MIDI-to-WAV, robust vocal noise removal (DeepFilterNet support), and stem splitting (Demucs-based).

---

## Features

- **Watermark Detection**: Detects robust watermark patterns in MIDI files, including transpositions and timing variations.
- **Vocal Noise Removal**: Remove background noise and trim silence from vocal WAV files. Supports classic spectral gating and DeepFilterNet AI denoising (`--deepfilternet`).
- **Audio Slicing**: Slice audio files into equal-length segments.
- **Batch Normalization**: Batch normalize all audio files in a folder for consistent loudness.
- **BPM Detection**: Estimate the tempo (BPM) of audio files.
- **Key Detection**: Estimate the musical key of audio files.
- **MIDI to WAV**: Convert MIDI files to WAV using a SoundFont and FluidSynth.
- **Trap Beat Generator**: Generate robust, musical, and watermarked trap beat MIDI files with advanced humanization and signature embedding.
- **Mastering**: Advanced mastering chain (multiband compression, transient shaping, stereo imaging, LUFS normalization, soft limiting, energy-based EQ, dithering, before/after stats). Supports AI mastering (`--ai-mastering`).
- **Stem Splitter**: Split audio into stems (2, 4, or 6) using Demucs AI.

---

## Installation

1. **Clone the repository** (if not already):
   ```bash
   git clone <repo-url>
   cd AutoMaster
   ```
2. **Install Python dependencies:**
   ```bash
   pip install pydub torchaudio librosa soundfile mido scipy numpy
   # For DeepFilterNet denoising:
   pip install deepfilternet
   # For Demucs stem splitting:
   pip install demucs
   # For MIDI to WAV (FluidSynth):
   sudo apt-get install fluidsynth
   ```
3. **(Optional) Install Demucs CLI for stem splitting:**
   ```bash
   pip install demucs
   ```

---

## Usage

Run the tool from the command line:

```bash
python AUDIOTOOLS.py <command> [arguments] [options]
```

### Commands & Options

- **Watermark Detection**
  ```
  python AUDIOTOOLS.py watermark-detect <input.mid>
  ```
- **Vocal Noise Removal**
  ```
  python AUDIOTOOLS.py remove-vocal-noise <input.wav> <output.wav> [--noise-strength=0.5] [--silence-db=30] [--deepfilternet]
  ```
  - `--deepfilternet`: Use DeepFilterNet AI denoising (requires `deepfilternet`)
- **Audio Slicing**
  ```
  python AUDIOTOOLS.py slice <input.wav> <segment_length_sec> <output_prefix>
  ```
- **Batch Normalize**
  ```
  python AUDIOTOOLS.py normalize <folder>
  ```
- **BPM Detection**
  ```
  python AUDIOTOOLS.py bpm <input.wav>
  ```
- **Key Detection**
  ```
  python AUDIOTOOLS.py key <input.wav>
  ```
- **MIDI to WAV**
  ```
  python AUDIOTOOLS.py midi2wav <input.mid> <output.wav> <soundfont.sf2>
  ```
- **Trap Beat Generator**
  ```
  python AUDIOTOOLS.py trapbeat [output.mid]
  ```
- **Mastering**
  ```
  python AUDIOTOOLS.py master <input.wav> <output.wav> [--lufs=-14.0] [--format=wav] [--no-dither] [--ai-mastering]
  ```
  - `--ai-mastering`: Use AI-based mastering (placeholder, extend as needed)
- **Stem Splitter (Demucs)**
  ```
  python AUDIOTOOLS.py stem-split <input.wav> <output_dir> [--stems=2|4|6]
  ```
  - Requires Demucs CLI (`pip install demucs`)

---

## Examples

- **Remove vocal noise with DeepFilterNet:**
  ```
  python AUDIOTOOLS.py remove-vocal-noise vocals.wav vocals_clean.wav --deepfilternet
  ```
- **Master a track with AI mastering:**
  ```
  python AUDIOTOOLS.py master mix.wav master.wav --ai-mastering
  ```
- **Split a song into 4 stems:**
  ```
  python AUDIOTOOLS.py stem-split song.wav stems_out --stems=4
  ```
- **Generate a trap beat MIDI:**
  ```
  python AUDIOTOOLS.py trapbeat mybeat.mid
  ```

---

## Requirements

- Python 3.7+
- See [Installation](#installation) for required packages
- For MIDI-to-WAV: [FluidSynth](https://www.fluidsynth.org/) and a SoundFont file
- For stem splitting: [Demucs](https://github.com/facebookresearch/demucs)

---

## Advanced Features

- **Watermarking**: Trap beat generator embeds robust watermark patterns and inaudible signatures for copyright protection.
- **AI Mastering**: Easily extend the placeholder to use your own AI mastering API or model.
- **DeepFilterNet**: State-of-the-art neural denoising for vocals.
- **Demucs**: High-quality AI stem separation for music production and remixing.

---

## Troubleshooting

- **Missing dependencies**: The script will print clear error messages if a required package or tool is missing.
- **Demucs not found**: Install with `pip install demucs` and ensure the `demucs` command is in your PATH.
- **FluidSynth not found**: Install via your OS package manager (e.g., `sudo apt-get install fluidsynth`).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Credits

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- [Demucs](https://github.com/facebookresearch/demucs)
- [FluidSynth](https://www.fluidsynth.org/)
- [Librosa](https://librosa.org/)
- [PyDub](https://github.com/jiaaro/pydub)
- [Mido](https://mido.readthedocs.io/)

---

## Contact

For questions, suggestions, or contributions, please open an issue or pull request on the repository.
