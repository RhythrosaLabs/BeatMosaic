<div align="center">

# 🎵 BeatMosaic

**Turn any image into a playable drum machine and sample pack**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-Audio-blueviolet?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

BeatMosaic converts any image into 16 unique audio samples arranged in an interactive drum machine. Each section of a 4×4 grid maps to a segment of the image, generating sine, square, sawtooth, or triangle waves based on the image's pixel data — color, brightness, and texture all influence the sound.

## ✨ Features

- **Image → Sample Pack** — upload any image and instantly get 16 unique audio samples
- **Interactive 4×4 Grid** — click cells to play samples, like a browser-based MPC
- **Dynamic Wave Synthesis** — sine, square, sawtooth, and triangle waves derived from pixel data
- **Effects & Modulation** — reverb, delay, rhythmic patterns applied per-cell
- **Record & Export** — capture your session and save WAV files locally
- **Streamlit UI** — clean, browser-based interface with live image preview

## 🚀 Quick Start

```bash
git clone https://github.com/RhythrosaLabs/BeatMosaic.git
cd BeatMosaic
pip install -r requirements.txt
streamlit run main.py
```

Then open your browser to `http://localhost:8501`, upload an image, and start jamming.

## 🛠️ Tech Stack

- **Python** — core logic
- **Streamlit** — web UI
- **Librosa** — audio analysis and processing
- **NumPy** — pixel-to-waveform math
- **Pillow** — image loading and segmentation
- **SoundFile** — WAV export

## 🎨 How It Works

1. Upload an image → BeatMosaic splits it into a 4×4 grid (16 cells)
2. Each cell's average color, brightness, and texture are extracted
3. Those values map to waveform type, frequency, amplitude, and effects
4. Click any cell to play its generated sample
5. Record a sequence and export as a WAV file

## 📸 Demo

Each image produces a completely different sonic palette. Try abstract art for ambient textures, or high-contrast photos for punchy drum hits.

## 🤝 Contributing

PRs welcome! Open an issue first for major changes.

## 📄 License

MIT

## 💛 Support

If BeatMosaic sparks some creativity, consider supporting development:

👉 [Donate via PayPal](https://paypal.me/noodlebake) — @noodlebake

---
<div align="center">Made with ❤️ by <a href="https://github.com/RhythrosaLabs">RhythrosaLabs</a></div>
