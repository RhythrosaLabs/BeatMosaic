import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import wave
import io
import base64
import zipfile

# Constants
SAMPLE_RATE = 44100
DURATION = 0.1
MAX_FREQ = 2000
MIN_FREQ = 100

# Helper functions
def save_wav(audio, rate):
    buffer = io.BytesIO()
    with wave.open(buffer, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())
    buffer.seek(0)
    return buffer

def normalize_waveform(wave):
    return (wave / np.max(np.abs(wave)) * 32767).astype(np.int16)

# Sound generation functions (unchanged)
def generate_sine_wave(freq, duration, volume):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = (volume * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    return normalize_waveform(wave)

def generate_square_wave(freq, duration, volume):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    wave = (volume * np.sign(signal) * 32767).astype(np.int16)
    return normalize_waveform(wave)

def generate_sawtooth_wave(freq, duration, volume):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    signal = 2.0 * (t * freq - np.floor(t * freq + 0.5))
    wave = (volume * signal * 32767).astype(np.int16)
    return normalize_waveform(wave)

def generate_triangle_wave(freq, duration, volume):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    signal = 2.0 * np.abs(2.0 * (t * freq - np.floor(t * freq + 0.5))) - 1.0
    wave = (volume * signal * 32767).astype(np.int16)
    return normalize_waveform(wave)

def generate_kick(duration):
    wave = generate_sine_wave(60, duration, 1.0)
    envelope = np.linspace(1, 0, int(SAMPLE_RATE * duration))
    wave = wave * envelope
    return normalize_waveform(wave)

def generate_snare(duration):
    tone = generate_sine_wave(200, duration * 0.1, 0.5)
    noise = np.random.uniform(-1, 1, int(SAMPLE_RATE * duration))
    noise_start = len(tone)
    noise = noise[:len(noise) - noise_start]
    wave = np.concatenate([tone, noise])
    return normalize_waveform(wave)

def generate_hi_hat(duration):
    wave = np.random.uniform(-1, 1, int(SAMPLE_RATE * duration))
    return normalize_waveform(wave)

# Sound effects functions (unchanged)
def apply_rhythm(sound, pattern=[1, 0, 1, 0, 0]):
    expanded_pattern = []
    for p in pattern:
        expanded_pattern.extend([p] * 4410)  # Repeat each value 4410 times for a slower rhythm
    sample_pattern = np.tile(expanded_pattern, len(sound) // len(expanded_pattern) + 1)
    return sound * sample_pattern[:len(sound)]

def apply_reverb(sound, num_reflections=5, decay_factor=0.6):
    reverbed_sound = np.copy(sound)
    for i in range(1, num_reflections + 1):
        delayed_sound = np.roll(sound, i * 2000) * (decay_factor ** i)
        reverbed_sound = (reverbed_sound.astype(np.float64) + delayed_sound).astype(np.int64)
    return reverbed_sound

def apply_delay(sound, delay_time=0.03, decay_factor=0.7):
    delayed_sound = np.roll(sound, int(SAMPLE_RATE * delay_time))
    return sound + delayed_sound * decay_factor

# Main function to process image and generate audio
def process_image_to_audio(img):
    audio_samples = {}
    section_width = img.width // 4
    section_height = img.height // 4
    
    for i in range(4):
        for j in range(4):
            left = i * section_width
            upper = j * section_height
            right = left + section_width
            lower = upper + section_height
            section = img.crop((left, upper, right, lower))
            
            section_rgb = np.array(section)
            r = np.mean(section_rgb[:,:,0])
            g = np.mean(section_rgb[:,:,1])
            b = np.mean(section_rgb[:,:,2])
            
            grayscale_section = ImageOps.grayscale(section)
            brightness = np.mean(np.array(grayscale_section)) / 255.0
            contrast = np.std(np.array(grayscale_section)) / 255.0
            
            hsv_section = section.convert('HSV')
            saturation = np.mean(np.array(hsv_section)[:,:,1]) / 255.0
            
            # Determine waveform based on multiple factors
            if brightness < 0.33 and contrast < 0.5:
                wave_func = generate_sine_wave
            elif brightness < 0.66 or saturation > 0.5:
                wave_func = generate_square_wave
            elif contrast > 0.7:
                wave_func = generate_sawtooth_wave
            else:
                wave_func = generate_triangle_wave

            # Duration based on RGB
            duration_modifier = ((r + g + b) / 3) / 255.0
            wave_duration = DURATION * duration_modifier

            # Assign designated drum sounds and name accordingly
            center_y = upper + section_height // 2
            freq = np.interp(center_y, [0, img.height], [MAX_FREQ, MIN_FREQ])
            
            if i == 0 and j == 0:
                wave = generate_kick(wave_duration)
            elif i == 1 and j == 0:
                wave = generate_snare(wave_duration)
            elif i == 2 and j == 0:
                wave = generate_hi_hat(wave_duration)
            elif i == 3 and j == 0:
                wave = generate_triangle_wave(freq, wave_duration, brightness)  # Representing the tom sound
            else:
                wave = wave_func(freq, wave_duration, brightness)

            # Rhythmic Variation
            rhythm_pattern = [int(r > 128), int(g > 128), int(b > 128)]
            wave = apply_rhythm(wave, rhythm_pattern)

            # Apply Effects
            if saturation < 0.33:
                wave = apply_reverb(wave)
            elif saturation < 0.66:
                wave = apply_delay(wave)
            else:
                wave = apply_reverb(wave)
                wave = apply_delay(wave)
            
            audio_samples[(i, j)] = wave

    return audio_samples

# Streamlit app
def main():
    st.title("Image to Audio Sample Pack Grid")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        audio_samples = process_image_to_audio(image)

        # Display the 4x4 grid of buttons
        for i in range(4):
            cols = st.columns(4)
            for j in range(4):
                with cols[j]:
                    if st.button(f"Download {i+1}x{j+1}"):
                        wav_buffer = save_wav(audio_samples[(i, j)], SAMPLE_RATE)
                        st.download_button(
                            label=f"Download {i+1}x{j+1}.wav",
                            data=wav_buffer,
                            file_name=f"sample_{i+1}x{j+1}.wav",
                            mime="audio/wav"
                        )

        if st.button("Generate and Download Sample Pack"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                for (i, j), wave in audio_samples.items():
                    wav_buffer = save_wav(wave, SAMPLE_RATE)
                    zip_file.writestr(f"sample_{i+1}x{j+1}.wav", wav_buffer.getvalue())
            
            zip_buffer.seek(0)
            st.download_button(
                label="Download Sample Pack",
                data=zip_buffer,
                file_name="sample_pack.zip",
                mime="application/zip"
            )

if __name__ == "__main__":
    main()
