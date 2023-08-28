import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps, ImageTk, ImageDraw
import numpy as np
import sounddevice as sd
import os
import wave
import queue

#================================================================================
#================================================================================

# preset

SAMPLE_RATE = 44100
DURATION = 0.1
MAX_FREQ = 2000
MIN_FREQ = 100

def save_wav(filename, audio, rate):
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())

def normalize_waveform(wave):
    return (wave / np.max(np.abs(wave)) * 32767).astype(np.int16)

def dynamic_range_compression(wave, threshold=-20.0, ratio=4.0):
    # Convert threshold from dB to linear scale
    threshold_amplitude = 10.0 ** (threshold / 20.0)
    
    # Apply compression
    compressed_wave = np.where(
        wave > threshold_amplitude,
        threshold_amplitude + (wave - threshold_amplitude) / ratio,
        wave
    )
    return compressed_wave

#================================================================================
#================================================================================

# sound generation

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

def generate_arpeggio(base_freq, duration, volume, direction="up"):
    frequencies = [base_freq, base_freq * 1.25, base_freq * 1.5]
    waves = [generate_sine_wave(freq, duration, volume) for freq in frequencies]
    if direction == "down":
        waves = waves[::-1]
    wave = np.concatenate(waves)
    return normalize_waveform(wave)

def generate_chord(root_freq, duration, volume):
    frequencies = [root_freq, root_freq * 1.25, root_freq * 1.5]
    waves = [generate_sine_wave(freq, duration, volume) for freq in frequencies]
    wave = np.sum(waves, axis=0)
    return normalize_waveform(wave)

def generate_hi_hat(duration):
    wave = np.random.uniform(-1, 1, int(SAMPLE_RATE * duration))
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

#================================================================================
#================================================================================

# sound fx

def apply_rhythmic_pattern_corrected(wave, pattern):
    sample_pattern = np.repeat(pattern, len(wave) // len(pattern))
    return wave * sample_pattern

def apply_amplitude_modulation(wave, freq, depth):
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    modulator = 1 + depth * np.sin(2 * np.pi * freq * t)
    return wave * modulator

def adjust_attack_release(wave, attack_time, release_time):
    attack_samples = int(SAMPLE_RATE * attack_time)
    release_samples = int(SAMPLE_RATE * release_time)
    attack_envelope = np.linspace(0, 1, attack_samples)
    sustain_envelope = np.ones(len(wave) - attack_samples - release_samples)
    release_envelope = np.linspace(1, 0, release_samples)
    envelope = np.concatenate([attack_envelope, sustain_envelope, release_envelope])
    return wave * envelope

def apply_reverb(sound, num_reflections=5, decay_factor=0.6):  # Increased reflections and decay

    reverbed_sound = np.copy(sound)
    for i in range(1, num_reflections + 1):
        delayed_sound = np.roll(sound, i * 2000) * (decay_factor ** i)
        reverbed_sound = (reverbed_sound.astype(np.float64) + delayed_sound).astype(np.int64)
    return reverbed_sound

def apply_delay(sound, delay_time=0.03, decay_factor=0.7):
    delayed_sound = np.roll(sound, int(SAMPLE_RATE * delay_time))
    return sound + delayed_sound * decay_factor

def apply_rhythm(sound, pattern=[1, 0, 1, 0, 0]):
    expanded_pattern = []
    for p in pattern:
        expanded_pattern.extend([p] * 4410)  # Repeat each value 4410 times for a slower rhythm
    sample_pattern = np.tile(expanded_pattern, len(sound) // len(expanded_pattern) + 1)
    return sound * sample_pattern[:len(sound)]

#================================================================================
#================================================================================

# main function

def play_image_as_audio_with_buttons(img, filepath):
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_folder = base_name + " Sound Pack"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    section_width = img.width // 4
    section_height = img.height // 4
    
    audio_samples = {}  # dictionary to store audio samples for each grid
    
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
            if i == 0 and j == 0:
                wave = generate_kick(wave_duration)
                output_file = os.path.join(output_folder, "kick.wav")
            elif i == 1 and j == 0:
                wave = generate_snare(wave_duration)
                output_file = os.path.join(output_folder, "snare.wav")
            elif i == 2 and j == 0:
                wave = generate_hi_hat(wave_duration)
                output_file = os.path.join(output_folder, "hi_hat.wav")
            elif i == 3 and j == 0:
                center_y = upper + section_height // 2
                freq = np.interp(center_y, [0, img.height], [MAX_FREQ, MIN_FREQ])
                wave = generate_triangle_wave(freq, wave_duration, brightness)  # Representing the tom sound
                output_file = os.path.join(output_folder, "tom.wav")
            else:
                center_y = upper + section_height // 2
                freq = np.interp(center_y, [0, img.height], [MAX_FREQ, MIN_FREQ])
                wave = wave_func(freq, wave_duration, brightness)
                output_file = os.path.join(output_folder, f"section_{i+1}x{j+1}.wav")

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
            
            save_wav(output_file, wave, SAMPLE_RATE)
            audio_samples[(i, j)] = wave  # Store the waveform in the dictionary

    return audio_samples

#================================================================================
#================================================================================

def select_image_with_buttons():
    filepath = filedialog.askopenfilename()
    if not filepath:
        return
    
    global image  # Making it global for other functions to access
    image = Image.open(filepath)
    
    audio_samples = play_image_as_audio_with_buttons(image, filepath)
    
    for i in range(4):
        for j in range(4):
            section_photo = ImageTk.PhotoImage(image=image.crop((i*image.width//4, j*image.height//4, (i+1)*image.width//4, (j+1)*image.height//4)))
            
            btn = tk.Button(frame, image=section_photo, command=lambda i=i, j=j: play_section_sound(i, j, audio_samples))
            btn.image = section_photo
            btn.grid(row=i, column=j, sticky="nsew")

#================================================================================
#================================================================================

def create_grid_overlay(image):
    # Draw 4x4 grid on the image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for i in range(1, 4):
        draw.line([(width/4)*i, 0, (width/4)*i, height], fill="white")
        draw.line([0, (height/4)*i, width, (height/4)*i], fill="white")
    return image


#================================================================================
#================================================================================

def play_section_sound(i, j, audio_samples):
    # Convert the audio sample to int16 and play the sound of a specific grid section
    sd.play(audio_samples[(i, j)].astype(np.int16))

#================================================================================
#================================================================================

# Global variable for recording queue
recorded_samples = queue.Queue()

def record_callback(outdata, frames, time, status):
    """This function will be called in a separate thread by sounddevice during recording."""
    recorded_samples.put(outdata.copy())

def start_recording():
    global recording_stream
    recording_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.int16, callback=record_callback)
    recording_stream.start()

def stop_recording():
    global recording_stream
    recording_stream.stop()
    recording_stream.close()

def save_recording():
    filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    if not filename:
        return
    
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        
        while not recorded_samples.empty():
            wf.writeframes(recorded_samples.get())

#================================================================================
#================================================================================

# GUI Setup
root = tk.Tk()
root.title("Image to Audio Sample Pack Grid")

# Create a frame for image and buttons
frame = tk.Frame(root)
frame.pack(pady=20)

# Create a 4x4 grid of empty frames by default
labels = [[None for _ in range(4)] for _ in range(4)]
for i in range(4):
    for j in range(4):
        labels[i][j] = tk.Label(frame, relief="solid", borderwidth=1, width=15, height=6)
        labels[i][j].grid(row=i, column=j, padx=5, pady=5)

btn_select = tk.Button(root, text="Select Image", command=select_image_with_buttons)
btn_select.pack(pady=20)

root.mainloop()
