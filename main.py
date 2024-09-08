import streamlit as st
import numpy as np
from PIL import Image, ImageStat
import io
import base64
import librosa
import soundfile as sf
from scipy import signal

# Constants
SAMPLE_RATE = 44100
DURATION = 2.0

def analyze_image_section(img_section):
    """Analyze the image section and extract relevant features."""
    stat = ImageStat.Stat(img_section)
    brightness = sum(stat.mean) / 3
    contrast = sum(stat.stddev) / 3
    
    # Convert to HSV for color analysis
    hsv_img = img_section.convert('HSV')
    hsv_stat = ImageStat.Stat(hsv_img)
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'hue': hsv_stat.mean[0],
        'saturation': hsv_stat.mean[1],
        'value': hsv_stat.mean[2]
    }

def generate_drum_sound(features):
    """Generate a drum sound based on image features."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Use brightness for the fundamental frequency
    freq = np.interp(features['brightness'], [0, 255], [50, 200])
    
    # Use contrast for decay
    decay = np.interp(features['contrast'], [0, 255], [5, 20])
    
    # Generate drum sound
    drum = np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)
    
    # Add some noise based on saturation
    noise_amount = features['saturation'] / 255
    noise = np.random.normal(0, 0.1, len(t)) * noise_amount
    
    return (drum + noise) * 0.5

def generate_tonal_sound(features):
    """Generate a tonal sound based on image features."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Map hue to a musical scale (C major scale frequencies)
    scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
    note_index = int(features['hue'] / 255 * len(scale))
    freq = scale[note_index]
    
    # Generate harmonics
    harmonics = [1, 0.5, 0.3, 0.2, 0.1]
    tone = np.zeros_like(t)
    for i, amplitude in enumerate(harmonics, 1):
        tone += amplitude * np.sin(2 * np.pi * freq * i * t)
    
    # Apply envelope
    attack = int(0.01 * SAMPLE_RATE)
    decay = int(0.1 * SAMPLE_RATE)
    sustain_level = 0.7
    release = int(0.3 * SAMPLE_RATE)
    
    envelope = np.ones_like(tone)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)
    envelope[-release:] = np.linspace(sustain_level, 0, release)
    
    return tone * envelope

def apply_effects(sound, features):
    """Apply audio effects based on image features."""
    # Apply low-pass filter based on brightness
    cutoff = np.interp(features['brightness'], [0, 255], [500, 10000])
    sos = signal.butter(10, cutoff, btype='low', fs=SAMPLE_RATE, output='sos')
    sound = signal.sosfilt(sos, sound)
    
    # Apply distortion based on contrast
    distortion_amount = np.interp(features['contrast'], [0, 255], [1, 10])
    sound = np.tanh(sound * distortion_amount) / distortion_amount
    
    return sound

def generate_audio_sample(img_section):
    """Generate an audio sample based on the image section."""
    features = analyze_image_section(img_section)
    
    # Decide between drum and tonal sound based on value
    if features['value'] < 128:
        base_sound = generate_drum_sound(features)
    else:
        base_sound = generate_tonal_sound(features)
    
    # Apply effects
    processed_sound = apply_effects(base_sound, features)
    
    # Normalize
    processed_sound = processed_sound / np.max(np.abs(processed_sound))
    
    return (processed_sound * 32767).astype(np.int16)

def process_image_to_audio(img):
    """Process the entire image into audio samples."""
    audio_samples = {}
    for i in range(4):
        for j in range(4):
            left = j * (img.width // 4)
            upper = i * (img.height // 4)
            right = left + (img.width // 4)
            lower = upper + (img.height // 4)
            section = img.crop((left, upper, right, lower))
            audio_samples[f"{i},{j}"] = generate_audio_sample(section)
    return audio_samples

def audio_to_base64(audio):
    """Convert audio array to base64 string."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, SAMPLE_RATE, format='wav')
    audio_b64 = base64.b64encode(buffer.getvalue()).decode()
    return audio_b64

def main():
    st.set_page_config(layout="wide", page_title="Image to Audio Sampler")
    
    st.title("Advanced Image to Audio Sampler")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        audio_samples = process_image_to_audio(image)
        audio_data = {k: audio_to_base64(v) for k, v in audio_samples.items()}
        
        st.markdown(
            f"""
            <script>
            const audioData = {audio_data};
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const pads = {{}};
            
            Object.entries(audioData).forEach(([key, base64Data]) => {{
                const binaryData = atob(base64Data);
                const arrayBuffer = new ArrayBuffer(binaryData.length);
                const uint8Array = new Uint8Array(arrayBuffer);
                for (let i = 0; i < binaryData.length; i++) {{
                    uint8Array[i] = binaryData.charCodeAt(i);
                }}
                audioContext.decodeAudioData(arrayBuffer).then(buffer => {{
                    pads[key] = buffer;
                }});
            }});
            
            function playPad(key) {{
                const source = audioContext.createBufferSource();
                source.buffer = pads[key];
                source.connect(audioContext.destination);
                source.start();
                console.log('Playing pad:', key);  // Debugging
            }}
            
            // Transport variables
            let isRecording = false;
            let startTime;
            let recordedNotes = [];
            
            function startRecording() {{
                isRecording = true;
                startTime = audioContext.currentTime;
                recordedNotes = [];
                console.log('Recording started');  // Debugging
            }}
            
            function stopRecording() {{
                isRecording = false;
                console.log('Recording stopped');  // Debugging
            }}
            
            function playRecording() {{
                console.log('Playing recording', recordedNotes);  // Debugging
                recordedNotes.forEach(note => {{
                    setTimeout(() => playPad(note.pad), note.time * 1000);
                }});
            }}
            
            function recordNote(pad) {{
                if (isRecording) {{
                    const note = {{
                        pad: pad,
                        time: audioContext.currentTime - startTime
                    }};
                    recordedNotes.push(note);
                    console.log('Recorded note:', note);  // Debugging
                }}
            }}
            
            window.playPad = playPad;
            window.startRecording = startRecording;
            window.stopRecording = stopRecording;
            window.playRecording = playRecording;
            window.recordNote = recordNote;
            </script>
            """,
            unsafe_allow_html=True
        )
        
        # MPC-style pad layout
        for i in range(4):
            cols = st.columns(4)
            for j in range(4):
                with cols[j]:
                    pad_image = image.crop((j*image.width//4, i*image.height//4, (j+1)*image.width//4, (i+1)*image.height//4))
                    buffered = io.BytesIO()
                    pad_image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    st.markdown(f"""
                        <div style="width:100%;padding-bottom:100%;position:relative;overflow:hidden;border-radius:10px;margin-bottom:10px;">
                            <img src="data:image/png;base64,{img_str}" style="position:absolute;top:0;left:0;width:100%;height:100%;object-fit:cover;">
                            <button onclick="playPad('{i},{j}'); recordNote('{i},{j}')" style="position:absolute;top:0;left:0;width:100%;height:100%;background:transparent;border:none;cursor:pointer;"></button>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Transport controls
        st.markdown("""
            <div style="display:flex;justify-content:center;margin-top:20px;">
                <button onclick="startRecording()" style="margin:0 10px;padding:10px 20px;">Record</button>
                <button onclick="stopRecording()" style="margin:0 10px;padding:10px 20px;">Stop</button>
                <button onclick="playRecording()" style="margin:0 10px;padding:10px 20px;">Play</button>
            </div>
        """, unsafe_allow_html=True)
        
        # Debugging output
        st.markdown("""
            <div id="debug" style="margin-top:20px;padding:10px;background-color:#f0f0f0;border-radius:5px;">
                <h3>Debug Output:</h3>
                <pre id="debugText"></pre>
            </div>
            <script>
                console.log = function(...args) {
                    document.getElementById('debugText').innerHTML += args.join(' ') + '\\n';
                }
            </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
