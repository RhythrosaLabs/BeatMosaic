import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import json
from scipy.io import wavfile
import librosa
import soundfile as sf

# Constants
SAMPLE_RATE = 44100
DURATION = 0.5  # Increased duration for better sound

# Helper functions
def generate_audio_sample(img_section, duration=DURATION):
    # Convert image section to grayscale
    gray = ImageOps.grayscale(img_section)
    # Normalize pixel values
    pixels = np.array(gray).flatten() / 255.0
    
    # Generate a more complex waveform
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    waveform = np.zeros_like(t)
    
    # Use pixel values to modulate different frequency components
    for i, px in enumerate(pixels[:10]):  # Use first 10 pixels for modulation
        freq = 110 * (i + 1)  # Harmonics of 110 Hz
        waveform += px * np.sin(2 * np.pi * freq * t)
    
    # Apply envelope
    envelope = np.exp(-t * 8)
    waveform *= envelope
    
    # Normalize
    waveform = waveform / np.max(np.abs(waveform))
    
    return (waveform * 32767).astype(np.int16)

def process_image_to_audio(img):
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
    buffer = io.BytesIO()
    wavfile.write(buffer, SAMPLE_RATE, audio)
    audio_b64 = base64.b64encode(buffer.getvalue()).decode()
    return audio_b64

# Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="MPC-Style Sampler")
    
    st.title("MPC-Style Image to Audio Sampler")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Process image to audio
        audio_samples = process_image_to_audio(image)
        
        # Convert audio samples to base64
        audio_data = {k: audio_to_base64(v) for k, v in audio_samples.items()}
        
        # Pass audio data to JavaScript
        st.markdown(
            f"""
            <script>
            const audioData = {json.dumps(audio_data)};
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const pads = {{}}; // Store AudioBuffers for each pad
            
            // Decode audio data for each pad
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
            }}
            
            // Transport variables
            let isRecording = false;
            let startTime;
            let recordedNotes = [];
            
            function startRecording() {{
                isRecording = true;
                startTime = audioContext.currentTime;
                recordedNotes = [];
            }}
            
            function stopRecording() {{
                isRecording = false;
            }}
            
            function playRecording() {{
                recordedNotes.forEach(note => {{
                    setTimeout(() => playPad(note.pad), note.time * 1000);
                }});
            }}
            
            function recordNote(pad) {{
                if (isRecording) {{
                    recordedNotes.push({{
                        pad: pad,
                        time: audioContext.currentTime - startTime
                    }});
                }}
            }}
            
            // Expose functions to Streamlit
            window.playPad = playPad;
            window.startRecording = startRecording;
            window.stopRecording = stopRecording;
            window.playRecording = playRecording;
            window.recordNote = recordNote;
            </script>
            """,
            unsafe_allow_html=True
        )
        
        # Create MPC-style pad layout
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

if __name__ == "__main__":
    main()
