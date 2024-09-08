import streamlit as st
import numpy as np
from PIL import Image, ImageStat
import io
import base64
import soundfile as sf
from scipy import signal

# Constants
SAMPLE_RATE = 44100
DURATION = 0.5  # Shortened for more immediate feedback

def analyze_image_section(img_section):
    stat = ImageStat.Stat(img_section)
    brightness = sum(stat.mean) / 3
    contrast = sum(stat.stddev) / 3
    hsv_img = img_section.convert('HSV')
    hsv_stat = ImageStat.Stat(hsv_img)
    return {
        'brightness': brightness / 255,  # Normalize to 0-1
        'contrast': contrast / 255,  # Normalize to 0-1
        'hue': hsv_stat.mean[0] / 255,  # Normalize to 0-1
        'saturation': hsv_stat.mean[1] / 255,  # Normalize to 0-1
        'value': hsv_stat.mean[2] / 255,  # Normalize to 0-1
    }

def generate_audio_sample(img_section):
    features = analyze_image_section(img_section)
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)

    # Base frequency (ensuring it's in audible range)
    base_freq = 110 + (features['hue'] * 440)  # Range: 110Hz to 550Hz

    # Generate harmonics
    harmonics = [1, 0.5, 0.25, 0.125]
    wave = np.zeros_like(t)
    for i, amp in enumerate(harmonics):
        wave += amp * np.sin(2 * np.pi * base_freq * (i + 1) * t)

    # Apply envelope
    envelope = np.exp(-t * (5 + 10 * features['brightness']))  # Brightness affects decay
    wave *= envelope

    # Add some noise based on saturation
    noise = np.random.normal(0, 0.1, len(t)) * features['saturation']
    wave += noise

    # Apply low-pass filter based on contrast
    cutoff = 1000 + (features['contrast'] * 4000)  # Range: 1kHz to 5kHz
    sos = signal.butter(10, cutoff, btype='low', fs=SAMPLE_RATE, output='sos')
    wave = signal.sosfilt(sos, wave)

    # Normalize
    wave = wave / np.max(np.abs(wave))

    return (wave * 32767).astype(np.int16)

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
    sf.write(buffer, audio, SAMPLE_RATE, format='wav')
    audio_b64 = base64.b64encode(buffer.getvalue()).decode()
    return audio_b64

def main():
    st.set_page_config(layout="wide", page_title="Image to Audio Sampler")
    
    st.title("Image to Audio Sampler")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
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
            }}
            
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
            
            window.playPad = playPad;
            window.startRecording = startRecording;
            window.stopRecording = stopRecording;
            window.playRecording = playRecording;
            window.recordNote = recordNote;
            </script>
            """,
            unsafe_allow_html=True
        )
        
        # Display spliced image and create interactive pads
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
