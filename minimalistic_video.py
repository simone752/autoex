import os
import random
import numpy as np
import subprocess
import wave
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
NUM_SEGMENTS = random.randint(5, 15)  # Video duration (5-15 seconds)
OUT_FPS = 24  # Output video frame rate
WIDTH, HEIGHT = random.choice([(640, 480), (800, 600), (1280, 720)])  # Random resolution
SAMPLE_RATE = 44100  # Audio sample rate

# --- Helper: Generate a unique tone/beep for each segment ---
def generate_beep(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    vibrato = 1 + 0.05 * np.sin(2 * np.pi * random.uniform(2, 5) * t)
    beep = 0.5 * np.sin(2 * np.pi * freq * t) * vibrato
    return beep

# --- Visual Generation ---
def generate_frames():
    os.makedirs("frames", exist_ok=True)
    palette = [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for _ in range(10)]

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        font = ImageFont.load_default()
    
    for i in range(NUM_SEGMENTS):
        bg_color = random.choice(palette)
        img = Image.new("RGB", (WIDTH, HEIGHT), bg_color)
        draw = ImageDraw.Draw(img)
        
        margin = random.randint(20, 60)
        shape_color = random.choice(palette)
        draw.rectangle([margin, margin, WIDTH - margin, HEIGHT - margin], outline=shape_color, width=random.randint(3, 8))
        
        if random.random() < 0.5:
            random_text = "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.randint(4, 8)))
            text_bbox = draw.textbbox((0, 0), random_text, font=font)  # Updated method
            text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            pos = (random.randint(0, WIDTH - text_width), random.randint(0, HEIGHT - text_height))
            draw.text(pos, random_text, fill=random.choice(palette), font=font)
        
        img.save(f"frames/frame_{i:03d}.png")

# --- Audio Generation ---
def generate_audio():
    audio_segments = []
    base_freq = random.choice([440, 550, 660])
    for i in range(NUM_SEGMENTS):
        freq = base_freq + i * random.uniform(5, 50)
        beep = generate_beep(freq, 1.0, SAMPLE_RATE)
        audio_segments.append(beep)
    
    audio = np.concatenate(audio_segments)
    audio = audio / np.max(np.abs(audio))
    audio_int16 = np.int16(audio * 32767)
    
    with wave.open("audio.wav", "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())

# --- Video Assembly ---
def create_video():
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "1", "-i", "frames/frame_%03d.png",
        "-r", str(OUT_FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", "temp_video.mp4"
    ], check=True)
    
    subprocess.run([
        "ffmpeg", "-y", "-i", "temp_video.mp4", "-i", "audio.wav",
        "-c:v", "copy", "-c:a", "aac", "extreme_video.mp4"
    ], check=True)

# --- Main Execution ---
def main():
    print(f"Generating a video of {NUM_SEGMENTS} seconds at resolution {WIDTH}x{HEIGHT} and {OUT_FPS} fps.")
    generate_frames()
    print("Frames generated.")
    generate_audio()
    print("Audio generated.")
    create_video()
    print("Video generation complete: extreme_video.mp4")

if __name__ == "__main__":
    main()
