import os
import random
import numpy as np
import subprocess
from PIL import Image, ImageDraw
import wave
import struct

def generate_visuals(frame_count, width=640, height=480, base_color=(50, 50, 200)):
    os.makedirs("frames", exist_ok=True)
    for i in range(frame_count):
        img = Image.new("RGB", (width, height), base_color)
        draw = ImageDraw.Draw(img)
        tone_shift = random.randint(-50, 50)
        color = (max(0, min(255, base_color[0] + tone_shift)),
                 max(0, min(255, base_color[1] + tone_shift)),
                 max(0, min(255, base_color[2] + tone_shift)))
        
        if random.random() > 0.7:
            x0, y0 = random.randint(0, width // 2), random.randint(0, height // 2)
            x1, y1 = x0 + random.randint(10, width // 2), y0 + random.randint(10, height // 2)
            draw.rectangle([x0, y0, x1, y1], outline=color, width=random.randint(1, 5))
        
        if random.random() > 0.8:
            face_x, face_y = random.randint(100, width - 100), random.randint(100, height - 100)
            draw.ellipse([face_x, face_y, face_x + 50, face_y + 50], outline="white", width=3)
            draw.ellipse([face_x + 15, face_y + 15, face_x + 20, face_y + 20], fill="white")
            draw.ellipse([face_x + 30, face_y + 15, face_x + 35, face_y + 20], fill="white")
            draw.line([face_x + 15, face_y + 35, face_x + 35, face_y + 35], fill="white", width=2)

        img.save(f"frames/frame_{i:03d}.png")

def generate_audio(duration=15, sample_rate=44100):
    num_samples = duration * sample_rate
    audio_data = np.zeros(num_samples, dtype=np.int16)
    freq = random.randint(300, 800)
    melody_pattern = [random.randint(200, 900) for _ in range(4)]
    
    for i in range(num_samples):
        if (i // sample_rate) % 3 == 0 and random.random() > 0.5:
            freq = random.choice(melody_pattern)
        if i % (sample_rate // 4) < (sample_rate // 16):
            audio_data[i] = int(32767 * np.sin(2 * np.pi * freq * i / sample_rate))
    
    with wave.open("temp_audio.wav", "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def create_video(output_path, frame_rate=24):
    frame_count = frame_rate * 15
    generate_visuals(frame_count)
    generate_audio()
    
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(frame_rate), "-i", "frames/frame_%03d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "temp_video.mp4"
    ], check=True)
    
    subprocess.run([
        "ffmpeg", "-y", "-i", "temp_video.mp4", "-i", "temp_audio.wav",
        "-c:v", "copy", "-c:a", "aac", output_path
    ], check=True)
    
    print(f"Video saved as {output_path}")

if __name__ == "__main__":
    create_video("extreme_video.mp4")
