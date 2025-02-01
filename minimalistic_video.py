import os
import random
import numpy as np
from PIL import Image, ImageDraw
import wave
import struct
import subprocess

def generate_visuals(frame_count, width=640, height=480):
    os.makedirs("frames", exist_ok=True)
    for i in range(frame_count):
        img = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(img)
        
        if random.random() > 0.5:
            x0, y0 = random.randint(0, width // 2), random.randint(0, height // 2)
            x1, y1 = random.randint(width // 2, width), random.randint(height // 2, height)
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw.rectangle([x0, y0, x1, y1], fill=color)
        
        if random.random() > 0.7:
            face_x, face_y = random.randint(100, 540), random.randint(100, 380)
            draw.ellipse([face_x, face_y, face_x+50, face_y+50], outline="white", width=2)
            draw.ellipse([face_x+15, face_y+15, face_x+25, face_y+25], fill="white")
            draw.ellipse([face_x+30, face_y+15, face_x+40, face_y+25], fill="white")
            draw.line([face_x+20, face_y+35, face_x+35, face_y+35], fill="white", width=2)

        img.save(f"frames/frame_{i:03d}.png")

def generate_audio(filename, duration=5, sample_rate=44100):
    num_samples = duration * sample_rate
    audio_data = []
    for i in range(num_samples):
        value = random.choice([0, 32767, -32768]) if random.random() > 0.8 else int(32767 * np.sin(2 * np.pi * random.randint(100, 2000) * i / sample_rate))
        audio_data.append(value)
        if random.random() > 0.98:
            audio_data.extend([0] * random.randint(1000, 5000))
    
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack("<" + "h" * len(audio_data), *audio_data))

def create_video(output_path, frame_rate=5):
    frame_count = 30
    generate_visuals(frame_count)
    generate_audio("temp_audio.wav", duration=frame_count // frame_rate)
    
    subprocess.run(["ffmpeg", "-framerate", str(frame_rate), "-i", "frames/frame_%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "temp_video.mp4"], check=True)
    subprocess.run(["ffmpeg", "-i", "temp_video.mp4", "-i", "temp_audio.wav", "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_path], check=True)

def main():
    create_video("extreme_video.mp4")
    print("Video generation complete.")

if __name__ == "__main__":
    main()
