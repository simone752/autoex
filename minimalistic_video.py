import numpy as np
import random
import wave
import struct
import os
from PIL import Image, ImageDraw
import subprocess

def generate_visuals(frame_count, width=640, height=480):
    os.makedirs("frames", exist_ok=True)
    for i in range(frame_count):
        img = Image.new("RGB", (width, height), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        draw = ImageDraw.Draw(img)
        
        for _ in range(random.randint(1, 5)):
            x0, x1 = sorted([random.randint(0, width) for _ in range(2)])
            y0, y1 = sorted([random.randint(0, height) for _ in range(2)])
            shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw.rectangle([x0, y0, x1, y1], outline=shape_color, width=random.randint(1, 5))
        
        img.save(f"frames/frame_{i:04d}.png")

def generate_audio(audio_length=10, sample_rate=44100):
    audio_data = []
    for i in range(audio_length * sample_rate):
        if random.random() < 0.02:
            sample = random.randint(-32768, 32767)  # Sudden noise bursts
        elif i % (random.randint(5000, 20000)) < 1000:
            sample = int(20000 * np.sin(2 * np.pi * i / (random.randint(400, 1200))))  # Structured pulses
        else:
            sample = 0  # Silence
        audio_data.append(struct.pack("<h", sample))
    
    with wave.open("audio.wav", "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(audio_data))

def create_video(output_path, frame_rate=24):
    subprocess.run(["ffmpeg", "-y", "-framerate", str(frame_rate), "-i", "frames/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "video.mp4"], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", "video.mp4", "-i", "audio.wav", "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_path], check=True)

def main():
    frame_count = 240  # 10 seconds at 24 FPS
    print("Generating extreme visuals...")
    generate_visuals(frame_count)
    print("Generating eerie audio...")
    generate_audio()
    print("Creating video...")
    create_video("extreme_video.mp4")
    print("Video created: extreme_video.mp4")

if __name__ == "__main__":
    main()
