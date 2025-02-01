import numpy as np
import wave
import random
import os
import subprocess
from PIL import Image, ImageDraw

def generate_extreme_frames(num_frames, width=640, height=480):
    print("Generating extreme frames...")
    os.makedirs("frames", exist_ok=True)
    
    for i in range(num_frames):
        img = Image.new("RGB", (width, height), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        draw = ImageDraw.Draw(img)
        
        for _ in range(random.randint(5, 15)):
            x0, x1 = sorted([random.randint(0, width), random.randint(0, width)])
            y0, y1 = sorted([random.randint(0, height), random.randint(0, height)])
            shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            if random.random() < 0.5:
                draw.rectangle([x0, y0, x1, y1], outline=shape_color, width=random.randint(1, 5))
            else:
                draw.ellipse([x0, y0, x1, y1], outline=shape_color, width=random.randint(1, 5))
        
        img.save(f"frames/frame_{i:04d}.png")

def generate_extreme_audio(filename, duration=5, sample_rate=44100):
    print("Generating extreme audio...")
    num_samples = duration * sample_rate
    audio_data = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)
    
    with wave.open(filename, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

def create_video(output_filename, frame_rate=24):
    print("Creating video from frames...")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(frame_rate), "-i", "frames/frame_%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "temp_video.mp4"
    ], check=True)
    
    print("Merging audio and video...")
    subprocess.run([
        "ffmpeg", "-y", "-i", "temp_video.mp4", "-i", "temp_audio.wav", "-c:v", "copy", "-c:a", "aac", output_filename
    ], check=True)

def main():
    num_frames = 120  # 5 seconds at 24 FPS
    audio_duration = 5  # seconds
    output_video = "extreme_video.mp4"
    
    generate_extreme_frames(num_frames)
    generate_extreme_audio("temp_audio.wav", audio_duration)
    create_video(output_video)
    
    print("Video generation complete: " + output_video)

if __name__ == "__main__":
    main()
