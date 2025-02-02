import numpy as np
import pygame
import random
import cv2
import os
import wave
import ffmpeg
from PIL import Image, ImageDraw

# Constants
WIDTH, HEIGHT = random.choice([(640, 480), (800, 600), (1280, 720)])  # Random resolution
DURATION = random.uniform(5, 15)  # Random video duration (5 to 15 sec)
FPS = random.choice([10, 15, 24, 30])  # Random frame rate
FRAME_COUNT = int(DURATION * FPS)
COLOR_PALETTE = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])  # Base color theme

# Generate random visuals
def generate_visuals():
    os.makedirs("frames", exist_ok=True)
    
    for i in range(FRAME_COUNT):
        img = Image.new("RGB", (WIDTH, HEIGHT), "black")
        draw = ImageDraw.Draw(img)
        
        # Generate random rectangles
        for _ in range(random.randint(1, 5)):
            x0, y0, x1, y1 = sorted([random.randint(0, WIDTH) for _ in range(2)]), sorted([random.randint(0, HEIGHT) for _ in range(2)])
            draw.rectangle([x0[0], y0[0], x1[1], y1[1]], fill=random.choice(COLOR_PALETTE))
        
        # Save frame
        img.save(f"frames/frame_{i:03d}.png")

# Generate random audio
def generate_audio():
    sample_rate = 44100
    audio_length = int(DURATION * sample_rate)
    sound_array = np.zeros(audio_length, dtype=np.int16)
    
    for i in range(0, audio_length, sample_rate // random.randint(2, 10)):
        freq = random.choice([220, 440, 880]) if random.random() > 0.5 else 0  # Silence or sound
        if freq > 0:
            t = np.linspace(0, 1, sample_rate, False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            insert_length = min(len(wave), len(sound_array) - i)
            sound_array[i:i + insert_length] = wave[:insert_length]
    
    # Save audio
    with wave.open("audio.wav", "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(sound_array.tobytes())

# Compile video
def create_video():
    os.system(f"ffmpeg -framerate {FPS} -i frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p video.mp4 -y")
    os.system("ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac output_video.mp4 -y")

# Run generation
generate_visuals()
generate_audio()
create_video()

print("Video generation complete: output_video.mp4")
