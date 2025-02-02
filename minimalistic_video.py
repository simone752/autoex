import numpy as np
import random
import os
import ffmpeg  # You are importing ffmpeg, but not using it directly. Consider removing if not needed.
from PIL import Image, ImageDraw

# Video settings
WIDTH, HEIGHT = random.choice([(640, 480), (1280, 720), (1920, 1080)])
FRAME_RATE = random.choice([15, 24, 30])
DURATION = random.uniform(5, 15)
FRAME_COUNT = int(FRAME_RATE * DURATION)

# Audio settings
SAMPLE_RATE = 44100
BITS = 16
CHANNELS = 1

def generate_visuals():
    os.makedirs("frames", exist_ok=True)
    for i in range(FRAME_COUNT):
        img = Image.new("RGB", (WIDTH, HEIGHT), random.choice(["black", "white", "gray"]))
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(5, 20)):
            x0, y0 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
            x1, y1 = random.randint(x0, WIDTH), random.randint(y0, HEIGHT)
            draw.rectangle([x0, y0, x1, y1], outline=random.choice(["red", "blue", "yellow", "green"]))
        img.save(f"frames/frame_{i:04d}.png")

def generate_audio():
    audio_length = int(SAMPLE_RATE * DURATION)
    sound_array = np.zeros(audio_length, dtype=np.int16)
    for _ in range(random.randint(5, 15)):
        freq = random.choice([220, 440, 880, 1760])
        duration = random.uniform(0.1, 1.0)
        start = random.randint(0, audio_length - int(SAMPLE_RATE * duration))
        wave = (np.sin(2 * np.pi * np.arange(int(SAMPLE_RATE * duration)) * freq / SAMPLE_RATE) * 32767).astype(np.int16)
        sound_array[start:start + len(wave)] += wave[:min(len(wave), len(sound_array) - start)]

    np.savetxt("audio.raw", sound_array, fmt="%d")

def create_video():
    os.system(f"ffmpeg -y -framerate {FRAME_RATE} -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p video.mp4")
    os.system(f"ffmpeg -y -f s16le -ar {SAMPLE_RATE} -ac {CHANNELS} -i audio.raw -c:a aac -b:a 128k audio.mp4")
    os.system("ffmpeg -y -i video.mp4 -i audio.mp4 -c:v copy -c:a aac extreme_video.mp4")

generate_visuals()
generate_audio()
create_video()
