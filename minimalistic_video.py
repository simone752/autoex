import os
import random
import pygame
import numpy as np
import cv2
import ffmpeg
from PIL import Image, ImageDraw, ImageFont

# Ensure pygame runs in headless environments
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Video and audio parameters
WIDTH, HEIGHT = random.choice([(640, 480), (800, 600), (1280, 720)])  # Random resolutions
DURATION = random.uniform(5, 15)  # Random duration between 5-15 sec
FPS = random.choice([15, 24, 30])  # Random frame rates

FRAME_COUNT = int(DURATION * FPS)
COLOR_PALETTE = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5)]
WORDS = ["ERROR", "LOADING", "DATA", "VOID", "NULL", "SYSTEM", "CODE", "####", "????", "EXIT"]
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Audio parameters
SAMPLE_RATE = 44100
NOTE_FREQS = [random.randint(200, 800) for _ in range(8)]  # Random notes
SILENCE_PROB = 0.2  # 20% probability of silence

# Generate Video Frames
def generate_video_frames():
    frames = []
    for i in range(FRAME_COUNT):
        img = Image.new("RGB", (WIDTH, HEIGHT), random.choice(COLOR_PALETTE))
        draw = ImageDraw.Draw(img)

        # Random rectangles
        for _ in range(random.randint(1, 5)):
            x0, y0 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
            x1, y1 = x0 + random.randint(20, 200), y0 + random.randint(20, 200)
            draw.rectangle([x0, y0, x1, y1], outline=random.choice(COLOR_PALETTE), width=random.randint(2, 6))

        # Random text
        if random.random() > 0.5:
            text = random.choice(WORDS)
            draw.text((random.randint(10, WIDTH - 100), random.randint(10, HEIGHT - 50)), text, fill=random.choice(COLOR_PALETTE))

        # Convert to OpenCV format
        frames.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    return frames

# Generate Sound
def generate_audio():
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)
    sound_array = np.zeros(int(DURATION * SAMPLE_RATE), dtype=np.int16)

    for i in range(0, len(sound_array), SAMPLE_RATE // random.choice([2, 4, 8])):
        if random.random() > SILENCE_PROB:
            freq = random.choice(NOTE_FREQS)
            t = np.linspace(0, 1, SAMPLE_RATE // random.choice([2, 4, 8]), endpoint=False)
            wave = 32767 * np.sin(2 * np.pi * freq * t)
insert_length = min(len(wave), len(sound_array) - i)
sound_array[i:i + insert_length] = wave[:insert_length].astype(np.int16)

    pygame.mixer.quit()
    return sound_array

# Save Video
def save_video(frames, audio_data):
    video_filename = "extreme_video.mp4"
    audio_filename = "extreme_audio.wav"

    # Save audio
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)
    pygame.mixer.Sound(audio_data).save(audio_filename)

    # Save video using OpenCV
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), FPS, (WIDTH, HEIGHT))
    for frame in frames:
        out.write(frame)
    out.release()

    # Combine video and audio using FFmpeg
    ffmpeg.input(video_filename).input(audio_filename).output("final_video.mp4", vcodec="libx264", acodec="aac").run(overwrite_output=True)

# Main execution
frames = generate_video_frames()
audio_data = generate_audio()
save_video(frames, audio_data)

print("Video generation complete: final_video.mp4")
