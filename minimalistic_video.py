import numpy as np
import cv2
import pygame
import random
import wave
import os

# Video settings
WIDTH, HEIGHT = 640, 480
DURATION = random.randint(10, 20)  # 10-20 seconds per video
FPS = random.choice([5, 8, 10])  # Slow, uneasy pacing
FRAME_COUNT = DURATION * FPS
OUTPUT_FILE = "extreme_video.mp4"

# Thematic Elements
THEMES = {
    "ERROR": {
        "colors": [(0, 0, 0), (255, 0, 0), (0, 255, 255)],
        "words": ["MALFUNCTION", "DATA CORRUPT", "???", "01011011"],
        "sounds": ["static", "glitch", "electronic"]
    },
    "FLESH": {
        "colors": [(150, 0, 0), (255, 150, 150), (80, 30, 30)],
        "words": ["MORPH", "INSIDE", "MEAT", "HUNGER"],
        "sounds": ["wet", "heartbeat", "distant voice"]
    },
    "LOST": {
        "colors": [(0, 0, 50), (10, 10, 100), (5, 5, 150)],
        "words": ["NO EXIT", "WHERE AM I?", "VOID", "HELLO?"],
        "sounds": ["echo", "radio static", "whisper"]
    },
    "DECAY": {
        "colors": [(40, 20, 10), (80, 40, 20), (160, 80, 40)],
        "words": ["ROT", "DISSOLVE", "TIME?", "FORGOTTEN"],
        "sounds": ["low hum", "distant scraping", "clock ticking"]
    }
}

# Pick a random theme for this video
SELECTED_THEME = random.choice(list(THEMES.keys()))
THEME_DATA = THEMES[SELECTED_THEME]

# Initialize pygame mixer
os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.mixer.init(frequency=44100, size=-16, channels=1)

def generate_frames():
    print(f"Generating {DURATION}-second nightmare video with theme: {SELECTED_THEME}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(FRAME_COUNT):
        bg_color = random.choice(THEME_DATA["colors"])
        frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)

        # Distortion effects
        if random.random() > 0.5:
            noise = np.random.randint(0, 100, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

        # Flashing symbols & cryptic words
        if random.random() > 0.3:
            text = random.choice(THEME_DATA["words"])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = random.uniform(0.8, 2.5)
            position = (random.randint(50, WIDTH-150), random.randint(50, HEIGHT-50))
            text_color = (random.randint(100, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.putText(frame, text, position, font, font_size, text_color, 2, cv2.LINE_AA)

        # Occasional screen tear effect
        if random.random() > 0.7:
            y_pos = random.randint(0, HEIGHT)
            frame[y_pos:y_pos + 5, :] = frame[y_pos + 10:y_pos + 15, :]

        # Rare hidden message (only appears for 1 frame)
        if i == random.randint(1, FRAME_COUNT - 2):
            cv2.putText(frame, "HELP ME", (WIDTH//3, HEIGHT//2), font, 2, (255, 255, 255), 3, cv2.LINE_AA)

        video.write(frame)

    video.release()
    print("Video generation complete.")

def generate_audio():
    SAMPLE_RATE = 44100
    samples = np.zeros(SAMPLE_RATE * DURATION, dtype=np.int16)

    # Choose a main sound effect
    sound_style = random.choice(THEME_DATA["sounds"])

    for i in range(DURATION):
        freq = random.choice([220, 330, 440, 666, 880])  # Eerie frequencies
        volume = random.randint(5000, 20000)

        wave_data = (volume * np.sin(2 * np.pi * np.arange(SAMPLE_RATE) * freq / SAMPLE_RATE)).astype(np.int16)
        start, end = i * SAMPLE_RATE, (i + 1) * SAMPLE_RATE
        samples[start:end] = wave_data[:SAMPLE_RATE]

        # Add different distortions based on the sound style
        if sound_style == "glitch" and random.random() > 0.5:
            glitch_start = random.randint(start, end - SAMPLE_RATE // 10)
            samples[glitch_start:glitch_start + SAMPLE_RATE // 10] = np.random.randint(-20000, 20000, SAMPLE_RATE // 10)

        elif sound_style == "heartbeat" and random.random() > 0.5:
            beat_start = random.randint(start, end - SAMPLE_RATE // 5)
            samples[beat_start:beat_start + SAMPLE_RATE // 20] = samples[beat_start:beat_start + SAMPLE_RATE // 20] // 2

        elif sound_style == "whisper" and random.random() > 0.5:
            whisper_start = random.randint(0, SAMPLE_RATE * (DURATION // 2))
            samples[whisper_start:whisper_start + SAMPLE_RATE // 4] = np.random.randint(-5000, 5000, SAMPLE_RATE // 4)

    # Save audio
    with wave.open("audio.wav", "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    print("Audio generation complete.")

def combine_video_audio():
    os.system(f"ffmpeg -y -i video_temp.mp4 -i audio.wav -c:v libx264 -c:a aac -strict experimental {OUTPUT_FILE}")
    os.remove("video_temp.mp4")
    os.remove("audio.wav")
    print(f"Final nightmare video ({SELECTED_THEME}) is ready.")

if __name__ == "__main__":
    generate_frames()
    generate_audio()
    combine_video_audio()
