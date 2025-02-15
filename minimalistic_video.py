import numpy as np
import cv2
import pygame
import random
import string
import wave
import os

# Video settings
WIDTH, HEIGHT = 640, 480
DURATION = random.randint(10, 20)
FPS = random.choice([6, 10, 12])
FRAME_COUNT = DURATION * FPS
OUTPUT_FILE = "extreme_video.mp4"

# Color schemes for an eerie effect
COLOR_SCHEMES = [
    [(10, 10, 10), (255, 0, 0), (0, 0, 255)],
    [(50, 50, 50), (0, 255, 0), (255, 255, 0)],
    [(30, 20, 40), (180, 0, 255), (0, 255, 255)],
    [(0, 0, 0), (255, 255, 255), (128, 0, 128)],
]

# Expanding cryptic phrases and symbols
WORDS = [
    "ERROR", "NO SIGNAL", "HEROHIIHODISTETHTMNGASGG", "IT'S WATCHING", "MISSING", "HELP ME",
    "WHO ARE YOU", "UNKNOWN CODE", "INITIATING SEQUENCE", "HEROHIIHODISTETHTMNGASGGHEROHIIHODISTETHTMNGASGG"
]
SYMBOLS = ["∆", "Ω", "∑", "∂", "⊗", "Ξ", "☠", "✖", "ψ", "λ", "#@$!", "011001"]
NOISES = ["whistle", "shout", "bang", "glitch", "metal scrape", "heartbeat"]

# Initialize pygame mixer for eerie sound effects
os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.mixer.init(frequency=44100, size=-16, channels=1)

def generate_frames():
    """Generates eerie, unsettling video frames with glitches, distortions, and shapes."""
    print(f"Generating {DURATION}-second nightmare video...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))
    
    for i in range(FRAME_COUNT):
        bg_color = random.choice(random.choice(COLOR_SCHEMES))
        frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)

        # Distorted noise overlay
        if random.random() > 0.5:
            noise = np.random.randint(0, 100, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

        # Random glitch effect
        if random.random() > 0.6:
            y_pos = random.randint(0, HEIGHT - 15)
            frame[y_pos:y_pos + 5, :] = frame[y_pos + 10:y_pos + 15, :]

        # Add eerie shapes
        if random.random() > 0.5:
            shape_type = random.choice(["circle", "rectangle"])
            color = (random.randint(100, 255), random.randint(0, 255), random.randint(0, 255))
            if shape_type == "circle":
                cv2.circle(frame, (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)), random.randint(20, 100), color, -1)
            else:
                cv2.rectangle(frame, (random.randint(20, WIDTH - 100), random.randint(20, HEIGHT - 100)), (random.randint(100, WIDTH - 20), random.randint(100, HEIGHT - 20)), color, -1)
        
        # Cryptic text overlay
        text = random.choice(WORDS) + " " + random.choice(SYMBOLS)
        if random.random() > 0.3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = random.uniform(0.8, 2.0)
            position = (random.randint(50, WIDTH - 150), random.randint(50, HEIGHT - 50))
            text_color = (random.randint(100, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.putText(frame, text, position, font, font_size, text_color, 2, cv2.LINE_AA)

        video.write(frame)

    video.release()
    print("Video generation complete.")

def generate_audio():
    """Generates eerie, unsettling soundscapes with random noises."""
    SAMPLE_RATE = 44100
    samples = np.zeros(SAMPLE_RATE * DURATION, dtype=np.int16)
    
    for i in range(DURATION):
        freq = random.choice([220, 440, 880, 120, 666, 333])
        volume = random.randint(5000, 15000)
        wave_data = (volume * np.sin(2 * np.pi * np.arange(SAMPLE_RATE) * freq / SAMPLE_RATE)).astype(np.int16)
        
        start = i * SAMPLE_RATE
        end = min(start + SAMPLE_RATE, len(samples))
        samples[start:end] = wave_data[:end - start]

        # Add random unsettling noises
        if random.random() > 0.5:
            noise = random.choice(NOISES)
            print(f"Adding noise: {noise}")
        
    with wave.open("audio_temp.wav", "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    print("Audio generation complete.")

def combine_video_audio():
    """Combines video and audio into a final MP4 file."""
    os.system(f"ffmpeg -y -i video_temp.mp4 -i audio_temp.wav -c:v copy -c:a aac -strict experimental {OUTPUT_FILE}")
    os.remove("video_temp.mp4")
    os.remove("audio_temp.wav")
    print(f"Final nightmare video saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_frames()
    generate_audio()
    combine_video_audio()

