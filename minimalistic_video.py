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
    "ERROR", "NO SIGNAL", "OEOOEWEUELDEUAOOHAHEHOISRATEENLHOOOHTFJ", "IT'S WATCHING", "MISSING", "HELP ME",
    "WHO ARE YOU", "UNKNOWN CODE", "INITIATING SEQUENCE", "NOT ALONE"
]
SYMBOLS = ["∆", "Ω", "∑", "∂", "⊗", "Ξ", "☠", "✖", "ψ", "λ", "#@$!", "011001"]

# Initialize pygame mixer for eerie sound effects
os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.mixer.init(frequency=44100, size=-16, channels=1)

def generate_frames():
    """Generates eerie, unsettling video frames with glitches and distortions."""
    print(f"Generating {DURATION}-second nightmare video...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))
    
    last_color = None
    last_text = None
    
    for i in range(FRAME_COUNT):
        while True:
            bg_color = random.choice(random.choice(COLOR_SCHEMES))
            if bg_color != last_color:
                last_color = bg_color
                break
        
        frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)

        # Distorted noise overlay
        if random.random() > 0.5:
            noise = np.random.randint(0, 100, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

        # Random glitch effect
        if random.random() > 0.6:
            y_pos = random.randint(0, HEIGHT - 15)
            frame[y_pos:y_pos + 5, :] = frame[y_pos + 10:y_pos + 15, :]

        # Cryptic text overlay
        while True:
            text = random.choice(WORDS) + " " + random.choice(SYMBOLS)
            if text != last_text:
                last_text = text
                break
        
        if random.random() > 0.3:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = random.uniform(0.8, 2.0)
            position = (random.randint(50, WIDTH - 150), random.randint(50, HEIGHT - 50))
            text_color = (random.randint(100, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.putText(frame, text, position, font, font_size, text_color, 2, cv2.LINE_AA)

        # Flickering effect (random black frames)
        if random.random() > 0.8:
            frame[:, :] = (0, 0, 0)

        # Rare hidden message
        if i == random.randint(1, FRAME_COUNT - 2):
            cv2.putText(frame, "HELP ME", (WIDTH // 3, HEIGHT // 2), font, 2, (255, 255, 255), 3, cv2.LINE_AA)

        video.write(frame)

    video.release()
    print("Video generation complete.")

def generate_audio():
    """Generates eerie, unsettling soundscapes."""
    SAMPLE_RATE = 44100
    samples = np.zeros(SAMPLE_RATE * DURATION, dtype=np.int16)
    
    last_freq = None
    
    for i in range(DURATION):
        while True:
            freq = random.choice([220, 440, 880, 120, 666, 333])
            if freq != last_freq:
                last_freq = freq
                break
        
        volume = random.randint(4000, 12000)
        wave_data = (volume * np.sin(2 * np.pi * np.arange(SAMPLE_RATE) * freq / SAMPLE_RATE)).astype(np.int16)
        
        start = i * SAMPLE_RATE
        end = min(start + SAMPLE_RATE, len(samples))
        samples[start:end] = wave_data[:end - start]

        # Random silent moments or distortions
        if random.random() > 0.7:
            samples[start:start + SAMPLE_RATE // 4] = 0  # Silence
        if random.random() > 0.6:
            samples[start:start + SAMPLE_RATE // 5] *= -1  # Inverted sound

    # Save as WAV
    with wave.open("audio_temp.wav", "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    print("Audio generation complete.")

def combine_video_audio():
    """Combines video and audio into a final mp4 file."""
    os.system(f"ffmpeg -y -i video_temp.mp4 -i audio_temp.wav -c:v copy -c:a aac -strict experimental {OUTPUT_FILE}")
    os.remove("video_temp.mp4")
    os.remove("audio_temp.wav")
    print(f"Final nightmare video saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_frames()
    generate_audio()
    combine_video_audio()
