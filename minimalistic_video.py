import numpy as np
import cv2
import random
import string
import wave
import os

# Video settings
WIDTH, HEIGHT = 640, 480
DURATION = random.randint(10, 20)  # Extended duration (10-20 sec)
FPS = random.choice([6, 12, 15, 20])  # Slower frame rates for unsettling effect
FRAME_COUNT = DURATION * FPS
OUTPUT_FILE = "extreme_video.mp4"

# Colors (alternating red/blue like Webdriver Torso, but with eerie additions)
BASE_COLORS = [(255, 0, 0), (0, 0, 255)]
ALT_COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (50, 50, 50), (200, 50, 50), (50, 200, 50)]

# Random words and symbols for overlay (mix of normal and unsettling words)
WORDS = ["SIGNAL", "HEROHIIHODISTETH", "ERROR", "TEST", "HEROHIIHODISTETH", "LOST", "VOID", "HEROHIIHODISTETH?", "WHO?", "HEROHIIHODISTETH", "NOISE"]
SYMBOLS = ["∆", "Ω", "∑", "∂", "∫", "≈", "⊗", "Ξ", "Ψ", "?", "#", "!!", "!!!"]

# Glitch effect probability
GLITCH_PROB = 0.2

def generate_frames():
    print(f"Generating a {DURATION}-second eerie video at {WIDTH}x{HEIGHT} resolution and {FPS} fps.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(FRAME_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)  # White background

        # Unpredictable color shifts and glitches
        if random.random() < GLITCH_PROB:
            color_choice = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Glitchy random colors
        else:
            color_choice = BASE_COLORS[i % 2] if random.random() > 0.3 else random.choice(ALT_COLORS)

        # Random rectangles (sometimes oversized or misplaced)
        x0, y0 = random.randint(0, WIDTH // 2), random.randint(0, HEIGHT // 2)
        x1, y1 = random.randint(WIDTH // 2, WIDTH), random.randint(HEIGHT // 2, HEIGHT)
        if random.random() < 0.15:
            x0, y0, x1, y1 = 0, 0, WIDTH, HEIGHT  # Full screen flash
        cv2.rectangle(frame, (x0, y0), (x1, y1), color_choice, -1)

        # Occasionally add random eerie text or symbols
        if random.random() > 0.6:
            text = random.choice(WORDS) + " " + random.choice(SYMBOLS)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (random.randint(50, WIDTH-100), random.randint(50, HEIGHT-50)), 
                        font, random.uniform(0.5, 1.5), (0, 0, 0), random.choice([1, 2, 3]), cv2.LINE_AA)

        # Distorted noise overlay
        if random.random() < 0.1:
            noise = np.random.randint(0, 50, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

        video.write(frame)

    video.release()
    print("Video generation complete.")

def generate_audio():
    SAMPLE_RATE = 44100
    samples = np.zeros(SAMPLE_RATE * DURATION, dtype=np.int16)

    for i in range(DURATION):
        freq = random.choice([220, 440, 880, 1760])  # Low to high unsettling tones
        volume = random.randint(3000, 15000)
        wave_data = (volume * np.sin(2 * np.pi * np.arange(SAMPLE_RATE) * freq / SAMPLE_RATE)).astype(np.int16)

        # Apply eerie effects
        if random.random() < 0.3:
            wave_data = np.flip(wave_data)  # Reverse sound chunks
        if random.random() < 0.2:
            wave_data = wave_data * np.hamming(len(wave_data))  # Soft fade in/out effect

        start, end = i * SAMPLE_RATE, (i + 1) * SAMPLE_RATE
        samples[start:end] = wave_data[:SAMPLE_RATE]

    # Add static or background hum
    if random.random() > 0.5:
        static_noise = (np.random.normal(0, 1000, samples.shape)).astype(np.int16)
        samples += static_noise

    # Introduce unexpected silence
    if random.random() > 0.5:
        silence_start = random.randint(1, DURATION - 2) * SAMPLE_RATE
        samples[silence_start:silence_start + (SAMPLE_RATE // 3)] = 0

    # Save audio as WAV
    with wave.open("audio.wav", "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    print("Audio generation complete.")

def combine_video_audio():
    temp_output = "temp_extreme_video.mp4"
    
    # Ensure ffmpeg overwrites correctly
    os.system(f"ffmpeg -y -i video_temp.mp4 -i audio.wav -c:v copy -c:a aac {temp_output}")

    # Clean up temp files
    os.remove("video_temp.mp4")
    os.remove("audio.wav")
    os.rename(temp_output, OUTPUT_FILE)

    print("Final eerie video with sound is ready.")

if __name__ == "__main__":
    generate_frames()
    generate_audio()
    combine_video_audio()
