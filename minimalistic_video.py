import numpy as np
import cv2
import random
import string
import wave
import os

# Video settings
WIDTH, HEIGHT = 640, 480
DURATION = random.randint(8, 15)  # Random duration (8-15 sec)
FPS = random.choice([10, 15, 24, 30])  # Varying frame rates
FRAME_COUNT = DURATION * FPS
OUTPUT_FILE = "extreme_video.mp4"

# Colors (like Webdriver Torso, but randomized)
BASE_COLORS = [(255, 0, 0), (0, 0, 255)]
ALT_COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 255)]

# Random words and symbols for overlay
WORDS = ["SIGNAL", "DATA", "ERROR", "TEST", "TRANSMISSION", "LOST", "CODE"]
SYMBOLS = ["∆", "Ω", "∑", "∂", "∫", "≈", "⊗", "Ξ", "Ψ"]

def generate_frames():
    print(f"Generating a {DURATION}-second video at {WIDTH}x{HEIGHT} resolution and {FPS} fps.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(FRAME_COUNT):
        frame = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)  # White background

        # Alternate rectangle colors (like Webdriver Torso)
        color_choice = BASE_COLORS[i % 2] if random.random() > 0.3 else random.choice(ALT_COLORS)
        x0, y0 = random.randint(0, WIDTH // 2), random.randint(0, HEIGHT // 2)
        x1, y1 = random.randint(WIDTH // 2, WIDTH), random.randint(HEIGHT // 2, HEIGHT)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color_choice, -1)

        # Occasionally add random text or symbols
        if random.random() > 0.7:
            text = random.choice(WORDS) + " " + random.choice(SYMBOLS)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (random.randint(50, WIDTH-100), random.randint(50, HEIGHT-50)), 
                        font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        video.write(frame)

    video.release()
    print("Video generation complete.")

def generate_audio():
    SAMPLE_RATE = 44100
    samples = np.zeros(SAMPLE_RATE * DURATION, dtype=np.int16)

    for i in range(DURATION):
        freq = random.choice([440, 880, 1760])  # Random A4-A6 notes like Webdriver Torso
        volume = random.randint(5000, 15000)
        wave_data = (volume * np.sin(2 * np.pi * np.arange(SAMPLE_RATE) * freq / SAMPLE_RATE)).astype(np.int16)
        start, end = i * SAMPLE_RATE, (i + 1) * SAMPLE_RATE
        samples[start:end] = wave_data[:SAMPLE_RATE]

    # Add moments of silence
    if random.random() > 0.5:
        silence_start = random.randint(1, DURATION - 2) * SAMPLE_RATE
        samples[silence_start:silence_start + (SAMPLE_RATE // 2)] = 0

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

    print("Final video with sound is ready.")

if __name__ == "__main__":
    generate_frames()
    generate_audio()
    combine_video_audio()
