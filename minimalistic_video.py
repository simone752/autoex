import numpy as np
import cv2
import pygame
import random
import wave
import os

# Video settings
WIDTH, HEIGHT = 640, 480
DURATION = random.randint(10, 20)  # Longer duration for a lingering effect
FPS = random.choice([6, 10, 12])  # Fewer frames per second for eerie slowness
FRAME_COUNT = DURATION * FPS
OUTPUT_FILE = "extreme_video.mp4"

# Color palettes for background shifts
BACKGROUND_COLORS = [
    (10, 10, 10), (30, 5, 20), (15, 15, 40), (5, 25, 5), (40, 10, 10), (5, 5, 50)
]
FOREGROUND_COLORS = [
    (200, 50, 50), (50, 200, 50), (50, 50, 200), (220, 220, 50), (150, 50, 220)
]

# Random unsettling words/symbols
WORDS = ["HEROHIIHODISTETH", "NOISE", "0001101", "HEROHIIHODISTETH", "LOST", "HEROHIIHODISTETHHEROHIIHODISTETH", "END", "HALT"]
SYMBOLS = ["∆", "Ω", "⊗", "Ξ", "∑", "∞", "⧖", "⨉", "⨀"]

# Initialize pygame mixer for glitchy sound generation
os.environ["SDL_AUDIODRIVER"] = "dummy"
pygame.mixer.init(frequency=44100, size=-16, channels=1)

def generate_frames():
    print(f"Generating {DURATION}-second uncanny video at {WIDTH}x{HEIGHT}, {FPS} fps.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(FRAME_COUNT):
        # Background shifts between dark tones
        bg_color = random.choice(BACKGROUND_COLORS)
        frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)

        # Add unpredictable grain/noise effect
        if random.random() > 0.6:
            noise = np.random.randint(0, 50, (HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

        # Overlay shifting symbols & words
        if random.random() > 0.3:
            text = random.choice(WORDS) + " " + random.choice(SYMBOLS)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = random.uniform(0.8, 2.0)
            position = (random.randint(50, WIDTH-150), random.randint(50, HEIGHT-50))
            color = random.choice(FOREGROUND_COLORS)
            cv2.putText(frame, text, position, font, font_size, color, 2, cv2.LINE_AA)

        # Occasional scanline effect
        if random.random() > 0.7:
            for _ in range(random.randint(3, 10)):
                y_pos = random.randint(0, HEIGHT)
                cv2.line(frame, (0, y_pos), (WIDTH, y_pos), (50, 50, 50), 1)

        video.write(frame)

    video.release()
    print("Video generation complete.")

def generate_audio():
    SAMPLE_RATE = 44100
    samples = np.zeros(SAMPLE_RATE * DURATION, dtype=np.int16)

    for i in range(DURATION):
        freq = random.choice([220, 330, 440, 660, 880])  # Creepy harmonic tones
        volume = random.randint(5000, 20000)

        wave_data = (volume * np.sin(2 * np.pi * np.arange(SAMPLE_RATE) * freq / SAMPLE_RATE)).astype(np.int16)
        start, end = i * SAMPLE_RATE, (i + 1) * SAMPLE_RATE
        samples[start:end] = wave_data[:SAMPLE_RATE]

        # Introduce random audio distortions
        if random.random() > 0.7:
            glitch_start = random.randint(start, end - SAMPLE_RATE // 10)
            samples[glitch_start:glitch_start + SAMPLE_RATE // 10] = np.random.randint(-20000, 20000, SAMPLE_RATE // 10)

        # Reverse segments of the sound
        if random.random() > 0.5:
            reverse_start = random.randint(0, SAMPLE_RATE * (DURATION // 2))
            samples[reverse_start:reverse_start + SAMPLE_RATE // 4] = samples[reverse_start:reverse_start + SAMPLE_RATE // 4][::-1]

    # Save the audio as a WAV file
    with wave.open("audio.wav", "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    print("Audio generation complete.")

def combine_video_audio():
    if not os.path.exists("video_temp.mp4") or not os.path.exists("audio.wav"):
        print("Error: Missing video or audio file. Video generation failed.")
        return
    
    temp_output = "temp_uncanny_video.mp4"

    # Ensure old files are removed before creating a new one
    if os.path.exists(temp_output):
        os.remove(temp_output)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    os.system(f"ffmpeg -y -i video_temp.mp4 -i audio.wav -c:v libx264 -c:a aac -strict experimental {temp_output}")

    if not os.path.exists(temp_output):
        print("FFmpeg failed to generate the final video. Check your installation.")
        return

    os.rename(temp_output, OUTPUT_FILE)
    os.remove("video_temp.mp4")
    os.remove("audio.wav")

    print("Final eerie video with sound is ready.")

if __name__ == "__main__":
    generate_frames()
    generate_audio()
    combine_video_audio()

