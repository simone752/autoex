import numpy as np
import cv2
import random
import string
import wave
import os

# Video settings
WIDTH, HEIGHT = 640, 480
DURATION = random.randint(8, 15)  # 8-15 sec
FPS = random.choice([4, 6, 8, 12])  # Uncanny slow movement
FRAME_COUNT = DURATION * FPS
OUTPUT_FILE = "uncanny_video.mp4"

# Backgrounds: Dark gradients, noise textures, corrupted visuals
BACKGROUND_MODES = ["gradient", "noise", "glitch", "solid_dark"]
WORDS = ["HEROHIIHODISTETH", "LOST?", "NOTHING", "HEROHIIHODISTETH", "LISTEN", "???", "ERROR", "HEROHIIHODISTETH?", "HELLO?", "HEROHIIHODISTETH"]
SYMBOLS = ["∆", "Ω", "∑", "∂", "∫", "≈", "⊗", "Ξ", "Ψ", "?", "#", "!!!", "%", "@", "█"]

# Functions for weird backgrounds
def generate_gradient():
    """Creates a weird gradient with shifting colors."""
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        color = [random.randint(0, 50), random.randint(0, 50), random.randint(100, 255)]
        img[y, :, :] = color
    return img

def generate_noise():
    """Generates static noise effect."""
    return np.random.randint(0, 100, (HEIGHT, WIDTH, 3), dtype=np.uint8)

def generate_glitch():
    """Creates a corrupted frame effect."""
    img = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    for _ in range(random.randint(3, 7)):
        x, y = random.randint(0, WIDTH//2), random.randint(0, HEIGHT//2)
        w, h = random.randint(WIDTH//4, WIDTH//2), random.randint(HEIGHT//4, HEIGHT//2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (random.randint(100, 255), 0, random.randint(100, 255)), -1)
    return img

def generate_solid_dark():
    """A solid dark color with slight variations."""
    color = [random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)]
    return np.full((HEIGHT, WIDTH, 3), color, dtype=np.uint8)

# Generate video
def generate_frames():
    print(f"Generating an {DURATION}-second eerie video at {WIDTH}x{HEIGHT} resolution and {FPS} fps.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))

    for i in range(FRAME_COUNT):
        background_type = random.choice(BACKGROUND_MODES)

        # Select a disturbing background
        if background_type == "gradient":
            frame = generate_gradient()
        elif background_type == "noise":
            frame = generate_noise()
        elif background_type == "glitch":
            frame = generate_glitch()
        else:
            frame = generate_solid_dark()

        # Occasionally add weird, broken text
        if random.random() > 0.5:
            text = random.choice(WORDS) + " " + random.choice(SYMBOLS)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (random.randint(20, WIDTH-100), random.randint(50, HEIGHT-50)), 
                        font, random.uniform(0.8, 2.0), (random.randint(150, 255), 0, 0), random.randint(1, 3), cv2.LINE_AA)

        # Sudden color shifts or inverted frame effect
        if random.random() < 0.2:
            frame = 255 - frame  # Invert colors for sudden flashes

        video.write(frame)

    video.release()
    print("Video generation complete.")

# Generate eerie audio
def generate_audio():
    SAMPLE_RATE = 44100
    samples = np.zeros(SAMPLE_RATE * DURATION, dtype=np.int16)

    for i in range(DURATION):
        freq = random.choice([220, 440, 880, 100, 50])  # Mix of deep and high frequencies
        volume = random.randint(3000, 12000)
        wave_data = (volume * np.sin(2 * np.pi * np.arange(SAMPLE_RATE) * freq / SAMPLE_RATE)).astype(np.int16)

        # Distorted sound processing
        if random.random() < 0.3:
            wave_data = np.flip(wave_data)  # Reverse sound
        if random.random() < 0.2:
            wave_data = wave_data * np.hamming(len(wave_data))  # Soft fading
        if random.random() < 0.1:
            wave_data = wave_data * -1  # Phase inversion for creepier effect

        start, end = i * SAMPLE_RATE, (i + 1) * SAMPLE_RATE
        samples[start:end] = wave_data[:SAMPLE_RATE]

    # Background hum or static noise
    if random.random() > 0.5:
        static_noise = (np.random.normal(0, 1000, samples.shape)).astype(np.int16)
        samples += static_noise

    # Moments of eerie silence
    if random.random() > 0.4:
        silence_start = random.randint(1, DURATION - 2) * SAMPLE_RATE
        samples[silence_start:silence_start + (SAMPLE_RATE // 3)] = 0

    # Save audio as WAV
    with wave.open("audio.wav", "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())

    print("Audio generation complete.")

# Combine video and audio
def combine_video_audio():
    temp_output = "temp_uncanny_video.mp4"
    os.system(f"ffmpeg -y -i video_temp.mp4 -i audio.wav -c:v copy -c:a aac {temp_output}")
    os.remove("video_temp.mp4")
    os.remove("audio.wav")
    os.rename(temp_output, OUTPUT_FILE)
    print("Final eerie video with sound is ready.")

if __name__ == "__main__":
    generate_frames()
    generate_audio()
    combine_video_audio()

