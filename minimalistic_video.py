import os
import random
import numpy as np
import cv2
import pygame
import ffmpeg
from gtts import gTTS

# Constants
DURATION = random.randint(5, 15)  # Random video duration between 5 and 15 seconds
WIDTH, HEIGHT = random.choice([(640, 480), (1280, 720), (1920, 1080)])  # Random resolution
FRAME_RATE = random.choice([24, 30, 60])  # Random frame rate
FRAME_COUNT = DURATION * FRAME_RATE

# Generate unique text
WORDS = ["chaos", "echo", "void", "random", "silence", "dark", "light", "fracture", "glitch", "noise"]
random_text = " ".join(random.sample(WORDS, random.randint(2, 5)))

def generate_frames():
    os.makedirs("frames", exist_ok=True)
    for i in range(FRAME_COUNT):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.rectangle(frame, (random.randint(0, WIDTH//2), random.randint(0, HEIGHT//2)),
                      (random.randint(WIDTH//2, WIDTH), random.randint(HEIGHT//2, HEIGHT)),
                      color, -1)
        cv2.putText(frame, random_text, (50, HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(f"frames/frame_{i:04d}.png", frame)

def generate_audio():
    pygame.mixer.init()
    sample_rate = 44100
    samples = np.zeros((DURATION * sample_rate, 2), dtype=np.int16)
    for i in range(0, samples.shape[0], sample_rate // random.randint(4, 12)):
        frequency = random.choice([220, 440, 880, 1760])
        waveform = np.sin(2 * np.pi * np.arange(sample_rate) * frequency / sample_rate)
        samples[i:i+sample_rate//random.randint(8, 16), 0] = (waveform * 32767).astype(np.int16)
    pygame.mixer.quit()
    pygame.mixer.init(frequency=sample_rate, size=-16, channels=2)
    pygame.sndarray.make_sound(samples).play()
    pygame.time.delay(DURATION * 1000)
    pygame.mixer.quit()
    tts = gTTS(text=random_text, lang="en")
    tts.save("audio.mp3")

def create_video():
    os.system("ffmpeg -framerate {} -i frames/frame_%04d.png -c:v libx264 -r {} temp_video.mp4".format(FRAME_RATE, FRAME_RATE))
    os.system("ffmpeg -i temp_video.mp4 -i audio.mp3 -c:v copy -c:a aac -strict experimental output.mp4")

def cleanup():
    os.system("rm -rf frames temp_video.mp4 audio.mp3")

if __name__ == "__main__":
    generate_frames()
    generate_audio()
    create_video()
    cleanup()

