## minimalistic_video_fast.py
# Rewritten, fast & reliable version for ARG-style glitch video generation

import numpy as np
import cv2
import wave
import os
import random
from scipy.io import wavfile
from scipy.signal import chirp

# Config
WIDTH, HEIGHT = 640, 360
FPS = 24
DURATION = 10  # seconds
FRAMES = FPS * DURATION
AUDIO_SAMPLE_RATE = 44100
AUDIO_DURATION = DURATION

# Output
VIDEO_FILENAME = "output_glitch.mp4"
AUDIO_FILENAME = "temp_audio.wav"
FINAL_OUTPUT = "final_output.mp4"

# Utilities
def make_text_frame(text, frame_num):
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (50, HEIGHT//2), font, 1, color, 2, cv2.LINE_AA)
    return img

def glitch_frame(frame):
    for _ in range(random.randint(1, 4)):
        y = random.randint(0, HEIGHT-10)
        h = random.randint(1, 5)
        x_shift = random.randint(-30, 30)
        frame[y:y+h] = np.roll(frame[y:y+h], x_shift, axis=1)
    return frame

# Video Generation
out = cv2.VideoWriter(VIDEO_FILENAME, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))
for i in range(FRAMES):
    text = random.choice(["GLITCH", "SIGNAL", "CORRUPT", "VOID"])
    frame = make_text_frame(text, i)
    frame = glitch_frame(frame)
    out.write(frame)
out.release()

# Audio Generation (simple chirp and noise)
t = np.linspace(0, AUDIO_DURATION, int(AUDIO_SAMPLE_RATE * AUDIO_DURATION))
audio = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.random.randn(len(t))
audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
wavfile.write(AUDIO_FILENAME, AUDIO_SAMPLE_RATE, audio)

# Merge audio + video
os.system(f"ffmpeg -y -i {VIDEO_FILENAME} -i {AUDIO_FILENAME} -c:v copy -c:a aac -shortest {FINAL_OUTPUT}")

# Clean temp files
os.remove(VIDEO_FILENAME)
os.remove(AUDIO_FILENAME)

print(f"✅ Video generated: {FINAL_OUTPUT}")











