
import random
import numpy as np
import cv2
import os
import pygame
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip

# Configuration Parameters
VIDEO_DURATION = random.uniform(5, 15)  # Randomized duration
FRAME_RATE = random.choice([12, 24, 30])  # Different speeds
RESOLUTION = random.choice([(640, 480), (800, 600), (1280, 720)])
COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5)]
NOISE_LEVEL = random.uniform(0.01, 0.1)
TEXT_WORDS = ["ERROR", "SIGNAL", "LOST", "DECODE", "VOID", "DATA"]

# Generate Video Frames
def generate_video_frames():
    frame_count = int(VIDEO_DURATION * FRAME_RATE)
    video_path = "output.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, RESOLUTION)
    
    for _ in range(frame_count):
        frame = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        frame[:, :] = random.choice(COLORS)
        if random.random() > 0.5:
            cv2.rectangle(frame, (random.randint(0, RESOLUTION[0] // 2), random.randint(0, RESOLUTION[1] // 2)),
                          (random.randint(RESOLUTION[0] // 2, RESOLUTION[0]), random.randint(RESOLUTION[1] // 2, RESOLUTION[1])),
                          random.choice(COLORS), -1)
        if random.random() > 0.7:
            text = random.choice(TEXT_WORDS)
            cv2.putText(frame, text, (random.randint(0, RESOLUTION[0] - 100), random.randint(50, RESOLUTION[1] - 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, random.choice(COLORS), 2, cv2.LINE_AA)
        out.write(frame)
    out.release()
    return video_path

# Generate Audio
def generate_audio():
    pygame.mixer.init()
    sample_rate = 44100
    duration = VIDEO_DURATION
    audio = np.zeros(int(sample_rate * duration))
    
    for i in range(random.randint(3, 6)):
        start = random.randint(0, len(audio) - 1000)
        freq = random.choice([220, 440, 880, 1760])  # Major/Minor Scale
        for j in range(1000):
            if start + j < len(audio):
                audio[start + j] = np.sin(2 * np.pi * freq * (j / sample_rate)) * 0.5
        
    if random.random() > 0.5:
        noise = np.random.uniform(-NOISE_LEVEL, NOISE_LEVEL, len(audio))
        audio += noise
    
    audio = np.clip(audio, -1, 1)
    audio_clip = AudioArrayClip(audio.reshape(-1, 1), fps=sample_rate)
    return audio_clip

# Merge Audio & Video
def create_video():
    video_path = generate_video_frames()
    audio_clip = generate_audio()
    video_clip = VideoFileClip(video_path).set_audio(audio_clip)
    final_video_path = "extreme_video.mp4"
    video_clip.write_videofile(final_video_path, codec='libx264', fps=FRAME_RATE)
    return final_video_path

if __name__ == "__main__":
    create_video()
