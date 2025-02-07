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
WORDS = ["SIGNAL", "HEROHIIHODISTETH", "ERROR", "TMNGASGGHAATSS", "TRANSMISSION", "FNTNFERWNERNT", "GSUDRYWREYWNAT", "SNOTRDLENFDOEOOEWEUELDEUAO?", "WHO?", "OHAHEHOISRATEENLHOOOHTFJ", "NOISE"]
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
                        font, random.uniform(0.5, 1.5), (0, 0, 0), 

