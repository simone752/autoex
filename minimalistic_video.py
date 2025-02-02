import os
import random
import numpy as np
import subprocess
import wave
import struct
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# ------------------------------
# Global Constants & Helper Functions
# ------------------------------
SCARY_WORDS = ["ERROR", "ALERT", "OVERRIDE", "SYSTEM FAILURE", "WARNING", "VIRUS", "CRITICAL", "SHUTDOWN"]
IMAGE_URL = "https://picsum.photos/{w}/{h}"  # Random image API

def random_resolution():
    options = [(320, 240), (640, 480), (800, 600), (1280, 720)]
    return random.choice(options)

def random_frame_rate():
    return random.randint(5, 30)

def random_duration(max_sec=15, min_sec=5):
    return random.randint(min_sec, max_sec)

def download_random_image(width, height):
    try:
        url = IMAGE_URL.format(w=width, h=height)
        resp = requests.get(url)
        if resp.status_code == 200:
            return Image.open(BytesIO(resp.content))
    except Exception as e:
        print("Error downloading image:", e)
    return None

# ------------------------------
# Visual Generation
# ------------------------------
def generate_visuals(frame_count, resolution, style, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    width, height = resolution

    # Try to load a truetype font; if not available, use default.
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for i in range(frame_count):
        # Choose a background based on a tonal color that will persist throughout the video
        base_color = (random.randint(30, 70), random.randint(30, 70), random.randint(150, 200))
        img = Image.new("RGB", (width, height), base_color)
        draw = ImageDraw.Draw(img)

        # Different style patterns
        if style == 1:
            # Minimal geometric shapes plus a random word
            if random.random() < 0.7:
                x0 = random.randint(0, width // 2)
                y0 = random.randint(0, height // 2)
                x1 = x0 + random.randint(20, width // 2)
                y1 = y0 + random.randint(20, height // 2)
                color = tuple(random.randint(100, 255) for _ in range(3))
                draw.rectangle([x0, y0, x1, y1], outline=color, width=random.randint(2, 5))
            if random.random() < 0.5:
                word = random.choice(SCARY_WORDS)
                pos = (random.randint(0, width - 100), random.randint(0, height - 30))
                draw.text(pos, word, font=font, fill="white")
        elif style == 2:
            # Glitch style: overlapping shapes and text
            for _ in range(random.randint(2, 6)):
                shape_type = random.choice(["ellipse", "line", "rectangle"])
                x0 = random.randint(0, width)
                y0 = random.randint(0, height)
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                color = tuple(random.randint(0, 255) for _ in range(3))
                if shape_type == "ellipse":
                    draw.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=color, width=random.randint(1, 3))
                elif shape_type == "line":
                    draw.line([x0, y0, x1, y1], fill=color, width=random.randint(1, 3))
                else:
                    draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=color, width=random.randint(1, 3))
            if random.random() < 0.5:
                draw.text((random.randint(0, width - 100), random.randint(0, height - 30)), random.choice(SCARY_WORDS), font=font, fill="yellow")
        elif style == 3:
            # Webdriver Torso inspired: a large central rectangle and minimal text
            margin = random.randint(20, 50)
            draw.rectangle([margin, margin, width - margin, height - margin], outline="red", width=5)
            if random.random() < 0.4:
                draw.text((width // 3, height // 2), random.choice(SCARY_WORDS), font=font, fill="white")
        elif style == 4:
            # Incorporate a random downloaded image snippet plus simple overlays
            if random.random() < 0.5:
                snippet = download_random_image(width // 2, height // 2)
                if snippet is not None:
                    snippet = snippet.resize((width // 2, height // 2))
                    img.paste(snippet, (random.randint(0, width - width // 2), random.randint(0, height - height // 2)))
            for _ in range(2):
                x0 = random.randint(0, width)
                y0 = random.randint(0, height)
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                draw.line([x0, y0, x1, y1], fill="white", width=2)
            if random.random() < 0.3:
                draw.text((random.randint(0, width - 100), random.randint(0, height - 30)), random.choice(SCARY_WORDS), font=font, fill="cyan")
        else:
            # Default minimal style: two diagonal lines
            draw.line([0, 0, width, height], fill="white", width=2)
            draw.line([width, 0, 0, height], fill="white", width=2)

        # Occasionally insert a random abstract face
        if random.random() < 0.2:
            face_x = random.randint(50, width - 100)
            face_y = random.randint(50, height - 100)
            draw.ellipse([face_x, face_y, face_x + 50, face_y + 50], outline="white", width=2)
            draw.ellipse([face_x + 15, face_y + 15, face_x + 20, face_y + 20], fill="white")
            draw.ellipse([face_x + 30, face_y + 15, face_x + 35, face_y + 20], fill="white")
            draw.line([face_x + 15, face_y + 35, face_x + 35, face_y + 35], fill="white", width=2)

        img.save(os.path.join(output_folder, f"frame_{i:03d}.png"))

# ------------------------------
# Audio Generation
# ------------------------------
def chord_frequencies(root_freq, scale_type="major"):
    # Major: intervals 0, 4, 7; Minor: intervals 0, 3, 7
    intervals = [0, 4, 7] if scale_type == "major" else [0, 3, 7]
    return [root_freq * (2 ** (i/12)) for i in intervals]

def generate_audio(filename, duration, sample_rate=44100):
    num_samples = duration * sample_rate
    audio = np.zeros(num_samples, dtype=np.float32)

    # Choose scale type and chord progression
    scale_type = random.choice(["major", "minor"])
    root_freq = random.choice([220, 261.63, 293.66, 329.63, 349.23, 392.00])
    chord = chord_frequencies(root_freq, scale_type)

    pos = 0
    while pos < num_samples:
        chord_duration = int(random.uniform(0.5, 2) * sample_rate)
        t = np.linspace(0, chord_duration/sample_rate, chord_duration, endpoint=False)
        chord_wave = sum(np.sin(2 * np.pi * freq * t) for freq in chord) / len(chord)

        # Add industrial noise and beats
        if random.random() < 0.3:
            beat = np.random.uniform(-0.5, 0.5, chord_duration) * np.hanning(chord_duration)
            chord_wave += beat
        if random.random() < 0.2:
            chord_wave *= 0  # momentary silence

        end_pos = min(pos + chord_duration, num_samples)
        audio[pos:end_pos] = chord_wave[:end_pos - pos]
        pos += chord_duration

    # Add extra percussive effects
    for _ in range(random.randint(5, 15)):
        pos = random.randint(0, num_samples - 1000)
        audio[pos:pos + 1000] += np.random.uniform(-0.3, 0.3, 1000)

    # Normalize and convert to int16
    max_val = np.max(np.abs(audio)) if np.max(np.abs(audio)) != 0 else 1
    audio = audio / max_val
    audio_int16 = np.int16(audio * 32767)

    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

# ------------------------------
# Video Assembly
# ------------------------------
def create_video(output_filename):
    # Randomize video parameters
    resolution = random_resolution()
    frame_rate = random_frame_rate()
    duration = random_duration(15, 5)  # between 5 and 15 seconds
    frame_count = frame_rate * duration
    style = random.choice([1, 2, 3, 4])
    
    print(f"Creating video with resolution {resolution}, {frame_rate} fps, duration {duration}s, style {style}")
    
    generate_visuals(frame_count, resolution, style)
    generate_audio("temp_audio.wav", duration)
    
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(frame_rate),
        "-i", "frames/frame_%03d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "temp_video.mp4"
    ], check=True)
    
    subprocess.run([
        "ffmpeg", "-y", "-i", "temp_video.mp4", "-i", "temp_audio.wav",
        "-c:v", "copy", "-c:a", "aac", output_filename
    ], check=True)
    
    print(f"Video saved as {output_filename}")

# ------------------------------
# Main Entry Point
# ------------------------------
if __name__ == "__main__":
    create_video("extreme_video.mp4")
