import os
import random
import numpy as np
import subprocess
from PIL import Image, ImageDraw, ImageFont
import wave
import struct

# ------------------------------
# Helper functions for random parameters
# ------------------------------
def random_resolution():
    resolutions = [(320, 240), (640, 480), (1280, 720)]
    return random.choice(resolutions)

def random_frame_rate():
    return random.randint(5, 30)

def random_duration(max_sec):
    return random.randint(5, max_sec)

# ------------------------------
# Visual Generation
# ------------------------------
def generate_visuals(frame_count, resolution, style, output_folder="frames"):
    os.makedirs(output_folder, exist_ok=True)
    width, height = resolution
    # Try to load a font for text; fallback if not available.
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    words = ["ERR", "SAT", "WARNING", "CNT", "OVER", "VI", "SYSTEM"]
    
    for i in range(frame_count):
        # Background based on a chosen tonal color; vary slightly with each frame.
        base_color = (random.randint(30, 70), random.randint(30, 70), random.randint(150, 200))
        img = Image.new("RGB", (width, height), base_color)
        draw = ImageDraw.Draw(img)
        
        # Style selection drives the visual appearance.
        if style == 1:
            # Minimal geometric shapes with text
            if random.random() < 0.7:
                x0 = random.randint(0, width//2)
                y0 = random.randint(0, height//2)
                x1 = x0 + random.randint(10, width//2)
                y1 = y0 + random.randint(10, height//2)
                color = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
                draw.rectangle([x0, y0, x1, y1], outline=color, width=random.randint(1,5))
            if random.random() < 0.3:
                word = random.choice(words)
                draw.text((random.randint(0, width-100), random.randint(0, height-30)), word, font=font, fill=(255,255,255))
        
        elif style == 2:
            # Fast glitch style with multiple shapes and random text overlays
            for _ in range(random.randint(1,5)):
                shape_type = random.choice(["ellipse", "line", "rectangle"])
                x0 = random.randint(0, width)
                y0 = random.randint(0, height)
                x1 = random.randint(0, width)
                y1 = random.randint(0, height)
                color = tuple(random.randint(0,255) for _ in range(3))
                if shape_type == "ellipse":
                    draw.ellipse([min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)], outline=color, width=random.randint(1,3))
                elif shape_type == "line":
                    draw.line([x0, y0, x1, y1], fill=color, width=random.randint(1,3))
                else:
                    draw.rectangle([min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)], outline=color, width=random.randint(1,3))
            if random.random() < 0.5:
                draw.text((random.randint(0, width-100), random.randint(0, height-30)), random.choice(words), font=font, fill=(255,255,255))
        
        elif style == 3:
            # Webdriver Torso inspired: large central rectangle with minimalistic details
            margin = random.randint(20, 50)
            tonal_shift = random.randint(-20, 20)
            rect_color = (min(255, max(0, base_color[0]+tonal_shift)),
                          min(255, max(0, base_color[1]+tonal_shift)),
                          min(255, max(0, base_color[2]+tonal_shift)))
            draw.rectangle([margin, margin, width-margin, height-margin], outline=rect_color, width=5)
            if random.random() < 0.4:
                draw.text((width//4, height//2), random.choice(words), font=font, fill=(255,255,255))
        
        else:
            # Default minimal style with random diagonal lines
            draw.line([0, 0, width, height], fill=(255,255,255), width=2)
            draw.line([width, 0, 0, height], fill=(255,255,255), width=2)
        
        # Occasionally insert a random face (abstract representation)
        if random.random() < 0.2:
            face_x = random.randint(50, width-100)
            face_y = random.randint(50, height-100)
            draw.ellipse([face_x, face_y, face_x+50, face_y+50], outline="white", width=2)  # head
            draw.ellipse([face_x+15, face_y+15, face_x+20, face_y+20], fill="white")         # left eye
            draw.ellipse([face_x+30, face_y+15, face_x+35, face_y+20], fill="white")         # right eye
            draw.line([face_x+15, face_y+35, face_x+35, face_y+35], fill="white", width=2)    # mouth
        
        img.save(os.path.join(output_folder, f"frame_{i:03d}.png"))

# ------------------------------
# Audio Generation (Chord progression with beats and pauses)
# ------------------------------
def chord_frequencies(root_freq, scale_type="major"):
    # For a major chord, use intervals 0, 4, 7; for minor: 0, 3, 7.
    intervals = [0, 4, 7] if scale_type == "major" else [0, 3, 7]
    return [root_freq * (2 ** (i/12)) for i in intervals]

def generate_audio(filename, duration, sample_rate=44100):
    num_samples = duration * sample_rate
    audio = np.zeros(num_samples, dtype=np.float32)
    
    # Choose randomly between major or minor scale
    scale_type = random.choice(["major", "minor"])
    root_freq = random.choice([220, 261.63, 293.66, 329.63, 349.23, 392.00])  # A3, C4, D4, E4, F4, G4
    chord = chord_frequencies(root_freq, scale_type)
    
    # Generate chord progression and rhythmic pulses
    pos = 0
    while pos < num_samples:
        # Each chord lasts between 0.5 and 2 seconds.
        chord_duration = int(random.uniform(0.5, 2) * sample_rate)
        t = np.linspace(0, chord_duration/sample_rate, chord_duration, endpoint=False)
        chord_wave = np.zeros(chord_duration, dtype=np.float32)
        for freq in chord:
            chord_wave += np.sin(2 * np.pi * freq * t)
        chord_wave /= len(chord)
        # Randomly add a beat or effect
        if random.random() < 0.3:
            beat = np.random.uniform(-1, 1, chord_duration) * np.hanning(chord_duration)
            chord_wave += beat * 0.5
        # Occasionally insert a moment of silence
        if random.random() < 0.2:
            chord_wave *= 0
        end_pos = min(pos + chord_duration, num_samples)
        audio[pos:end_pos] = chord_wave[:end_pos-pos]
        pos += chord_duration

    # Add extra random percussive beats between chords
    for _ in range(random.randint(5,15)):
        pos = random.randint(0, num_samples-1000)
        audio[pos:pos+1000] += np.random.uniform(-0.5, 0.5, 1000)
    
    # Normalize audio to prevent clipping
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
    # Randomly choose parameters for this video
    resolution = random_resolution()
    frame_rate = random_frame_rate()
    duration = random_duration(15)  # Maximum 15 seconds
    frame_count = frame_rate * duration
    style = random.choice([1, 2, 3, 4])
    
    print(f"Resolution: {resolution}, Frame Rate: {frame_rate}, Duration: {duration}s, Style: {style}")
    
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
# Main entry point
# ------------------------------
if __name__ == "__main__":
    create_video("extreme_video.mp4")
