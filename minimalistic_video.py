import os
import subprocess
import wave
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ===== CONFIGURATION =====
num_segments = 30           # Total seconds (segments) for the video
segment_duration = 1        # Duration of each segment (in seconds)
output_fps = 24             # Final video playback FPS
width, height = 640, 480     # Video resolution
sample_rate = 44100         # Audio sample rate (Hz)

# Filenames and directories
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)
audio_filename = "temp_audio.wav"
video_from_frames = "temp_video.mp4"
final_video = "extreme_video.mp4"

# Attempt to load a system font; fall back to default if not found.
try:
    font = ImageFont.truetype("arial.ttf", 32)
except Exception:
    font = ImageFont.load_default()

# ===== FRAME GENERATION (VISUALS) =====
print("Generating extreme frames...")
for i in range(num_segments):
    # Randomly choose between a very dark or an ultra-bright background
    if random.random() < 0.5:
        bg_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    else:
        bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw a number of random geometric shapes
    for j in range(random.randint(3, 8)):
        shape_type = random.choice(["ellipse", "rectangle", "line"])
        x0 = random.randint(0, width)
        y0 = random.randint(0, height)
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        shape_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if shape_type == "ellipse":
            draw.ellipse([x0, y0, x1, y1], outline=shape_color, width=random.randint(1, 5))
        elif shape_type == "rectangle":
            draw.rectangle([x0, y0, x1, y1], outline=shape_color, width=random.randint(1, 5))
        elif shape_type == "line":
            draw.line([x0, y0, x1, y1], fill=shape_color, width=random.randint(1, 5))
    
    # Occasionally overlay some abrupt, alarming text
    if random.random() < 0.5:
        text = random.choice(["ERROR", "ALERT", "WARNING", "SYSTEM FAILURE", "CRITICAL", "!!!"])
        text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        text_position = (random.randint(0, width - 100), random.randint(0, height - 50))
        draw.text(text_position, text, font=font, fill=text_color)
    
    # Apply a random effect to distort the image further
    effect = random.choice(["blur", "invert", "color_shift", "none"])
    if effect == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 3)))
    elif effect == "invert":
        img = Image.fromarray(255 - np.array(img))
    elif effect == "color_shift":
        arr = np.array(img)
        shift = random.randint(-50, 50)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + shift, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] - shift, 0, 255)
        img = Image.fromarray(arr.astype("uint8"))
    
    # Save the frame image
    frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
    img.save(frame_path)

# ===== AUDIO GENERATION (SOUND) =====
print("Generating extreme audio...")
audio_segments = []
for i in range(num_segments):
    # Pick wildly varying base and modulation frequencies
    base_freq = random.uniform(300, 1200)
    mod_freq = random.uniform(1, 10)
    mod_index = random.uniform(5, 30)
    t = np.linspace(0, segment_duration, int(sample_rate * segment_duration), endpoint=False)
    
    # Generate a modulated sine wave (vibrato and instability)
    sine_wave = np.sin(2 * np.pi * base_freq * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
    sine_wave *= random.uniform(0.4, 0.9)
    
    # Insert occasional noise bursts for glitchy abrasiveness
    if random.random() < 0.3:
        burst_start = random.randint(0, len(t) - 100)
        burst_duration = random.randint(20, 100)
        sine_wave[burst_start:burst_start + burst_duration] += np.random.normal(0, 1, burst_duration)
    
    # Add an overall layer of noise
    noise = np.random.normal(0, 0.1, sine_wave.shape)
    segment_audio = sine_wave + noise
    audio_segments.append(segment_audio)

# Concatenate all audio segments into one continuous array
audio = np.concatenate(audio_segments)
audio = audio / np.max(np.abs(audio))
audio_int16 = np.int16(audio * 32767)

# Write the audio to a WAV file
with wave.open(audio_filename, "w") as wav_file:
    wav_file.setparams((1, 2, sample_rate, len(audio_int16), "NONE", "not compressed"))
    wav_file.writeframes(audio_int16.tobytes())

# ===== VIDEO ASSEMBLY VIA FFMPEG =====
print("Creating video from frames...")
ffmpeg_video_cmd = [
    "ffmpeg",
    "-y",                      # Overwrite output if exists
    "-framerate", "1",         # Each image lasts 1 second
    "-i", os.path.join(frames_dir, "frame_%03d.png"),
    "-r", str(output_fps),     # Output playback FPS (e.g. 24 fps)
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    video_from_frames
]
subprocess.run(ffmpeg_video_cmd, check=True)

print("Combining video and audio into final output...")
ffmpeg_combine_cmd = [
    "ffmpeg",
    "-y",
    "-i", video_from_frames,
    "-i", audio_filename,
    "-shortest",              # End when the shortest input ends
    "-c:v", "copy",           # Copy video stream without re-encoding
    "-c:a", "aac",            # Encode audio with AAC
    final_video
]
subprocess.run(ffmpeg_combine_cmd, check=True)

print("Extreme video generated successfully:", final_video)
