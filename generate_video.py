import os
import subprocess
import wave
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# === CONFIGURATION PARAMETERS ===
num_segments = 20           # Total seconds (segments) of the video
segment_duration = 1        # Duration per segment in seconds
output_fps = 24             # Final output frame rate
width, height = 640, 480     # Video resolution
sample_rate = 44100         # Audio sample rate in Hz

# === DIRECTORIES AND FILENAMES ===
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)
audio_filename = "temp_audio.wav"
video_from_frames = "temp_video.mp4"
final_video = "minimalistic_video.mp4"

# === FRAME GENERATION (VISUALS) ===
print("Generating frames...")
for i in range(num_segments):
    # Create a very dark background (to evoke an eerie feel)
    bg_color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40))
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Draw a central rectangle reminiscent of Webdriver Torso but with a twist:
    # The rectangle’s fill is a random neon-ish color that contrasts with the dark background.
    margin = random.randint(50, 150)
    rect_color = (random.randint(100, 255), random.randint(0, 100), random.randint(100, 255))
    draw.rectangle([margin, margin, width - margin, height - margin], fill=rect_color)

    # Add a few sporadic glitch-like lines
    for _ in range(random.randint(2, 5)):
        x0 = random.randint(0, width)
        y0 = random.randint(0, height)
        x1 = x0 + random.randint(-100, 100)
        y1 = y0 + random.randint(-100, 100)
        line_color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        draw.line([x0, y0, x1, y1], fill=line_color, width=random.randint(1, 3))

    # Optionally apply a slight blur or noise effect
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))

    # Save frame as "frame_000.png", "frame_001.png", etc.
    frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
    img.save(frame_path)

# === AUDIO GENERATION (SOUND) ===
print("Generating audio...")
audio_segments = []
for i in range(num_segments):
    # Choose a base frequency (400-800 Hz) for a beep-like tone
    base_freq = random.uniform(400, 800)
    # Add frequency modulation for a vibrato effect
    mod_freq = random.uniform(2, 7)      # modulation frequency in Hz
    mod_index = random.uniform(5, 20)    # modulation depth
    t = np.linspace(0, segment_duration, int(sample_rate * segment_duration), endpoint=False)
    # Generate the modulated sine wave
    sine_wave = np.sin(2 * np.pi * base_freq * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
    # Random overall amplitude variation (to add unpredictability)
    sine_wave *= random.uniform(0.3, 0.7)
    # Add a very light layer of white noise to create a glitchy texture
    noise = np.random.normal(0, 0.05, sine_wave.shape)
    segment_audio = sine_wave + noise
    audio_segments.append(segment_audio)

# Concatenate audio segments into one continuous array
audio = np.concatenate(audio_segments)
# Normalize audio to 16-bit PCM range
audio = audio / np.max(np.abs(audio))
audio_int16 = np.int16(audio * 32767)

# Write audio to a WAV file
with wave.open(audio_filename, "w") as wav_file:
    n_channels = 1
    sampwidth = 2  # 16-bit PCM = 2 bytes
    n_frames = len(audio_int16)
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((n_channels, sampwidth, sample_rate, n_frames, comptype, compname))
    wav_file.writeframes(audio_int16.tobytes())

# === VIDEO ASSEMBLY VIA FFMPEG ===
print("Creating video from frames...")
# Build the ffmpeg command to convert frames to video.
# -framerate 1 means each image is held for 1 second.
# Then we set the output frame rate (-r) to output_fps (e.g. 24 fps) for smooth playback.
ffmpeg_video_cmd = [
    "ffmpeg",
    "-y",  # Overwrite output file if exists
    "-framerate", "1",
    "-i", os.path.join(frames_dir, "frame_%03d.png"),
    "-r", str(output_fps),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    video_from_frames
]
subprocess.run(ffmpeg_video_cmd, check=True)

print("Combining video and audio into final video...")
# Build the ffmpeg command to combine video and audio.
ffmpeg_combine_cmd = [
    "ffmpeg",
    "-y",
    "-i", video_from_frames,
    "-i", audio_filename,
    "-shortest",  # Ensure output ends when the shortest input ends
    "-c:v", "copy",  # Copy video stream without re-encoding
    "-c:a", "aac",   # Encode audio with AAC
    final_video
]
subprocess.run(ffmpeg_combine_cmd, check=True)

print("Video generated successfully:", final_video)
