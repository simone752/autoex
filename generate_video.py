import os
import random
import datetime
import numpy as np
import ffmpeg
from pydub.generators import WhiteNoise
from cryptography.fernet import Fernet
from PIL import Image

# Directory to store generated videos
os.makedirs("generated_videos", exist_ok=True)

# Generate a random color image (as video background)
def create_random_video(output_path, duration=10, width=640, height=480, fps=24):
    # Generate a random solid color
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Create a blank image
    img = Image.new("RGB", (width, height), color)
    img_path = "temp_frame.png"
    img.save(img_path)

    # Generate white noise audio
    audio = WhiteNoise().to_audio_segment(duration=duration * 1000)
    audio.export("temp_audio.wav", format="wav")

    # Create video from a single image using FFmpeg
    ffmpeg.input(img_path, loop=1, framerate=fps, t=duration).output("temp_video.mp4", vcodec="libx264").run(overwrite_output=True)

    # Combine video and audio
    ffmpeg.input("temp_video.mp4").input("temp_audio.wav").output(output_path, vcodec="libx264", acodec="aac").run(overwrite_output=True)

    # Clean up temp files
    os.remove(img_path)
    os.remove("temp_video.mp4")
    os.remove("temp_audio.wav")

# Generate random encrypted title from Don Quixote quotes
def generate_encrypted_title():
    quotes = [
        "In short, he became so absorbed in his books that he spent his nights reading from dusk till dawn.",
        "Destiny guides our fortunes more favorably than we could have expected.",
        "Time ripens all things; no man is born wise."
    ]
    selected_quote = random.choice(quotes)

    # Encrypt the title
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    encrypted_text = cipher_suite.encrypt(selected_quote.encode()).decode()
    
    return encrypted_text

# Main function
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    video_filename = f"generated_videos/video_{timestamp}.mp4"

    # Create video
    create_random_video(video_filename)

    # Generate encrypted title
    title = generate_encrypted_title()
    print(f"Generated video saved as {video_filename} with title: {title}")
