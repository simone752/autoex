import os
import random
import string
import moviepy.editor as mp
import datetime
from pydub.generators import WhiteNoise
from cryptography.fernet import Fernet

# Directory to store generated videos
os.makedirs("generated_videos", exist_ok=True)

# Generate random creepy visuals
def create_random_video(output_path):
    duration = random.randint(5, 15)  # video duration between 5 to 15 seconds
    width, height = 640, 480

    # Generate random colors frame by frame
    frames = [mp.ColorClip(size=(width, height), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), duration=0.1) for _ in range(duration * 10)]
    video = mp.concatenate_videoclips(frames, method="compose")

    # Add noise audio
    audio = WhiteNoise().to_audio_segment(duration=duration * 1000)
    audio.export("temp_audio.wav", format="wav")
    video = video.set_audio(mp.AudioFileClip("temp_audio.wav"))

    video.write_videofile(output_path, fps=24, codec='libx264')
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

# Main function to generate and upload video
if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    video_filename = f"generated_videos/video_{timestamp}.mp4"
    
    # Create the video
    create_random_video(video_filename)

    # Generate an encrypted title
    title = generate_encrypted_title()
    print(f"Generated video saved as {video_filename} with title: {title}")
