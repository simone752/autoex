import numpy as np
import pygame
import random
import string
import os
import ffmpeg
from PIL import Image, ImageDraw
import time  # Import the time module

# ... (Video settings remain the same)

# Audio settings (remain the same)

# ... (generate_visuals function remains the same)

def generate_audio():
    audio_length = int(SAMPLE_RATE * DURATION)
    sound_array = np.zeros(audio_length, dtype=np.int16)
    for _ in range(random.randint(5, 15)):
        freq = random.choice([220, 440, 880, 1760])
        duration = random.uniform(0.1, 1.0)
        start = random.randint(0, audio_length - int(SAMPLE_RATE * duration))
        wave = (np.sin(2 * np.pi * np.arange(int(SAMPLE_RATE * duration)) * freq / SAMPLE_RATE) * 32767).astype(np.int16)
        sound_array[start:start + len(wave)] += wave[:min(len(wave), len(sound_array) - start)]

    #  Write the raw audio file directly.  No need for pygame for this.
    np.savetxt("audio.raw", sound_array, fmt="%d") #corrected this line

    # The following code is no longer needed as we write the raw file directly.
    # pygame.mixer.init(frequency=SAMPLE_RATE, size=-BITS, channels=CHANNELS)
    # sound = pygame.sndarray.make_sound(sound_array)
    # pygame.mixer.Sound.play(sound)
    # pygame.time.delay(int(DURATION * 1000))  # This is problematic!
    # pygame.mixer.quit()


def create_video():
    os.system(f"ffmpeg -y -framerate {FRAME_RATE} -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p video.mp4")
    os.system(f"ffmpeg -y -f s16le -ar {SAMPLE_RATE} -ac {CHANNELS} -i audio.raw -c:a aac -b:a 128k audio.mp4")
    os.system("ffmpeg -y -i video.mp4 -i audio.mp4 -c:v copy -c:a aac final_video.mp4")

generate_visuals()
generate_audio()
create_video()
