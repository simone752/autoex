import numpy as np
import wave
import random
import subprocess
from PIL import Image, ImageDraw

def generate_visuals(frame_count, width=640, height=480):
    print("Generating extreme visuals...")
    for i in range(frame_count):
        img = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Sparse but bold elements
        if random.random() > 0.3:
            shape_type = random.choice(['line', 'rectangle', 'circle'])
            x0, y0, x1, y1 = sorted([random.randint(0, width) for _ in range(2)]), sorted([random.randint(0, height) for _ in range(2)])
            color = tuple(random.choices(range(256), k=3))
            
            if shape_type == 'line':
                draw.line((x0[0], y0[0], x1[0], y1[0]), fill=color, width=random.randint(1, 5))
            elif shape_type == 'rectangle':
                draw.rectangle((x0[0], y0[0], x1[0], y1[0]), outline=color, width=random.randint(1, 5))
            elif shape_type == 'circle':
                draw.ellipse((x0[0], y0[0], x1[0], y1[0]), outline=color, width=random.randint(1, 5))
        
        img.save(f'frame_{i:03d}.png')

def generate_audio(filename, duration=5, sample_rate=44100):
    print("Generating unpredictable audio...")
    audio = np.zeros(duration * sample_rate, dtype=np.int16)
    
    for i in range(0, len(audio), sample_rate // 10):
        if random.random() > 0.6:
            frequency = random.choice([220, 440, 880, 1760])
            pulse_duration = random.randint(1000, 5000)
            t = np.linspace(0, pulse_duration / sample_rate, pulse_duration, endpoint=False)
            waveform = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
            audio[i:i + pulse_duration] = waveform[:len(audio) - i]
        elif random.random() > 0.8:
            audio[i:i + 5000] = np.random.randint(-5000, 5000, 5000, dtype=np.int16)
        
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio.tobytes())

def create_video(output_path, frame_count=100):
    generate_visuals(frame_count)
    generate_audio("temp_audio.wav", duration=frame_count // 24)
    
    subprocess.run("ffmpeg -framerate 24 -i frame_%03d.png -c:v libx264 temp_video.mp4 -y", shell=True, check=True)
    subprocess.run("ffmpeg -i temp_video.mp4 -i temp_audio.wav -c:v copy -c:a aac " + output_path + " -y", shell=True, check=True)
    
    print("Video generation complete!")

if __name__ == "__main__":
    create_video("extreme_video.mp4")
