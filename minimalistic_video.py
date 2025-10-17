#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ABYSSAL_CIPHER_ENGINE v2.0
# "We have such sights to show you."
#
# Requires: numpy, opencv-python, noise, scipy

import numpy as np
import cv2
import random
import time
import inspect
import subprocess
import wave as wave_module
import hashlib
from noise import pnoise1
from scipy.fft import fft, ifft

# ██████████████ CONFIGURATION ██████████████

# Canvas & Time
X_RES, Y_RES = 1280, 720
FPS = 30
DURATION = random.randint(15, 25) # Unpredictable length
TOTAL_FRAMES = FPS * DURATION
SAMPLE_RATE = 44100

# Output Artifacts
OUTPUT_VIDEO = "final_extreme_video.mp4"
OUTPUT_AUDIO = "ultra_cryptic_audio.wav"

# ██████████████ CIPHER & CRYPTOGRAPHY ██████████████

# The system generates a seed and hides it in the video.
# The audience must find the seed to unlock the 'hidden_message'.
# Example: If the seed is "VOID", clues for V, O, I, D will appear.
SEED_WORDS = ["VOID", "ABYSS", "ECHO", "STATIC", "GLITCH", "NULL", "ENTITY", "SIGNAL"]
MASTER_SEED = random.choice(SEED_WORDS)
HIDDEN_MESSAGE = "THE_MACHINE_IS_WATCHING"

# Each character in the seed maps to a specific visual or audio event.
CIPHER_MAP = {
    'A': ('flash', (0, 0, 255)),   # Red Flash
    'B': ('glyph', 'Σ'),
    'C': ('datamosh_trigger', None),
    'D': ('glyph', '∞'),
    'E': ('flash', (0, 255, 0)),   # Green Flash
    'G': ('audio_event', 'morse_short'),
    'H': ('glyph', '⧫'),
    'I': ('feedback_spike', None),
    'L': ('audio_event', 'morse_long'),
    'N': ('glyph', '∅'),
    'O': ('flash', (255, 255, 255)), # White Flash
    'S': ('audio_event', 'morse_short'),
    'T': ('audio_event', 'morse_long'),
    'U': ('glyph', '██'),
    'V': ('feedback_spike', None),
    'Y': ('datamosh_trigger', None),
}

print(f"[*] ABYSSAL_CIPHER_ENGINE INITIALIZED.")
print(f"[*] GENERATING ARTIFACT WITH SEED: {MASTER_SEED}")
print(f"[*] MD5 HASH OF HIDDEN MESSAGE: {hashlib.md5(HIDDEN_MESSAGE.encode()).hexdigest()}")
print("-" * 20)

# ██████████████ AUDIO_CORE ██████████████

SCALES = {
    "phrygian": [0, 1, 3, 5, 7, 8, 10],  # Dark, Spanish scale
    "locrian": [0, 1, 3, 5, 6, 8, 10],   # Highly dissonant
}

def generate_melody(n_samples, vol=0.4):
    """Generates a heavily distorted melody from a dark scale."""
    notes = []
    scale_intervals = random.choice(list(SCALES.values()))
    base_note = random.uniform(80, 150)
    for _ in range(random.randint(4, 8)):
        semitone = random.choice(scale_intervals)
        notes.append(base_note * (2**(semitone/12)))

    total_wave = np.zeros(n_samples, dtype=np.float32)
    note_len = n_samples // len(notes)
    
    pos = 0
    for note_freq in notes:
        t = np.linspace(0, note_len / SAMPLE_RATE, note_len, endpoint=False)
        wave = np.sin(2 * np.pi * note_freq * t)
        # Apply envelope
        wave *= np.linspace(1.0, 0.0, note_len)
        total_wave[pos:pos+note_len] = wave
        pos += note_len

    # Heavy distortion (clipping)
    return np.clip(total_wave * 5.0, -1.0, 1.0) * vol

def granular_noise(n_samples, vol=0.6):
    """Creates a dense, crackling texture from tiny sound grains."""
    output = np.zeros(n_samples, dtype=np.float32)
    grain_size = random.randint(100, 500)
    num_grains = (n_samples // grain_size) * 4

    for _ in range(num_grains):
        start = random.randint(0, n_samples - grain_size)
        grain = (np.random.rand(grain_size) * 2 - 1) * np.hanning(grain_size)
        output[start:start+grain_size] += grain * random.uniform(0.1, 0.5)

    return np.clip(output, -1.0, 1.0) * vol

def convolution_reverb(data):
    """Applies a simple, dark reverb to a sound."""
    if len(data) == 0: return data
    ir_len = SAMPLE_RATE // 2
    impulse_response = (np.random.randn(ir_len) * np.exp(-np.arange(ir_len) * 0.005))
    reverbed = np.convolve(data, impulse_response, 'same')
    
    orig_max = np.max(np.abs(data))
    rev_max = np.max(np.abs(reverbed))
    if rev_max > 1e-6:
        reverbed *= (orig_max / rev_max) # Normalize to original peak
    return reverbed

def compose_audio(control_vector):
    """Composes the final, terrifying audio track."""
    total = np.zeros(DURATION * SAMPLE_RATE, dtype=np.float32)
    
    # Base layer of granular texture
    total += granular_noise(len(total), vol=0.3)
    
    # Add distorted melodies at high-intensity moments
    for i in range(DURATION):
        # Check intensity once per second
        time_sec = float(i)
        intensity = control_vector['audio_intensity'](time_sec)
        if intensity > 0.7 and random.random() < 0.5:
            start_sample = i * SAMPLE_RATE
            length = int(SAMPLE_RATE * random.uniform(1.5, 3.0))
            end_sample = min(len(total), start_sample + length)
            
            chunk = generate_melody(end_sample - start_sample, vol=intensity * 0.6)
            total[start_sample:end_sample] += chunk

    # Apply reverb to the whole track
    total = convolution_reverb(total)
    
    # Normalize
    mx = np.max(np.abs(total))
    if mx > 0: total /= mx
    return np.int16(total * 32767)

# ██████████████ VISUAL_CORE ██████████████

def fractal_noise_layer(shape, t):
    """Generates an evolving Perlin noise field."""
    h, w = shape
    mat = np.zeros(shape, dtype=np.float32)
    scale = 0.01  # Lower scale = larger features
    for y in range(h):
        for x in range(w):
            # pnoise3 gives us noise that evolves over time (t)
            mat[y, x] = pnoise1(x * scale + t * 5, octaves=4, persistence=0.5, lacunarity=2.0)
    
    # Normalize to 0-255
    cv2.normalize(mat, mat, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

def feedback_loop(frame, prev_frame, intensity):
    """Creates video feedback trails by blending with a transformed previous frame."""
    if prev_frame is None: return frame
    
    # Transformation: scale, rotate, shift
    h, w = frame.shape[:2]
    angle = (intensity - 0.5) * 2.0 # -1 to 1 degree
    scale = 1.005
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0, 2] += (intensity - 0.5) * 4 # x-shift
    M[1, 2] += (intensity - 0.5) * 4 # y-shift
    
    transformed_prev = cv2.warpAffine(prev_frame, M, (w, h))
    alpha = 0.9 - (intensity * 0.2)
    return cv2.addWeighted(frame, alpha, transformed_prev, 1 - alpha, 0)

def datamosh_sim(frame, prev_frame, intensity):
    """Simulates compression glitches by copying blocks from the last frame."""
    if prev_frame is None or random.random() > intensity: return frame
    
    output = frame.copy()
    num_blocks = int(intensity * 50)
    block_size = 64
    
    for _ in range(num_blocks):
        x = random.randrange(0, X_RES, block_size)
        y = random.randrange(0, Y_RES, block_size)
        
        block = prev_frame[y:y+block_size, x:x+block_size]
        output[y:y+block_size, x:x+block_size] = block
        
    return output

# ██████████████ MAIN ENGINE & STREAMING ██████████████

def generate_control_vector():
    """Generates time-varying control signals using Perlin noise."""
    base = random.randint(0, 1024)
    def create_interpolator(scale, octaves):
        points = []
        for i in range(DURATION * 5): # 5 points per second
            t = i / 5.0
            noise_val = pnoise1(t / scale + base, octaves=octaves)
            points.append((t, (noise_val + 0.5))) # Normalize roughly to 0-1
        
        times, values = zip(*points)
        # Linear interpolation is fine and fast
        return lambda t_in: np.interp(t_in, times, values)

    return {
        'visual_intensity': create_interpolator(scale=5, octaves=3),
        'audio_intensity': create_interpolator(scale=4, octaves=4),
        'color_shift': create_interpolator(scale=10, octaves=2),
    }

def stream_video(audio_wav_path):
    """Generates frames, embeds clues, and streams to FFmpeg."""
    control_vector = generate_control_vector()
    
    # Compose audio first to allow audio-reactive visuals (if desired)
    audio_data = compose_audio(control_vector)
    with wave_module.open(audio_wav_path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    # Schedule cipher clues throughout the video
    clue_schedule = {}
    for i, char in enumerate(MASTER_SEED):
        frame_time = int((i + 1) * 0.8 * (TOTAL_FRAMES / (len(MASTER_SEED)+1)))
        clue_schedule[frame_time] = CIPHER_MAP.get(char)
    
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{X_RES}x{Y_RES}', '-r', str(FPS), '-i', '-',
        '-i', audio_wav_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k', OUTPUT_VIDEO
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    prev_frame = None
    datamosh_active = False

    for i in range(TOTAL_FRAMES):
        tnorm = i / TOTAL_FRAMES
        
        # Get control values for this frame
        vis_intensity = control_vector['visual_intensity'](tnorm * DURATION)
        color_shift = control_vector['color_shift'](tnorm * DURATION)
        
        # --- Base Layer ---
        noise_fg = fractal_noise_layer((Y_RES, X_RES), tnorm)
        noise_bg = fractal_noise_layer((Y_RES, X_RES), tnorm + 100)
        
        # Create a dynamic, jarring color palette
        hue = int(color_shift * 180)
        c1 = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        c2 = cv2.cvtColor(np.uint8([[[ (hue+90)%180, 255, 128]]]), cv2.COLOR_HSV2BGR)[0][0]
        
        frame = np.zeros((Y_RES, X_RES, 3), dtype=np.uint8)
        frame[noise_bg > 128] = (int(c1[0]), int(c1[1]), int(c1[2]))
        frame[noise_fg > 128] = (int(c2[0]), int(c2[1]), int(c2[2]))

        # --- Apply Mutators ---
        frame = feedback_loop(frame, prev_frame, vis_intensity)
        if datamosh_active:
            frame = datamosh_sim(frame, prev_frame, vis_intensity)
            datamosh_active = False # One-shot effect

        # --- Embed Cipher Clues ---
        if i in clue_schedule:
            clue_type, clue_value = clue_schedule[i]
            if clue_type == 'flash':
                frame = cv2.addWeighted(frame, 0.3, np.full(frame.shape, clue_value, dtype=np.uint8), 0.7, 0)
            elif clue_type == 'glyph':
                cv2.putText(frame, clue_value, (X_RES//2, Y_RES//2), cv2.FONT_HERSHEY_SIMPLEX, 10, (255,255,255), 15)
            elif clue_type == 'datamosh_trigger':
                datamosh_active = True
            # (Audio clues would be embedded in compose_audio, not implemented here for simplicity)
        
        proc.stdin.write(frame.tobytes())
        prev_frame = frame.copy()
        
        # Progress bar
        print(f"\r -> ENCODING FRAGMENT: [{'█' * int(tnorm*20)}{' ' * (19-int(tnorm*20))}]", end="")

    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()
    if proc.returncode != 0: print("FFmpeg Error:", stderr)

if __name__ == '__main__':
    stream_video(OUTPUT_AUDIO)
    print(f"\n[+] FRAGMENTATION COMPLETE. ARTIFACT '{OUTPUT_VIDEO}' IS STABLE.")
    print("[!] THE CIPHER IS SET. THE SEED IS HIDDEN. LET THEM WATCH.")




































