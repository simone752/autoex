#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# JIT_CHAOS_KERNEL v3.0
# "Faster, more intense."
#
# Requires: numpy, opencv-python, noise, scipy

import numpy as np
import cv2
import random
import time
import hashlib
import subprocess
import wave as wave_module
from noise import pnoise3, pnoise1  # <-- ADD pnoise1 HERE
from scipy.fft import fft, ifft

# ██████████████ CONFIGURATION ██████████████

# --- Performance & Quality ---
X_RES, Y_RES = 1280, 720
INTERNAL_SCALE_FACTOR = 2 # Render at 1/2 resolution (640x360) then upscale. Increase for more speed.
IW, IH = X_RES // INTERNAL_SCALE_FACTOR, Y_RES // INTERNAL_SCALE_FACTOR

# --- Temporal & Audio ---
FPS = 30
DURATION = random.randint(13, 22)
TOTAL_FRAMES = FPS * DURATION
SAMPLE_RATE = 44100

# --- Output ---
OUTPUT_VIDEO = "final_extreme_video.mp4"
OUTPUT_AUDIO = "ultra_cryptic_audio.wav"

# --- Cipher System ---
SEED_WORDS = ["KERNEL", "PANIC", "BUFFER", "OVERFLOW", "SEGFAULT", "ROOT", "DAEMON"]
MASTER_SEED = random.choice(SEED_WORDS)
HIDDEN_MESSAGE = f"TRANSMISSION_LOCKED_BY_{MASTER_SEED}"

CIPHER_MAP = {
    'A': ('flash', (0, 0, 255)),   'B': ('glyph', 'β'),      'C': ('scanlines', None),
    'D': ('glyph', 'Δ'),      'E': ('flash', (0, 255, 0)), 'F': ('feedback_spike', None),
    'G': ('glyph', 'Γ'),      'K': ('datamosh_trigger', None),'L': ('glyph', 'Λ'),
    'M': ('scanlines', None), 'N': ('glyph', 'И'),      'O': ('flash', (255, 255, 255)),
    'P': ('datamosh_trigger', None),'R': ('feedback_spike', None),'S': ('glyph', '§'),
    'T': ('glyph', '†'),      'U': ('glyph', 'μ'),      'V': ('scanlines', None),
    'W': ('flash', (255,0,255))
}

print(f"[*] JIT_CHAOS_KERNEL INITIALIZED.")
print(f"[*] SEED: {MASTER_SEED} | TARGET MESSAGE HASH: {hashlib.md5(HIDDEN_MESSAGE.encode()).hexdigest()}")
print(f"[*] RENDERING AT {IW}x{IH} -> UPSCALING TO {X_RES}x{Y_RES}")
print("-" * 20)


# ██████████████ AUDIO_CORE (Optimized Reverb) ██████████████

def convolution_reverb(data):
    """Applies a faster, dark reverb."""
    if len(data) == 0: return data
    # Shorter IR for massive speedup
    ir_len = SAMPLE_RATE // 5
    impulse_response = (np.random.randn(ir_len) * np.exp(-np.arange(ir_len) * 0.008))
    reverbed = np.convolve(data, impulse_response, 'same')
    
    orig_max = np.max(np.abs(data))
    rev_max = np.max(np.abs(reverbed))
    if rev_max > 1e-6:
        reverbed *= (orig_max / rev_max)
    return reverbed

# (Other audio functions from previous version can be copied here without change)
# For brevity, assuming generate_melody and granular_noise exist.
SCALES = {"phrygian": [0, 1, 3, 5, 7, 8, 10], "locrian": [0, 1, 3, 5, 6, 8, 10]}
def generate_melody(n_samples, vol=0.4):
    notes = [100 * (2**(random.choice(random.choice(list(SCALES.values())))/12)) for _ in range(8)]
    total_wave = np.zeros(n_samples, dtype=np.float32)
    note_len = n_samples // len(notes)
    pos = 0
    for note_freq in notes:
        t = np.linspace(0, note_len / SAMPLE_RATE, note_len, endpoint=False)
        wave = np.sin(2 * np.pi * note_freq * t) * np.linspace(1.0, 0.0, note_len)
        total_wave[pos:pos+note_len] = wave
        pos += note_len
    return np.clip(total_wave * 5.0, -1.0, 1.0) * vol

def granular_noise(n_samples, vol=0.6):
    output = np.zeros(n_samples, dtype=np.float32)
    for _ in range((n_samples // 200)):
        start = random.randint(0, n_samples - 500)
        grain = (np.random.rand(500) * 2 - 1) * np.hanning(500)
        output[start:start+500] += grain * 0.2
    return np.clip(output, -1.0, 1.0) * vol

def compose_audio(control_vector):
    total = granular_noise(DURATION * SAMPLE_RATE, vol=0.3)
    for i in range(DURATION):
        if control_vector['audio_intensity'](float(i)) > 0.7:
            start = i * SAMPLE_RATE
            length = int(SAMPLE_RATE * 2.0)
            end = min(len(total), start + length)
            total[start:end] += generate_melody(end - start, vol=0.5)
    total = convolution_reverb(total)
    mx = np.max(np.abs(total)); total = total / mx if mx > 0 else total
    return np.int16(total * 32767)


# ██████████████ VISUAL_CORE (Vectorized & Optimized) ██████████████

# Create coordinate grids once to be reused in the loop
x = np.arange(IW, dtype=np.float32)
y = np.arange(IH, dtype=np.float32)
xx, yy = np.meshgrid(x, y)
vectorized_pnoise3 = np.vectorize(pnoise3)

def fractal_noise_layer(t, scale=0.02, octaves=4):
    """Generates an evolving Perlin noise field using vectorized operations."""
    # Use a 3D noise function, with time as the 3rd dimension
    mat = vectorized_pnoise3(xx * scale, yy * scale, np.full((IH, IW), t), octaves=octaves)
    # Normalize to 0-255
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

def feedback_loop(frame, prev_frame, intensity):
    if prev_frame is None: return frame
    h, w = frame.shape[:2]
    angle = (intensity - 0.5) * 2.5
    scale = 1.01
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    transformed_prev = cv2.warpAffine(prev_frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(frame, 0.85, transformed_prev, 0.15, 0)

# (Other visual functions like datamosh_sim can be used as-is)
def datamosh_sim(frame, prev_frame, intensity):
    if prev_frame is None or random.random() > intensity: return frame
    output = frame.copy()
    num_blocks = int(intensity * 40)
    for _ in range(num_blocks):
        bh = random.randint(16, 64)
        bw = random.randint(16, 64)
        x, y = random.randint(0, IW-bw), random.randint(0, IH-bh)
        output[y:y+bh, x:x+bw] = prev_frame[y:y+bh, x:x+bw]
    return output

def apply_scanlines(frame, intensity):
    lines = frame.copy()
    for y in range(0, IH, random.randint(3,6)):
        cv2.line(lines, (0,y), (IW,y), (0,0,0), 1)
    return cv2.addWeighted(frame, 1, lines, intensity*0.5, 0)


# ██████████████ MAIN ENGINE & STREAMING ██████████████

def generate_control_vector():
    """Generates time-varying control signals."""
    base = random.random() * 1024
    def create_interpolator(scale, octaves):
        times = np.arange(0, DURATION, 0.2)
        values = [(pnoise1(t / scale + base, octaves=octaves) + 0.5) for t in times]
        return lambda t_in: np.interp(t_in, times, values)
    return {
        'visual_intensity': create_interpolator(scale=5, octaves=3),
        'audio_intensity': create_interpolator(scale=4, octaves=4),
        'color_shift': create_interpolator(scale=10, octaves=2),
    }

def stream_video(audio_wav_path):
    control_vector = generate_control_vector()
    audio_data = compose_audio(control_vector)
    with wave_module.open(audio_wav_path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    clue_schedule = {int((i+1)*0.9*(TOTAL_FRAMES/(len(MASTER_SEED)+1))): CIPHER_MAP.get(c) for i, c in enumerate(MASTER_SEED) if c in CIPHER_MAP}

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{X_RES}x{Y_RES}', '-r', str(FPS), '-i', '-',
        '-i', audio_wav_path, '-c:v', 'libx264',
        '-preset', 'ultrafast',  # <--- SPEED OPTIMIZATION
        '-crf', '22', '-c:a', 'aac', '-b:a', '128k', OUTPUT_VIDEO
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    prev_frame = np.zeros((IH, IW, 3), dtype=np.uint8)
    datamosh_active = False

    for i in range(TOTAL_FRAMES):
        t_sec = i / FPS
        vis_intensity = control_vector['visual_intensity'](t_sec)
        
        # --- Base Layer at LOW resolution ---
        noise_layer = fractal_noise_layer(t_sec * 2, scale=0.03, octaves=5)
        color_map_choice = random.choice([cv2.COLORMAP_INFERNO, cv2.COLORMAP_MAGMA, cv2.COLORMAP_PLASMA])
        frame = cv2.applyColorMap(noise_layer, color_map_choice)

        # --- Apply Mutators at LOW resolution ---
        frame = feedback_loop(frame, prev_frame, vis_intensity)
        if datamosh_active:
            frame = datamosh_sim(frame, prev_frame, vis_intensity)
            datamosh_active = False

        # --- Embed Cipher Clues at LOW resolution ---
        if i in clue_schedule:
            clue_type, clue_value = clue_schedule[i]
            if clue_type == 'flash':
                frame = cv2.addWeighted(frame, 0.3, np.full(frame.shape, clue_value, dtype=np.uint8), 0.7, 0)
            elif clue_type == 'glyph':
                cv2.putText(frame, clue_value, (IW//2 - 50, IH//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 10)
            elif clue_type == 'datamosh_trigger':
                datamosh_active = True
            elif clue_type == 'scanlines':
                frame = apply_scanlines(frame, vis_intensity)

        prev_frame = frame.copy()
        
        # --- Final Upscale and Stream ---
        # INTER_NEAREST is fast and gives the desired blocky, retro-glitch look
        final_frame = cv2.resize(frame, (X_RES, Y_RES), interpolation=cv2.INTER_NEAREST)
        proc.stdin.write(final_frame.tobytes())
        
        print(f"\r -> KERNEL PROCESSING: FRAME {i+1}/{TOTAL_FRAMES} [INTENSITY: {vis_intensity:.2f}]", end="")

    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()
    if proc.returncode != 0: print("\nFFmpeg Error:", stderr)

if __name__ == '__main__':
    stream_video(OUTPUT_AUDIO)
    print(f"\n[+] KERNEL PANIC CONTAINED. STREAM SAVED TO '{OUTPUT_VIDEO}'.")




































