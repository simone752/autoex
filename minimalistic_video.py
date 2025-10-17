#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MULTIVERSAL_CHAOS_ENGINE v4.0
# "Every run, a new nightmare."
#
# Requires: numpy, opencv-python, noise, scipy

import numpy as np
import cv2
import random
import time
import hashlib
import subprocess
import wave as wave_module
from noise import pnoise3, pnoise1
from scipy.fft import fft, ifft

# ██████████████ CONFIGURATION ██████████████

# --- Performance & Quality ---
X_RES, Y_RES = 1280, 720
INTERNAL_SCALE_FACTOR = 3 # Increased for more speed (426x240 -> 1280x720)
IW, IH = X_RES // INTERNAL_SCALE_FACTOR, Y_RES // INTERNAL_SCALE_FACTOR

# --- Temporal & Audio ---
FPS = 30
DURATION = random.randint(10, 20) # Shorter duration for speed
TOTAL_FRAMES = FPS * DURATION
SAMPLE_RATE = 44100

# --- Output ---
OUTPUT_VIDEO = "final_extreme_video.mp4"
OUTPUT_AUDIO = "ultra_cryptic_audio.wav"

# --- Cipher System (simplified for modularity) ---
SEED_WORDS = ["NEXUS", "SIGNAL", "DECAY", "CHAOS", "ENTITY", "RELAY", "SHIFT"]
MASTER_SEED = random.choice(SEED_WORDS)
HIDDEN_MESSAGE = f"THE_CHANNEL_IS_OPENED_BY_{MASTER_SEED}"

CIPHER_MAP = {
    'A': ('flash', (0, 0, 255)),   'C': ('scanlines', None),
    'D': ('glyph', 'Δ'),      'E': ('flash', (0, 255, 0)),
    'H': ('datamosh_trigger', None),'I': ('glyph', '§'),
    'L': ('feedback_spike', None), 'N': ('glyph', 'И'),
    'O': ('flash', (255, 255, 255)), 'S': ('scanlines', None),
    'T': ('datamosh_trigger', None),'U': ('feedback_spike', None),
    'Y': ('glyph', '¥'),
}

# ██████████████ AESTHETIC BLUEPRINTS ██████████████
# Each run selects one of these for extreme variety.

AESTHETICS = {
    "DATA_GLITCH": {
        "visual_generators": [cv2.COLORMAP_JET, cv2.COLORMAP_INFERNO],
        "mutators": ['feedback_loop', 'datamosh_sim', 'apply_scanlines'],
        "base_color_mode": 'HSV_SHIFT', # Wild color shifts
        "audio_generators": ['generate_melody', 'chaotic_oscillator'],
        "noise_scale": 1.5, # High noise frequency
    },
    "ABYSSAL_STATIC": {
        # cv2.COLORMAP_BONE is a valid, dark, low-saturation colormap
        "visual_generators": [cv2.COLORMAP_BONE, cv2.COLORMAP_MAGMA],
        "mutators": ['pixel_sort_glitch', 'feedback_loop', 'render_text_overlay'],
        "base_color_mode": 'MONOCHROME',
        "audio_generators": ['granular_noise', 'chaotic_oscillator'],
        "noise_scale": 0.5,
    },
    "GEOMETRIC_HORROR": {
        "visual_generators": [cv2.COLORMAP_TURBO, cv2.COLORMAP_VIRIDIS],
        "mutators": ['geometric_distortion', 'datamosh_sim', 'apply_scanlines'],
        "base_color_mode": 'COMPLEMENTARY', # Two high-contrast colors
        "audio_generators": ['generate_melody', 'generate_shepard_risset'],
        "noise_scale": 1.0, # Balanced noise frequency
    },
}
CURRENT_AESTHETIC = random.choice(list(AESTHETICS.keys()))
print(f"[*] AESTHETIC BLUEPRINT LOCKED: {CURRENT_AESTHETIC}")


# ██████████████ AUDIO_CORE (More Noise) ██████████████

def chaotic_oscillator(n, vol=0.5):
    """Generates a fast, chaotic, noise-like tone."""
    if n <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False)
    x = np.zeros(n); x[0] = random.random()
    r = 3.999 # Max chaos
    for i in range(1, n): x[i] = r * x[i-1] * (1 - x[i-1])
    base_freq = random.uniform(500, 2000)
    sig = np.sin(2 * np.pi * base_freq * t * (1 + x * 0.1))
    return (sig * vol).astype(np.float32)

def generate_shepard_risset(n_samples, vol=0.4):
    """Generates an endlessly rising/falling tone (Unsettling)."""
    if n_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False)
    num_octaves = 6; num_tones = 10; base_freq = 40.0
    rate = (2.0 ** (1.0 / 12.0)) ** (3.0 * random.choice([-1, 1]))
    final_wave = np.zeros(n_samples)
    for i in range(num_tones):
        initial_freq = base_freq * (2.0 ** (i * num_octaves / num_tones))
        freq = initial_freq * (rate ** t)
        freq = base_freq * (2.0 ** (np.log2(freq / base_freq) % num_octaves))
        amplitude = np.sin(np.pi * np.log2(freq / base_freq) / num_octaves) ** 2
        final_wave += np.sin(2 * np.pi * freq * t) * amplitude
    mx = np.max(np.abs(final_wave))
    if mx > 1e-6: final_wave /= mx
    return (final_wave * vol).astype(np.float32)

def generate_melody(n_samples, vol=0.4):
    # (Implementation remains the same: dark, clipped melody)
    SCALES = {"phrygian": [0, 1, 3, 5, 7, 8, 10]}
    notes = [100 * (2**(random.choice(SCALES["phrygian"])/12)) for _ in range(5)]
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
    # (Implementation remains the same: dense, crackling texture)
    output = np.zeros(n_samples, dtype=np.float32)
    for _ in range((n_samples // 500)):
        start = random.randint(0, n_samples - 500)
        grain = (np.random.rand(500) * 2 - 1) * np.hanning(500)
        output[start:start+500] += grain * random.uniform(0.1, 0.4)
    return np.clip(output, -1.0, 1.0) * vol

def compose_audio(control_vector):
    # Select generator based on aesthetic blueprint
    current_generators = AESTHETICS[CURRENT_AESTHETIC]["audio_generators"]
    generator_func_name = random.choice(current_generators)
    generator_func = globals()[generator_func_name]

    total = np.zeros(DURATION * SAMPLE_RATE, dtype=np.float32)
    
    # 1. Base Layer (driven by selected generator)
    for i in range(DURATION // 2):
        start = i * SAMPLE_RATE * 2
        length = SAMPLE_RATE * 2
        end = min(len(total), start + length)
        
        chunk = generator_func(end - start, vol=0.5)
        total[start:end] += chunk

    # 2. INTENSE NOISE MODULATION (Always present for extreme texture)
    noise_mod_scale = AESTHETICS[CURRENT_AESTHETIC]["noise_scale"]
    t_mod = np.linspace(0, DURATION * noise_mod_scale, len(total), endpoint=False)
    # High-frequency Perlin noise modulation
    noise_mod = np.array([pnoise1(t_val, octaves=5) for t_val in t_mod]) * 0.5
    total += noise_mod.astype(np.float32)
    
    # 3. Apply Reverb (Faster version)
    ir_len = SAMPLE_RATE // 5
    impulse_response = (np.random.randn(ir_len) * np.exp(-np.arange(ir_len) * 0.008))
    total = np.convolve(total, impulse_response, 'same')
    
    # 4. Normalize and Clip
    mx = np.max(np.abs(total))
    if mx > 0: total = np.clip(total / mx, -1.0, 1.0)
    return np.int16(total * 32767)


# ██████████████ VISUAL_CORE (Aesthetic-Driven) ██████████████

# Coordinate grids initialized once
x = np.arange(IW, dtype=np.float32)
y = np.arange(IH, dtype=np.float32)
xx, yy = np.meshgrid(x, y)
vectorized_pnoise3 = np.vectorize(pnoise3)

def fractal_noise_layer(t, scale=0.03, octaves=5):
    """Vectorized Perlin noise."""
    mat = vectorized_pnoise3(xx * scale, yy * scale, np.full((IH, IW), t), octaves=octaves)
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

def get_colors(tnorm):
    """Generates colors based on the current aesthetic mode."""
    mode = AESTHETICS[CURRENT_AESTHETIC]["base_color_mode"]
    
    if mode == 'HSV_SHIFT':
        # Wildly shifting colors based on time
        hue = int((tnorm * 360 + random.uniform(0, 360)) % 180)
        c1 = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        c2 = cv2.cvtColor(np.uint8([[[ (hue+90)%180, 255, 128]]]), cv2.COLOR_HSV2BGR)[0][0]
        return c1, c2
    elif mode == 'MONOCHROME':
        # Black, white, and random horror accent
        accent_color = random.choice([(0,0,255), (0,255,0), (255,0,0)])
        c1 = (0, 0, 0)
        c2 = accent_color if tnorm > 0.5 else (255, 255, 255)
        return c1, c2
    elif mode == 'COMPLEMENTARY':
        # High contrast, static, unsettling colors
        hue = random.randint(0, 180)
        c1 = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
        c2 = cv2.cvtColor(np.uint8([[[ (hue+90)%180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        return c1, c2

# --- Mutator Functions ---
def feedback_loop(frame, prev_frame, intensity):
    if prev_frame is None: return frame
    h, w = frame.shape[:2]
    angle = (intensity - 0.5) * 3.0 # More dramatic
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.02) # Slight zoom
    transformed_prev = cv2.warpAffine(prev_frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(frame, 0.8, transformed_prev, 0.2, 0)

def datamosh_sim(frame, prev_frame, intensity):
    if prev_frame is None or random.random() > intensity: return frame
    output = frame.copy()
    num_blocks = int(intensity * 60) # More blocks
    for _ in range(num_blocks):
        bh = random.randint(8, 64)
        bw = random.randint(8, 64)
        x, y = random.randint(0, IW-bw), random.randint(0, IH-bh)
        output[y:y+bh, x:x+bw] = prev_frame[y:y+bh, x:x+bw]
    return output

def apply_scanlines(frame, intensity):
    lines = frame.copy()
    for y in range(0, IH, 2):
        cv2.line(lines, (0,y), (IW,y), (0,0,0), 1)
    return cv2.addWeighted(frame, 1, lines, intensity*0.3, 0)

def pixel_sort_glitch(frame, intensity):
    """Sorts pixel rows/columns based on brightness to create jarring tears."""
    if random.random() < intensity * 0.5:
        h, w, _ = frame.shape
        start_row = random.randint(0, h-1)
        end_row = min(h, start_row + random.randint(10, 50))
        for y in range(start_row, end_row):
            row = frame[y, :, :]
            # Sort the row based on the brightness of the blue channel
            sort_indices = np.argsort(row[:, 0]) 
            frame[y, :, :] = row[sort_indices]
    return frame

def render_text_overlay(frame, intensity):
    """Overlays random, opaque, blocky text fragments."""
    if random.random() < intensity:
        text = random.choice(["EAT", "SEE", "HURT", "FAIL", "LOOP", "CODE"])
        scale = random.uniform(0.5, 3.0)
        thickness = random.randint(1, 5)
        x = random.randint(0, IW - 50)
        y = random.randint(50, IH)
        color = tuple(int(c) for c in np.random.randint(100, 256, 3))
        alpha = 0.5 + intensity * 0.5
        
        overlay = frame.copy()
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    return frame

def geometric_distortion(frame, intensity):
    """Applies a high-frequency sine-wave distortion/warp."""
    h, w = frame.shape[:2]
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    # Distortion magnitude is tied to intensity
    mag = intensity * 15 
    freq = intensity * 0.5 + 0.1
    
    for y in range(h):
        for x in range(w):
            map_x[y, x] = x + mag * np.sin(x * freq + y * freq * 0.5)
            map_y[y, x] = y + mag * np.cos(y * freq + x * freq * 0.5)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ██████████████ MAIN ENGINE & STREAMING ██████████████

def generate_control_vector():
    """Generates time-varying control signals."""
    base = random.random() * 1024
    def create_interpolator(scale, octaves):
        times = np.arange(0, DURATION, 0.2)
        values = [(pnoise1(t / scale + base, octaves=octaves) + 0.5) for t in times]
        return lambda t_in: np.interp(t_in, times, values)
    return {
        'visual_intensity': create_interpolator(scale=4, octaves=3),
        'audio_intensity': create_interpolator(scale=3, octaves=4),
    }

def stream_video(audio_wav_path):
    control_vector = generate_control_vector()
    print("[*] Composing unique auditory stream...")
    audio_data = compose_audio(control_vector)
    with wave_module.open(audio_wav_path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    clue_schedule = {int((i+1)*0.9*(TOTAL_FRAMES/(len(MASTER_SEED)+1))): CIPHER_MAP.get(c) for i, c in enumerate(MASTER_SEED) if c in CIPHER_MAP}

    # FFMPEG Command: using 'ultrafast' for maximum speed
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{X_RES}x{Y_RES}', '-r', str(FPS), '-i', '-',
        '-i', audio_wav_path, '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '25', '-c:a', 'aac', '-b:a', '128k', OUTPUT_VIDEO
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    prev_frame = np.zeros((IH, IW, 3), dtype=np.uint8)
    datamosh_active = False
    mutator_funcs = [globals()[m] for m in AESTHETICS[CURRENT_AESTHETIC]["mutators"]]

    for i in range(TOTAL_FRAMES):
        t_sec = i / FPS
        vis_intensity = control_vector['visual_intensity'](t_sec)
        
        # --- Base Layer at LOW resolution (Fastest part) ---
        noise_layer = fractal_noise_layer(t_sec * 2, scale=0.03, octaves=5)
        c1, c2 = get_colors(i / TOTAL_FRAMES)
        
        frame = np.zeros((IH, IW, 3), dtype=np.uint8)
        frame[noise_layer > 128] = (int(c1[0]), int(c1[1]), int(c1[2]))
        frame[noise_layer <= 128] = (int(c2[0]), int(c2[1]), int(c2[2]))

        # --- Apply Mutators (Randomly ordered and intensity-driven) ---
        random.shuffle(mutator_funcs) # Randomize order for unpredictable look
        
        for mut_func in mutator_funcs:
            if random.random() < 0.2 + vis_intensity * 0.7:
                # Mutators must handle different argument counts
                if mut_func.__name__ == 'feedback_loop' or mut_func.__name__ == 'datamosh_sim':
                    frame = mut_func(frame, prev_frame, vis_intensity)
                else:
                    frame = mut_func(frame, vis_intensity)
        
        if datamosh_active: # Single frame trigger
            frame = datamosh_sim(frame, prev_frame, 1.0)
            datamosh_active = False

        # --- Embed Cipher Clues at LOW resolution ---
        if i in clue_schedule:
            clue_type, clue_value = clue_schedule[i]
            if clue_type == 'flash':
                frame = cv2.addWeighted(frame, 0.3, np.full(frame.shape, clue_value, dtype=np.uint8), 0.7, 0)
            elif clue_type == 'glyph':
                cv2.putText(frame, clue_value, (IW//2 - 20, IH//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
            elif clue_type == 'datamosh_trigger':
                datamosh_active = True
            elif clue_type == 'scanlines':
                frame = apply_scanlines(frame, vis_intensity)

        prev_frame = frame.copy()
        
        # --- Final Upscale and Stream (FASTEST resize: INTER_NEAREST) ---
        final_frame = cv2.resize(frame, (X_RES, Y_RES), interpolation=cv2.INTER_NEAREST)
        proc.stdin.write(final_frame.tobytes())
        
        print(f"\r -> KERNEL: [{CURRENT_AESTHETIC}] FRAME {i+1}/{TOTAL_FRAMES} [INTENSITY: {vis_intensity:.2f}]", end="")

    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()
    if proc.returncode != 0: print("\nFFmpeg Error:", stderr)

if __name__ == '__main__':
    stream_video(OUTPUT_AUDIO)
    print(f"\n[+] MULTIVERSAL FRAGMENT GENERATED. '{OUTPUT_VIDEO}' IS READY.")




































