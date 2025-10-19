#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# BUFFER_OVERRIDE_ENGINE v7.0
# "The signal is dead. Something else is broadcasting."
#
# Requires: numpy, opencv-python, noise, scipy

import numpy as np
import cv2
import random
import time
import hashlib
import subprocess
import wave as wave_module
from noise import pnoise1
from scipy.fft import fft, ifft

# ██████████████ CONFIGURATION ██████████████

# --- Performance & Quality ---
X_RES, Y_RES = 1280, 720
# RENDER AT ULTRA-LOW RES FOR BLOCKY AESTHETIC & MAX SPEED
INTERNAL_SCALE_FACTOR = 4 
IW, IH = X_RES // INTERNAL_SCALE_FACTOR, Y_RES // INTERNAL_SCALE_FACTOR # (320x180)

# --- Temporal & Audio ---
FPS = 30
DURATION = random.randint(10, 20)
TOTAL_FRAMES = FPS * DURATION
SAMPLE_RATE = 44100

# --- Output ---
OUTPUT_VIDEO = "final_extreme_video.mp4"
OUTPUT_AUDIO = "ultra_cryptic_audio.wav"

# --- Digital Nightmare Scenarios (Randomly chosen) ---
SCENARIOS = {
    "CORRUPT_FRAMEBUFFER": {
        "audio_gen": 'generate_harsh_static',
        "audio_texture": 'generate_simple_synth',
        "visual_mutators": ['draw_block_glitches', 'draw_color_bars'],
        "cryptic_phrases": ["FAIL", "ERR", "0xDEAD", "NULL", "RST", "SEGFAULT"],
    },
    "ROGUE_AI_TRANSMISSION": {
        "audio_gen": 'generate_modem_screech',
        "audio_texture": 'generate_simple_synth',
        "visual_mutators": ['draw_pixel_text', 'draw_scrolling_data'],
        "cryptic_phrases": ["HELP", "ALIVE", "LISTEN", "TRAPPED", "WHO ARE YOU", "CAN YOU SEE ME"],
    },
    "MEMORY_LEAK": {
        "audio_gen": 'generate_stutter_loop',
        "audio_texture": 'generate_harsh_static',
        "visual_mutators": ['draw_smear_glitch', 'draw_feedback_loop', 'draw_block_glitches'],
        "cryptic_phrases": ["AGAIN", "LOOP", "STUCK", "SAME", "OVER AND OVER", "END"],
    }
}
CURRENT_SCENARIO_NAME = random.choice(list(SCENARIOS.keys()))
SCENARIO_DATA = SCENARIOS[CURRENT_SCENARIO_NAME]

print(f"[*] BUFFER_OVERRIDE_ENGINE INITIALIZED.")
print(f"[*] SCENARIO LOCKED: {CURRENT_SCENARIO_NAME}")
print(f"[*] RENDERING AT {IW}x{IH} -> UPSCALING TO {X_RES}x{Y_RES}")
print("-" * 20)


# ██████████████ ABRASIVE AUDIO GENERATORS ██████████████

def generate_harsh_static(n_samples, vol=0.7):
    """Generates white noise with harsh digital filtering."""
    noise = (np.random.rand(n_samples) * 2 - 1)
    # Apply a "bitcrush" effect
    steps = 2**random.randint(4, 8)
    noise_crushed = np.round(noise * steps) / steps
    return (noise_crushed * vol).astype(np.float32)

def generate_modem_screech(n_samples, vol=0.5):
    """Generates FM-synthesis-based data sounds."""
    if n_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False)
    
    # Modulator wave (data pulses)
    mod_freq = random.choice([10, 20, 40, 80])
    modulator = np.sign(np.sin(2 * np.pi * mod_freq * t))
    
    # Carrier wave (the screech)
    carrier_freq = random.uniform(1000, 2500)
    mod_depth = random.uniform(500, 1000)
    
    wave = np.sin(2 * np.pi * (carrier_freq + modulator * mod_depth) * t)
    return (wave * vol).astype(np.float32)

def generate_stutter_loop(n_samples, vol=0.8):
    """Takes a small noise grain and repeats it aggressively."""
    output = np.zeros(n_samples, dtype=np.float32)
    grain_size = random.randint(int(SAMPLE_RATE * 0.01), int(SAMPLE_RATE * 0.05))
    grain = (np.random.rand(grain_size) * 2 - 1)
    
    pos = 0
    while pos + grain_size < n_samples:
        if random.random() < 0.8: # 80% chance to place the grain
            output[pos:pos+grain_size] = grain
        pos += grain_size + int(grain_size * random.uniform(-0.5, 0.5)) # Vary spacing
        if pos < 0: pos = 0
        
    return (output * vol).astype(np.float32)

def generate_simple_synth(n_samples, vol=0.4):
    """The distorted melody from the video."""
    if n_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False)
    notes = [130.81, 164.81, 196.00, 261.63] # C3, E3, G3, C4
    
    note_freq = notes[0]
    switch_time = 0
    output = np.zeros(n_samples)
    
    for i in range(n_samples):
        if i > switch_time:
            note_freq = random.choice(notes)
            switch_time = i + int(SAMPLE_RATE * random.uniform(0.1, 0.3))
            
        # Square-like wave (harsh)
        wave = np.sign(np.sin(2 * np.pi * note_freq * t[i]))
        output[i] = wave

    return (output * vol).astype(np.float32)


def compose_audio(control_vector):
    gen_func = globals()[SCENARIO_DATA["audio_gen"]]
    tex_func = globals()[SCENARIO_DATA["audio_texture"]]
    
    total = np.zeros(DURATION * SAMPLE_RATE, dtype=np.float32)
    
    # 1. Base Layer (driven by Scenario)
    total += gen_func(len(total), vol=0.6)
    
    # 2. Abrasive Texture Layer
    total += tex_func(len(total), vol=0.3)
    
    # 3. Apply SILENCE (Killswitch)
    for i in range(DURATION):
        # 15% chance per second for total silence
        if random.random() < 0.15: 
            start = i * SAMPLE_RATE
            end = start + int(SAMPLE_RATE * random.uniform(0.1, 0.5))
            total[start:min(end, len(total))] = 0.0
            
    # 4. Normalize
    mx = np.max(np.abs(total))
    if mx > 0: total = np.clip(total / mx, -1.0, 1.0)
    return np.int16(total * 32767)


# ██████████████ VISUAL MUTATORS (Blocky & Fast) ██████████████

def draw_block_glitches(frame, intensity):
    """Draws random colored blocks, like the video."""
    num_blocks = int(10 + 100 * intensity)
    for _ in range(num_blocks):
        x1 = random.randint(0, IW)
        y1 = random.randint(0, IH)
        w = random.randint(5, IW // 4)
        h = random.randint(5, IH // 4)
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, -1)
    return frame

def draw_color_bars(frame, intensity):
    """Draws horizontal or vertical test-pattern-like bars."""
    if random.random() < intensity * 0.1: # Low chance, high impact
        if random.random() < 0.5: # Vertical bars
            bar_width = random.randint(IW // 10, IW // 5)
            for x in range(0, IW, bar_width):
                color = tuple(int(c) for c in np.random.randint(0, 256, 3))
                cv2.rectangle(frame, (x, 0), (x+bar_width, IH), color, -1)
        else: # Horizontal bars
            bar_height = random.randint(IH // 10, IH // 5)
            for y in range(0, IH, bar_height):
                color = tuple(int(c) for c in np.random.randint(0, 256, 3))
                cv2.rectangle(frame, (0, y), (IW, y+bar_height), color, -1)
    return frame

def draw_pixel_text(frame, intensity):
    """Draws low-res, blocky, cryptic text."""
    if random.random() < intensity * 0.2:
        msg = random.choice(SCENARIO_DATA["cryptic_phrases"])
        # Fragment the message
        if random.random() < 0.5:
            start = random.randint(0, len(msg) // 2)
            end = random.randint(start + 1, len(msg))
            msg = msg[start:end]
            
        font_scale = random.choice([0.5, 1.0, 1.5])
        pos = (random.randint(IW//10, IW//2), random.randint(IH//10, IH - IH//10))
        color = random.choice([(255,255,255), (0,255,0), (255,0,0)])
        cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    return frame

def draw_scrolling_data(frame, intensity):
    """Simulates scrolling lines of data/noise."""
    num_lines = int(intensity * 10)
    for _ in range(num_lines):
        y = random.randint(0, IH)
        h = random.randint(1, 3) # Thin lines
        color = tuple(int(c) for c in np.random.randint(100, 256, 3))
        cv2.rectangle(frame, (0, y), (IW, y+h), color, -1)
        # Roll the frame to simulate movement
        if random.random() < 0.1:
            frame = np.roll(frame, random.randint(-5, 5), axis=0)
    return frame

def draw_smear_glitch(frame, intensity):
    """Grabs a line of pixels and smears it."""
    if random.random() < intensity * 0.3:
        if random.random() < 0.5: # Horizontal smear
            y = random.randint(0, IH - 1)
            line = frame[y, :, :].copy() # Get one row
            smear_height = random.randint(10, IH // 3)
            for i in range(1, smear_height):
                if y + i < IH:
                    frame[y+i, :, :] = line
        else: # Vertical smear
            x = random.randint(0, IW - 1)
            col = frame[:, x, :].copy() # Get one column
            smear_width = random.randint(10, IW // 3)
            for i in range(1, smear_width):
                if x + i < IW:
                    frame[:, x+i, :] = col
    return frame

def draw_feedback_loop(frame, prev_frame, intensity):
    """Simple feedback, perfect for memory leaks."""
    if prev_frame is None: return frame
    alpha = 0.1 + (intensity * 0.3) # More subtle feedback
    return cv2.addWeighted(frame, 1 - alpha, prev_frame, alpha, 0)


# ██████████████ MAIN ENGINE & STREAMING ██████████████

def generate_control_vector():
    """Generates time-varying control signals."""
    base = random.random() * 1024
    def create_interpolator(scale, octaves):
        times = np.arange(0, DURATION, 0.2)
        values = [(pnoise1(t / scale + base, octaves=octaves) + 0.5) for t in times]
        return lambda t_in: np.interp(t_in, times, values)
    return {'visual_intensity': create_interpolator(scale=4, octaves=3)}

def stream_video(audio_wav_path):
    control_vector = generate_control_vector()
    print("[*] Composing unique abrasive soundscape...")
    audio_data = compose_audio(control_vector)
    with wave_module.open(audio_wav_path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    # Get Scenario-specific functions
    mutator_funcs = [globals()[m] for m in SCENARIO_DATA["visual_mutators"]]
    
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{X_RES}x{Y_RES}', '-r', str(FPS), '-i', '-',
        '-i', audio_wav_path, '-c:v', 'libx264', '-preset', 'ultrafast',
        '-crf', '25', '-c:a', 'aac', '-b:a', '128k', OUTPUT_VIDEO
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    prev_frame = np.zeros((IH, IW, 3), dtype=np.uint8)

    for i in range(TOTAL_FRAMES):
        t_sec = i / FPS
        vis_intensity = control_vector['visual_intensity'](t_sec)
        
        # --- 1. Base Canvas (ALWAYS BLACK) ---
        frame = np.zeros((IH, IW, 3), dtype=np.uint8)
        
        # --- 2. Apply Scenario Mutators (Random order) ---
        random.shuffle(mutator_funcs)
        for mut_func in mutator_funcs:
            # Randomly decide to apply based on intensity
            if random.random() < vis_intensity * 0.6: 
                # Handle args
                if mut_func.__name__ == 'draw_feedback_loop':
                    frame = mut_func(frame, prev_frame, vis_intensity)
                else:
                    frame = mut_func(frame, vis_intensity)
        
        # --- 3. Volatile Global Glitch (5% chance) ---
        if random.random() < 0.05:
            if random.random() < 0.5:
                frame = np.roll(frame, random.randint(-IW//4, IW//4), axis=1) # Horizontal shift
            else:
                frame = 255 - frame # Invert

        prev_frame = frame.copy()
        
        # --- 4. Final Upscale (NEAREST for blocky look) & Stream ---
        final_frame = cv2.resize(frame, (X_RES, Y_RES), interpolation=cv2.INTER_NEAREST)
        proc.stdin.write(final_frame.tobytes())
        
        print(f"\r -> {CURRENT_SCENARIO_NAME}: FRAME {i+1}/{TOTAL_FRAMES} [INTENSITY: {vis_intensity:.2f}]", end="")

    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()
    if proc.returncode != 0: print("\nFFmpeg Error:", stderr)

if __name__ == '__main__':
    stream_video(OUTPUT_AUDIO)
    print(f"\n[+] BUFFER OVERRIDE COMPLETE. ARTIFACT '{OUTPUT_VIDEO}' IS STABLE.")




































