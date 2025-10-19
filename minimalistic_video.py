#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# NARRATIVE_CHAOS_ENGINE v6.0
# "Four stories. Four nightmares. All of them are for you."
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
INTERNAL_SCALE_FACTOR = 3 # Render at 426x240 for MAX SPEED
IW, IH = X_RES // INTERNAL_SCALE_FACTOR, Y_RES // INTERNAL_SCALE_FACTOR

# --- Temporal & Audio ---
FPS = 30
DURATION = random.randint(12, 22)
TOTAL_FRAMES = FPS * DURATION
SAMPLE_RATE = 44100

# --- Output ---
OUTPUT_VIDEO = "final_extreme_video.mp4"
OUTPUT_AUDIO = "ultra_cryptic_audio.wav"

# --- Cipher System (Integrated into Scenarios) ---
# A seed is still generated, but the messages are now thematic.
SEED_WORDS = ["EYE", "GRID", "FLESH", "SIGIL", "VOID", "STATIC"]
MASTER_SEED = random.choice(SEED_WORDS)
HIDDEN_MESSAGE = f"THE_BARRIER_IS_BROKEN_BY_{MASTER_SEED}"

# ██████████████ SCENARIO BLUEPRINTS (The Core) ██████████████

SCENARIOS = {
    "THE_WATCHER": {
        "visual_gen": 'fractal_noise', # Static
        "palette": [(0, 0, 0), (255, 255, 255), (0, 0, 200)], # Black, White, Blood Red
        "audio_gen": 'low_rumble',
        "audio_texture": 'wet_clicks',
        "core_mutator": 'draw_eye_motif', # Its "character"
        "cryptic_phrases": ["I SEE YOU", "WATCHING", "AWAKE", "FOUND YOU", f"SEED:{MASTER_SEED}"],
    },
    "MACHINE_GOD": {
        "visual_gen": 'geometric_stripes', # Grids
        "palette": [(10, 0, 0), (0, 255, 100), (0, 200, 255)], # Black, Toxic Green, Electric Blue
        "audio_gen": 'chaotic_oscillator',
        "audio_texture": 'digital_screech',
        "core_mutator": 'draw_grid_overlay', # Its "character"
        "cryptic_phrases": ["OBEY", "ASSIMILATE", "LOGIC", "ERROR", "NULL", f"KEY:{MASTER_SEED}"],
    },
    "FLESH_SINGULARITY": {
        "visual_gen": 'fractal_noise', # Cellular
        "palette": [(40, 0, 10), (200, 150, 220), (100, 100, 255)], # Dark Red, Sickly Pink, Fleshy Red
        "audio_gen": 'low_rumble',
        "audio_texture": 'wet_clicks',
        "core_mutator": 'warp_distortion', # Its "character" (breathing/growing)
        "cryptic_phrases": ["GROW", "BECOME", "ONE", "HUNGER", "JOIN", f"WORD:{MASTER_SEED}"],
    },
    "THE_RITUAL": {
        "visual_gen": 'geometric_stripes', # Sigils
        "palette": [(20, 10, 20), (200, 180, 0), (255, 255, 255)], # Deep Purple, Gold, White
        "audio_gen": 'generate_shepard_risset',
        "audio_texture": 'digital_screech',
        "core_mutator": 'draw_sigil_motif', # Its "character"
        "cryptic_phrases": ["SUMMON", "IT COMES", "THE CIRCLE", "OPEN", f"PASS:{MASTER_SEED}"],
    }
}
CURRENT_SCENARIO_NAME = random.choice(list(SCENARIOS.keys()))
SCENARIO_DATA = SCENARIOS[CURRENT_SCENARIO_NAME]

print(f"[*] NARRATIVE_CHAOS_ENGINE INITIALIZED.")
print(f"[*] SCENARIO LOCKED: {CURRENT_SCENARIO_NAME}")
print(f"[*] HASH OF HIDDEN MESSAGE: {hashlib.md5(HIDDEN_MESSAGE.encode()).hexdigest()}")
print("-" * 20)


# ██████████████ ABRASIVE AUDIO GENERATORS ██████████████

def low_rumble(n_samples, vol=0.6):
    """Generates a deep, vibrating bass rumble."""
    if n_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False)
    # Slow sine wave
    base_freq = random.uniform(30, 50)
    wave = np.sin(2 * np.pi * base_freq * t)
    # Modulate with slow noise
    mod_t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False) * 0.5
    mod_wave = np.array([pnoise1(t_val, octaves=4) for t_val in mod_t]) * 0.5
    return (wave * (0.5 + mod_wave) * vol).astype(np.float32)

def wet_clicks(n_samples, vol=0.5):
    """Simulates wet, clicking, organic sounds."""
    output = np.zeros(n_samples, dtype=np.float32)
    num_clicks = int(n_samples / (SAMPLE_RATE * 0.1)) # 10 clicks per second
    for _ in range(num_clicks):
        start = random.randint(0, n_samples - 1000)
        # Short grain with a sharp attack
        grain = (np.random.rand(1000) * 2 - 1) * np.exp(-np.arange(1000) * 0.01)
        output[start:start+1000] += grain * random.uniform(0.1, 0.4)
    return (output * vol).astype(np.float32)

def digital_screech(n_samples, vol=0.4):
    """A harsh, high-frequency digital noise burst."""
    if n_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples, endpoint=False)
    # High-frequency, fast-modulating noise
    r = 3.999
    x = np.zeros(n_samples); x[0] = random.random()
    for i in range(1, n_samples): x[i] = r * x[i-1] * (1 - x[i-1])
    base_freq = random.uniform(4000, 8000)
    sig = np.sin(2 * np.pi * base_freq * t * (1 + x * 0.5))
    return (sig * vol).astype(np.float32)

def audio_spike(vol=1.0):
    """Generates a brief, piercing burst of noise."""
    duration_samples = int(SAMPLE_RATE * random.uniform(0.05, 0.3))
    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False)
    freq = np.exp(t * random.uniform(1, 5)) * random.uniform(500, 5000)
    wave = np.sin(2 * np.pi * freq * t) * np.hanning(duration_samples)
    return (wave * vol).astype(np.float32)

# (Other audio gens from previous version)
def chaotic_oscillator(n, vol=0.5):
    if n <= 0: return np.array([], dtype=np.float32); t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False); x = np.zeros(n); x[0] = random.random(); r = 3.999; 
    for i in range(1, n): x[i] = r * x[i-1] * (1 - x[i-1]); base_freq = random.uniform(500, 2000); sig = np.sin(2 * np.pi * base_freq * t * (1 + x * 0.1)); return (sig * vol).astype(np.float32)
def generate_shepard_risset(n, vol=0.4):
    if n <= 0: return np.array([], dtype=np.float32); t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False); num_octaves = 6; num_tones = 10; base_freq = 40.0; rate = (2.0 ** (1.0 / 12.0)) ** (3.0 * random.choice([-1, 1])); final_wave = np.zeros(n); 
    for i in range(num_tones): initial_freq = base_freq * (2.0 ** (i * num_octaves / num_tones)); freq = initial_freq * (rate ** t); freq = base_freq * (2.0 ** (np.log2(freq / base_freq) % num_octaves)); amplitude = np.sin(np.pi * np.log2(freq / base_freq) / num_octaves) ** 2; final_wave += np.sin(2 * np.pi * freq * t) * amplitude; 
    mx = np.max(np.abs(final_wave)); 
    if mx > 1e-6: final_wave /= mx; 
    return (final_wave * vol).astype(np.float32)

def compose_audio(control_vector):
    # 1. Get Scenario-specific generators
    gen_func = globals()[SCENARIO_DATA["audio_gen"]]
    tex_func = globals()[SCENARIO_DATA["audio_texture"]]
    
    total = np.zeros(DURATION * SAMPLE_RATE, dtype=np.float32)
    
    # 2. Base Layer (driven by selected generator)
    total += gen_func(len(total), vol=0.6)
    
    # 3. Abrasive Texture Layer
    total += tex_func(len(total), vol=0.4)

    # 4. Volatile Auditory Spikes (10% chance every second)
    for i in range(DURATION):
        if random.random() < 0.10:
            spike = audio_spike(vol=random.uniform(0.7, 1.0))
            start_idx = int(i * SAMPLE_RATE + random.uniform(0, SAMPLE_RATE - len(spike)))
            start_idx = max(0, start_idx)
            end_idx = min(len(total), start_idx + len(spike))
            if end_idx - start_idx == len(spike): # Only add if it fits
                total[start_idx:end_idx] += spike
    
    # 5. Apply Reverb & Normalize
    ir_len = SAMPLE_RATE // 5 # Fast reverb
    impulse_response = (np.random.randn(ir_len) * np.exp(-np.arange(ir_len) * 0.008))
    total = np.convolve(total, impulse_response, 'same')
    
    mx = np.max(np.abs(total))
    if mx > 0: total = np.clip(total / mx, -1.0, 1.0)
    return np.int16(total * 32767)


# ██████████████ VISUAL GENERATORS & MUTATORS ██████████████

# Coordinate grids initialized once
x = np.arange(IW, dtype=np.float32)
y = np.arange(IH, dtype=np.float32)
xx, yy = np.meshgrid(x, y)
vectorized_pnoise3 = np.vectorize(pnoise3)

def fractal_noise(t, scale=0.03, octaves=5):
    """Vectorized Perlin noise."""
    mat = vectorized_pnoise3(xx * scale, yy * scale, np.full((IH, IW), t), octaves=octaves)
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

def geometric_stripes(t):
    """Generates sharp, shifting, chaotic lines."""
    mat = np.sin((xx + yy) * 0.1 + t * 5)
    noise_2d = vectorized_pnoise3(xx * 0.05, yy * 0.05, 0, octaves=3) * 0.5
    mat += noise_2d
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

def apply_themed_palette(frame_mono, palette):
    """Applies a 2 or 3-color palette to a mono frame. FAST."""
    frame_bgr = np.full((IH, IW, 3), palette[0], dtype=np.uint8)
    frame_bgr[frame_mono > 100] = palette[1]
    if len(palette) > 2:
        frame_bgr[frame_mono > 200] = palette[2]
    return frame_bgr

# --- Scenario-Specific "Character" Mutators ---

def draw_eye_motif(frame, intensity):
    """Draws a 'watching' eye that follows intensity."""
    if random.random() < intensity * 0.3:
        x, y = IW // 2, IH // 2
        size = int(10 + intensity * 80)
        pupil_size = int(size * 0.2)
        # Sclera (white part)
        cv2.ellipse(frame, (x, y), (size, size // 2), 0, 0, 360, (255, 255, 255), -1)
        # Pupil (red)
        cv2.circle(frame, (x, y), pupil_size, (0, 0, 200), -1)
    return frame

def draw_grid_overlay(frame, intensity):
    """Draws a flickering digital grid."""
    if random.random() < intensity * 0.5:
        step = random.choice([10, 20, 30])
        color = SCENARIO_DATA["palette"][1]
        for i in range(0, IW, step):
            cv2.line(frame, (i, 0), (i, IH), color, 1)
        for i in range(0, IH, step):
            cv2.line(frame, (0, i), (IW, i), color, 1)
    return frame

def warp_distortion(frame, intensity):
    """Applies a 'breathing' sine-wave distortion."""
    mag = intensity * 10
    freq = intensity * 0.3 + 0.1
    map_x = xx + mag * np.sin(yy * freq)
    map_y = yy + mag * np.sin(xx * freq)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def draw_sigil_motif(frame, intensity):
    """Draws a random, cryptic sigil."""
    if random.random() < intensity * 0.2:
        x, y = IW // 2, IH // 2
        size = int(20 + intensity * 100)
        color = SCENARIO_DATA["palette"][1]
        # Triangle
        pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
        cv2.polylines(frame, [pts], True, color, 2)
        # Circle
        cv2.circle(frame, (x, y), size // 2, color, 2)
    return frame

# --- General & Volatile Mutators ---

def feedback_loop(frame, prev_frame, intensity):
    if prev_frame is None: return frame
    angle = (intensity - 0.5) * 2.0; M = cv2.getRotationMatrix2D((IW/2, IH/2), angle, 1.015);
    transformed_prev = cv2.warpAffine(prev_frame, M, (IW, IH), borderMode=cv2.BORDER_REPLICATE);
    return cv2.addWeighted(frame, 0.85, transformed_prev, 0.15, 0)

def datamosh_sim(frame, prev_frame, intensity):
    if prev_frame is None or random.random() > intensity * 0.7: return frame
    output = frame.copy(); num_blocks = int(intensity * 40);
    for _ in range(num_blocks):
        bh, bw = random.randint(8, 64), random.randint(8, 64); x, y = random.randint(0, IW-bw), random.randint(0, IH-bh);
        output[y:y+bh, x:x+bw] = prev_frame[y:y+bh, x:x+bw];
    return output

def random_color_invert(frame):
    """VOLATILE: Inverts colors on a random slice of the screen."""
    h, w = frame.shape[:2]; x, y = random.randint(0, w//2), random.randint(0, h//2);
    w_slice, h_slice = random.randint(w//4, w), random.randint(h//4, h);
    slice_ = frame[y:y+h_slice, x:x+w_slice];
    frame[y:y+h_slice, x:x+w_slice] = 255 - slice_;
    return frame

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
    visual_gen_func = globals()[SCENARIO_DATA["visual_gen"]]
    core_mutator_func = globals()[SCENARIO_DATA["core_mutator"]]
    palette = [np.array(c, dtype=np.uint8) for c in SCENARIO_DATA["palette"]]
    cryptic_phrases = SCENARIO_DATA["cryptic_phrases"]
    
    # Pool of general mutators
    general_mutators = [feedback_loop, datamosh_sim]
    
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
        
        # --- 1. Base Layer (Themed Generator + Palette) ---
        mono_frame = visual_gen_func(t_sec * 2)
        frame = apply_themed_palette(mono_frame, palette)

        # --- 2. Core "Character" Mutator ---
        frame = core_mutator_func(frame, vis_intensity)

        # --- 3. General Chaos Mutators (Random) ---
        for mut_func in general_mutators:
            if random.random() < vis_intensity * 0.3: # Low chance
                frame = mut_func(frame, prev_frame, vis_intensity)
        
        # --- 4. Volatile Mutations (High Randomness) ---
        if random.random() < 0.03:
            frame = random_color_invert(frame)
        if random.random() < 0.005:
            frame = 255 - frame # Full invert

        # --- 5. Cryptic Message Layer ---
        if random.random() < 0.01: # 1% chance per frame
            msg = random.choice(cryptic_phrases)
            pos = (random.randint(IW//10, IW//2), random.randint(IH//10, IH - IH//10))
            cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        prev_frame = frame.copy()
        
        # --- 6. Final Upscale and Stream ---
        final_frame = cv2.resize(frame, (X_RES, Y_RES), interpolation=cv2.INTER_NEAREST)
        proc.stdin.write(final_frame.tobytes())
        
        print(f"\r -> {CURRENT_SCENARIO_NAME}: FRAME {i+1}/{TOTAL_FRAMES} [INTENSITY: {vis_intensity:.2f}]", end="")

    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()
    if proc.returncode != 0: print("\nFFmpeg Error:", stderr)

if __name__ == '__main__':
    stream_video(OUTPUT_AUDIO)
    print(f"\n[+] NARRATIVE FRAGMENT GENERATED. ARTIFACT '{OUTPUT_VIDEO}' IS STABLE.")

# --- Cipher System ---
SEED_WORDS = ["WORMHOLE", "CHAOS", "ENTITY", "RELAY", "SHIFT", "PULSE", "GLITCH"]
MASTER_SEED = random.choice(SEED_WORDS)
HIDDEN_MESSAGE = f"THE_GATE_IS_OPENED_BY_{MASTER_SEED}"

CIPHER_MAP = {
    'A': ('flash', (0, 0, 255)),   'C': ('scanlines', None),
    'E': ('flash', (0, 255, 0)),   'H': ('datamosh_trigger', None),
    'L': ('feedback_spike', None), 'M': ('random_glyph', None),
    'O': ('flash', (255, 255, 255)), 'R': ('random_glyph', None),
    'S': ('scanlines', None),      'T': ('datamosh_trigger', None),
    'W': ('feedback_spike', None), 'Z': ('random_glyph', None),
}

# ██████████████ THEMATIC BLUEPRINTS (Core Randomizer) ██████████████

THEMATIC_MODES = {
    "DATA_STREAM_HIEROGLYPHS": {
        "visual_gen": 'geometric_stripes',
        "colormap": cv2.COLORMAP_JET,
        "primary_color": (255, 255, 255), # White on stripes
        "secondary_color": (0, 0, 0), # Black on stripes
        "audio_gen": 'chaotic_oscillator',
        "texture_mod": 1.5, # High frequency, static texture
        "message": "DECIPHER THE INPUT",
    },
    "VOID_OF_STATIC": {
        "visual_gen": 'fractal_noise',
        "colormap": cv2.COLORMAP_BONE, # Monochrome/desolate
        "primary_color": (50, 50, 50), # Dark gray
        "secondary_color": (255, 0, 0), # Blood red
        "audio_gen": 'granular_noise',
        "texture_mod": 0.5, # Low frequency, bass rumble
        "message": "NOTHING IS REAL",
    },
    "THE_ECHOING_CATHEDRAL": {
        "visual_gen": 'fractal_noise',
        "colormap": cv2.COLORMAP_TURBO, # Spectral, hallucinatory
        "primary_color": (255, 255, 0), # Yellow/Gold
        "secondary_color": (50, 0, 150), # Deep Purple
        "audio_gen": 'generate_shepard_risset',
        "texture_mod": 1.0, # Balanced texture
        "message": "THE LOOP IS FOREVER",
    },
    "PHANTOM_MEMORY_LEAK": {
        "visual_gen": 'geometric_stripes',
        "colormap": cv2.COLORMAP_MAGMA,
        "primary_color": (255, 100, 0), # Orange
        "secondary_color": (0, 0, 255), # Cyan/Blue
        "audio_gen": 'generate_melody',
        "texture_mod": 2.0, # Sharp, crackling texture
        "message": "DID YOU FORGET?",
    },
    "HYPERCUBE_CRUSH": {
        "visual_gen": 'fractal_noise',
        "colormap": cv2.COLORMAP_VIRIDIS,
        "primary_color": (0, 255, 0),
        "secondary_color": (255, 0, 0),
        "audio_gen": 'chaotic_oscillator',
        "texture_mod": 3.0, # Extremely high frequency, piercing texture
        "message": "THE DIMENSION FOLDS",
    }
}
CURRENT_THEME = random.choice(list(THEMATIC_MODES.keys()))
THEME_DATA = THEMATIC_MODES[CURRENT_THEME]

print(f"[*] OMNICHANNEL CORE INITIALIZED: {CURRENT_THEME}")
print(f"[*] THEMATIC MESSAGE: {THEME_DATA['message']}")
print("-" * 20)


# ██████████████ AUDIO GENERATORS ██████████████
# Assuming functions chaotic_oscillator, granular_noise, generate_shepard_risset, generate_melody exist.

def chaotic_oscillator(n, vol=0.5):
    # (Fast, chaotic tone generator)
    if n <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False)
    x = np.zeros(n); x[0] = random.random(); r = 3.999 
    for i in range(1, n): x[i] = r * x[i-1] * (1 - x[i-1])
    base_freq = random.uniform(500, 2000)
    sig = np.sin(2 * np.pi * base_freq * t * (1 + x * 0.1))
    return (sig * vol).astype(np.float32)

def granular_noise(n_samples, vol=0.6):
    # (Dense, crackling texture)
    output = np.zeros(n_samples, dtype=np.float32)
    for _ in range((n_samples // 500)):
        start = random.randint(0, n_samples - 500)
        grain = (np.random.rand(500) * 2 - 1) * np.hanning(500)
        output[start:start+500] += grain * random.uniform(0.1, 0.4)
    return np.clip(output, -1.0, 1.0) * vol

def generate_shepard_risset(n_samples, vol=0.4):
    # (Endlessly rising/falling tone)
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
    # (Dark, clipped melody)
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

# --- Volatile Auditory Mutation ---
def audio_spike(vol=1.0):
    """Generates a brief, piercing burst of noise."""
    duration_samples = int(SAMPLE_RATE * random.uniform(0.05, 0.3))
    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False)
    # Fast frequency sweep (siren/alarm)
    freq = np.exp(t * random.uniform(1, 5)) * random.uniform(500, 5000)
    wave = np.sin(2 * np.pi * freq * t) * np.hanning(duration_samples)
    return (wave * vol).astype(np.float32)

def compose_audio(control_vector):
    generator_func = globals()[THEME_DATA["audio_gen"]]
    total = np.zeros(DURATION * SAMPLE_RATE, dtype=np.float32)
    
    # 1. Base Layer (driven by selected generator)
    for i in range(DURATION // 2):
        start = i * SAMPLE_RATE * 2
        length = SAMPLE_RATE * 2
        end = min(len(total), start + length)
        chunk = generator_func(end - start, vol=0.5)
        total[start:end] += chunk

    # 2. INTENSE NOISE MODULATION (Thematic Texture)
    noise_mod_scale = THEME_DATA["texture_mod"]
    t_mod = np.linspace(0, DURATION * noise_mod_scale, len(total), endpoint=False)
    noise_mod = np.array([pnoise1(t_val, octaves=5) for t_val in t_mod]) * 0.5
    total += noise_mod.astype(np.float32)

    # 3. Volatile Auditory Spikes (5% chance every second)
    for i in range(DURATION):
        if random.random() < 0.05:
            spike = audio_spike(vol=random.uniform(0.8, 1.2))
            start_idx = i * SAMPLE_RATE
            end_idx = start_idx + len(spike)
            total[start_idx:end_idx] += spike
    
    # 4. Apply Reverb & Normalize
    ir_len = SAMPLE_RATE // 5
    impulse_response = (np.random.randn(ir_len) * np.exp(-np.arange(ir_len) * 0.008))
    total = np.convolve(total, impulse_response, 'same')
    
    mx = np.max(np.abs(total))
    if mx > 0: total = np.clip(total / mx, -1.0, 1.0)
    return np.int16(total * 32767)


# ██████████████ VISUAL GENERATORS & MUTATORS ██████████████

x = np.arange(IW, dtype=np.float32)
y = np.arange(IH, dtype=np.float32)
xx, yy = np.meshgrid(x, y)
vectorized_pnoise3 = np.vectorize(pnoise3)

def fractal_noise(t, scale=0.03, octaves=5):
    """Vectorized Perlin noise."""
    mat = vectorized_pnoise3(xx * scale, yy * scale, np.full((IH, IW), t), octaves=octaves)
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

def geometric_stripes(t):
    """Generates sharp, shifting, chaotic lines."""
    h, w = IH, IW
    mat = np.zeros((h, w), dtype=np.float32)
    # Fast-moving diagonal lines
    mat = np.sin((xx + yy) * 0.1 + t * 5)
    # Add a chaotic element driven by 2D noise
    noise_2d = vectorized_pnoise3(xx * 0.05, yy * 0.05, 0, octaves=3) * 0.5
    mat += noise_2d
    mat = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
    return mat.astype(np.uint8)

def apply_themed_colors(frame_mono):
    """Applies the current theme's colors to the grayscale/noise base."""
    # 1. Apply colormap
    base_color_frame = cv2.applyColorMap(frame_mono, THEME_DATA["colormap"])
    
    # 2. Blend with primary/secondary theme colors for sharper contrast
    c1 = np.array(THEME_DATA["primary_color"], dtype=np.uint8)
    c2 = np.array(THEME_DATA["secondary_color"], dtype=np.uint8)
    
    mask = frame_mono > 128
    
    # Create a simple two-color layer
    theme_layer = np.full((IH, IW, 3), c2, dtype=np.uint8)
    theme_layer[mask] = c1
    
    # Blend the colormap and the sharp theme colors
    return cv2.addWeighted(base_color_frame, 0.7, theme_layer, 0.3, 0)


# --- Mutators (need to be defined for the core to call them) ---
def feedback_loop(frame, prev_frame, intensity):
    if prev_frame is None: return frame
    h, w = frame.shape[:2]
    angle = (intensity - 0.5) * 3.0
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.02)
    transformed_prev = cv2.warpAffine(prev_frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(frame, 0.8, transformed_prev, 0.2, 0)

def datamosh_sim(frame, prev_frame, intensity):
    if prev_frame is None or random.random() > intensity: return frame
    output = frame.copy()
    num_blocks = int(intensity * 60)
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

def random_color_invert(frame):
    """Inverts colors on a random slice of the screen."""
    h, w = frame.shape[:2]
    x, y = random.randint(0, w//2), random.randint(0, h//2)
    w_slice, h_slice = random.randint(w//4, w), random.randint(h//4, h)
    slice_ = frame[y:y+h_slice, x:x+w_slice]
    frame[y:y+h_slice, x:x+w_slice] = 255 - slice_
    return frame

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
    audio_data = compose_audio(control_vector)
    with wave_module.open(audio_wav_path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    # Compile cipher and mutator lists
    clue_schedule = {int((i+1)*0.9*(TOTAL_FRAMES/(len(MASTER_SEED)+1))): CIPHER_MAP.get(c) for i, c in enumerate(MASTER_SEED) if c in CIPHER_MAP}
    visual_gen_func = globals()[THEME_DATA["visual_gen"]]
    
    # All non-volatile mutators (feedback, datamosh, scanlines)
    primary_mutators = [feedback_loop, datamosh_sim, apply_scanlines] 
    
    # FFMPEG Command: using 'ultrafast'
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{X_RES}x{Y_RES}', '-r', str(FPS), '-i', '-',
        '-i', audio_wav_path, '-c:v', 'libx264', '-preset', 'ultrafast',
        '-crf', '25', '-c:a', 'aac', '-b:a', '128k', OUTPUT_VIDEO
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    prev_frame = np.zeros((IH, IW, 3), dtype=np.uint8)
    datamosh_active = False

    for i in range(TOTAL_FRAMES):
        t_sec = i / FPS
        vis_intensity = control_vector['visual_intensity'](t_sec)
        
        # --- Base Layer at LOW resolution (Themed Generator) ---
        mono_frame = visual_gen_func(t_sec * 2)
        frame = apply_themed_colors(mono_frame)

        # --- Volatile/Random Mutations (Visual Anarchy) ---
        if random.random() < 0.005: # 0.5% chance per frame for full color inversion
            frame = 255 - frame
        if random.random() < 0.05: # 5% chance for a random slice inversion
            frame = random_color_invert(frame)

        # --- Primary Mutators (Intensity-driven) ---
        random.shuffle(primary_mutators)
        for mut_func in primary_mutators:
            if random.random() < 0.1 + vis_intensity * 0.5:
                # Handle args
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
            elif clue_type == 'random_glyph':
                glyph = random.choice(["Ø", "Σ", "†", "∆", "§", "∞"])
                cv2.putText(frame, glyph, (IW//2 - 20, IH//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 5)
            elif clue_type == 'datamosh_trigger':
                datamosh_active = True
            elif clue_type == 'scanlines':
                frame = apply_scanlines(frame, vis_intensity)

        prev_frame = frame.copy()
        
        # --- Final Upscale and Stream ---
        final_frame = cv2.resize(frame, (X_RES, Y_RES), interpolation=cv2.INTER_NEAREST)
        proc.stdin.write(final_frame.tobytes())
        
        print(f"\r -> {CURRENT_THEME}: FRAME {i+1}/{TOTAL_FRAMES} [INTENSITY: {vis_intensity:.2f}]", end="")

    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()
    if proc.returncode != 0: print("\nFFmpeg Error:", stderr)

if __name__ == '__main__':
    stream_video(OUTPUT_AUDIO)
    print(f"\n[+] OMNICHANNEL FEED COMPLETE. ARTIFACT '{OUTPUT_VIDEO}' IS STABLE.")




































