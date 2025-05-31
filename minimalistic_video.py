# -*- coding: utf-8 -*-
# Requires: pip install numpy opencv-python pygame noise scipy
import numpy as np
import cv2
import pygame # Still used for mixer init check if needed
import random
import string
import wave
import os
import math
import time # For potential time-based effects
import noise # pip install noise
import base64 # For encoded words example
import binascii # For hex encoded words example
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft # For potential spectral effects later
import subprocess
import copy # For deep copying theme data during style breaks
import inspect # For checking function signatures

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720 # Increased resolution for better quality
MIN_DURATION, MAX_DURATION = 15, 40 # Slightly shorter range, focus on impact
# << INCREASED FPS RANGE >>
MIN_FPS, MAX_FPS = 15, 30 # Higher FPS for smoother motion, but still allowing some glitchiness
OUTPUT_FILE = "extreme_video.mp4"
HIGH_FREQ_CUTOFF = 22000 # Allow slightly higher frequencies
LOW_FREQ_CUTOFF = 18      # Allow slightly lower frequencies (removed trailing space)
SAMPLE_RATE = 44100
time_step = 1.0 / SAMPLE_RATE # Defined but may not be used in all audio parts directly

# --- Enhanced Thematic Elements ---
# Added new themes and expanded existing ones with more abrasive elements

# Helper function for Zalgo text (more abrasive text)
def zalgo_text(text):
    """Applies combining diacritical marks randomly to text."""
    if not isinstance(text, str): # Ensure input is a string
        return text
    zalgo_chars = [chr(i) for i in range(0x0300, 0x036F + 1)] # Combining Diacritical Marks
    output = ""
    for char in text:
        output += char
        # Add a random number of Zalgo characters after each original character
        num_zalgo = random.randint(1, 8) # More chaos
        for _ in range(num_zalgo):
            # Ensure the chosen character is valid before appending
            try:
                output += random.choice(zalgo_chars)
            except IndexError: # Should not happen with the defined range, but safety first
                pass
    return output

# << NEW THEMES & ENHANCEMENTS >>
THEMES = {
    "Digital Decay EXTREME+": {
        "colors": [ # More jarring palettes
            [(0, 0, 0), (0, 255, 0), (255, 0, 0), (255,255,0)], # R/G/B/Y Glitch
            [(255, 255, 255), (0, 0, 0), (50, 50, 50), (150,0,0)], # Stark + Blood Red
            [(0, 10, 20), (200, 220, 255), (255, 100, 0), (0,0,0)], # Cold Blue/Orange/Black
        ],
        "words": ["ERROR", "CORRUPT", "FRAGMENT", "NULL", "VOID", "DELETE", "SEGFAULT", "BUFFER", "OVERFLOW", "STATIC", "LOST", "SIGNAL", "404", "0xDEADBEEF", "PANIC", "UNSTABLE", "KERNEL"],
        "symbols": ["▌", "█", "▓", "⚠️", "⚡", "☣️", "中断", "异常", "뷁", "௹", "//", "%", "&", "*", "!", "?", "01", "||", "::", ">>", "FAIL"],
        "zalgo_words_chance": 0.4, # Chance to apply Zalgo effect to words
        "visual_effects": ["apply_perlin_noise", "apply_block_shift", "apply_color_channel_shift", "apply_warp", "apply_pixelation", "apply_scanlines", "apply_extreme_contrast", "apply_feedback", "apply_datamosh_sim", "apply_ascii_sim"], # Added new effects (use function names)
        "audio_freq_range": (30, 18000),
        "audio_noise_types": ["white", "glitch", "static", "digital_artifact", "screech", "digital_clipping_sim", "feedback_screech", "pink", "brown"], # Added harsher noises + pink/brown
        "audio_features": ["stutter", "bitcrush", "clicks", "extreme_panning", "distortion", "extreme_pitch", "spectral_glitch"], # Added spectral glitch
        "melody_scale": "minor_pentatonic", # Simple scale for buried melody
        "melody_distortion": 15.0, # Heavy distortion for melody
    },
    "Organic Corruption EXTREME+": {
        "colors": [ # More visceral colors
            [(40, 5, 5), (200, 30, 30), (30, 60, 30), (240, 230, 200)], # Dark Flesh/Bone
            [(10, 15, 5), (30, 70, 15), (180, 180, 150), (90, 10, 10)], # Decay/Dried Blood
            [(0, 0, 0), (120, 0, 5), (255, 60, 60), (50, 50, 80)], # Viscera/Bruise
        ],
        "words": ["GROW", "DECAY", "CONSUME", "INFECT", "MUTATE", "FLESH", "BONE", "ROT", "SPORE", "INSIDE", "BREATHE", "PULSATE", "WRITHE", "SWELL", "BURST", "MERGE", "ASSIMILATE"],
        "symbols": ["Ø", "§", " संक्रमित", "تلوث", "🦠", "🍄", "🦴", "👁️‍🗨️", "〰", "~~", "...", "呼吸", "∬", "∯", "⌬", "⏳", "⚕️", "♾️"], # Added medical/infinity
        "zalgo_words_chance": 0.6,
        "visual_effects": ["apply_perlin_noise", "apply_block_shift", "apply_warp", "apply_feedback", "apply_crt_ghost_sim", "apply_extreme_contrast", "apply_solarize", "apply_slit_scan_sim"], # Different effect focus (ensure functions exist)
        "audio_freq_range": (18, 2000), # Even lower floor
        "audio_noise_types": ["brown", "squelch", "heartbeat", "breathing", "wet_clicks", "sub_bass", "granular_flesh", "choking_sim", "pink"], # Added granular/choking + pink/brown
        "audio_features": ["wet_sounds", "slow_lfo", "dissonance", "low_rumble", "distortion", "extreme_pitch", "sub_bass_throb", "convolution_reverb_damp"], # Added reverb
        "melody_scale": "phrygian", # Dissonant scale
        "melody_distortion": 10.0,
    },
    "Algorithmic Dream (NEW)": {
        "colors": [
            [(10, 0, 20), (150, 180, 255), (255, 255, 255), (50, 50, 50)], # Deep Purple/Bright Blue/White/Grey
            [(0, 0, 0), (255, 100, 0), (0, 200, 200), (200, 200, 200)], # Black/Orange/Cyan/Light Grey
            [(80, 80, 100), (220, 220, 240), (30, 30, 50), (255, 0, 100)], # Dreamlike + Magenta shock
        ],
        "words": ["IF", "THEN", "ELSE", "RECURSE", "WHILE", "TRUE", "FALSE", "CALCULATE", "PROCESS", "SLEEP", "AWAKEN", "QUERY", "RESPONSE", "SIMULATE"],
        "symbols": ["{}", "()", "[]", "=>", "->", "...", "&&", "||", "!", "?", "#", "_", "λ", "Σ", "Π", "Δ"], # Code/Logic symbols
        "zalgo_words_chance": 0.2,
        "visual_effects": ["apply_perlin_noise", "apply_warp", "apply_feedback", "apply_color_channel_shift", "apply_vector_field_sim", "apply_ascii_sim", "apply_extreme_contrast", "apply_pixelation"], # Ensure functions exist
        "audio_freq_range": (50, 16000),
        "audio_noise_types": ["pink", "static", "digital_artifact", "modem_sim", "filtered_noise", "granular_synth", "brown"], # Ensure functions exist + pink/brown
        "audio_features": ["stutter", "bitcrush", "delay", "reverb", "extreme_panning", "spectral_glitch", "tts_fragments"], # Added TTS (TTS needs external lib)
        "melody_scale": "chromatic", # All notes possible
        "melody_distortion": 5.0, # Less harsh distortion
    },
}

# --- Scales for Buried Melody ---
SCALES = {
    "minor_pentatonic": [0, 3, 5, 7, 10], # Root, m3, p4, p5, m7
    "phrygian": [0, 1, 3, 5, 7, 8, 10], # Root, m2, m3, p4, p5, m6, m7 (dissonant)
    "chromatic": list(range(12)), # All notes
    "major": [0, 2, 4, 5, 7, 9, 11],
}

# --- Font Configuration ---
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux
    "/System/Library/Fonts/Arial Unicode.ttf", # MacOS
    "C:/Windows/Fonts/arialuni.ttf", # Windows (if installed)
    "C:/Windows/Fonts/arial.ttf",
    # "NotoSans-Regular.ttf", # Example: Add path to a specific broad Unicode font if available
]
AVAILABLE_FONTS = [f for f in FONT_PATHS if os.path.exists(f)]
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX # Fallback Hershey font
SELECTED_TTF_FONT = None

if AVAILABLE_FONTS:
    broad_fonts = [f for f in AVAILABLE_FONTS if 'unicode' in f.lower() or 'dejavu' in f.lower() or 'noto' in f.lower()]
    if broad_fonts:
        SELECTED_TTF_FONT = random.choice(broad_fonts)
    else:
        SELECTED_TTF_FONT = random.choice(AVAILABLE_FONTS)

print(f"Available TTF fonts found: {AVAILABLE_FONTS}")
if SELECTED_TTF_FONT:
    print(f"Selected TTF font (for potential use with PIL/FreeType): {SELECTED_TTF_FONT}")
else:
    print("No TTF fonts found or selected from the predefined paths.")

# For cv2.putText, we primarily use Hershey fonts due to dependency simplicity.
# Using TTF with cv2.putText typically requires OpenCV compiled with FreeType support
# and using the cv2.freetype module, which is a more complex setup.
# FONT_TO_USE was SELECTED_TTF_FONT if SELECTED_TTF_FONT else DEFAULT_FONT
# However, FONT_FACE_TO_USE is set to DEFAULT_FONT, meaning cv2.putText will use Hershey.
FONT_FACE_TO_USE = DEFAULT_FONT
print(f"Font face for cv2.putText: Hershey Font ({FONT_FACE_TO_USE})")


# Initialize pygame mixer
try:
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=2048)
except pygame.error as e:
    print(f"Pygame mixer init failed (non-critical for file output): {e}")


# --- Intensity Profile Generation ---
def generate_intensity_profile(duration_samples, sample_rate, scale=25.0, octaves=5, persistence=0.55, lacunarity=2.2):
    """Generates a more dynamic Perlin noise-based intensity profile."""
    duration_seconds = duration_samples / sample_rate
    num_points = int(duration_seconds * 15) # More points for finer control
    if num_points < 2: num_points = 2
    profile = np.zeros(num_points)
    base = random.randint(0, 500)

    for i in range(num_points):
        profile[i] = noise.pnoise1(i / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base)

    min_val, max_val = np.min(profile), np.max(profile)
    if max_val - min_val > 1e-6:
        normalized_profile = (profile - min_val) / (max_val - min_val)
        intensity_min = 0.1 # Allow deeper lows
        intensity_max = 2.5 # Allow higher peaks
        scaled_profile = intensity_min + normalized_profile * (intensity_max - intensity_min)
    else:
        scaled_profile = np.full(num_points, 1.0)

    profile_times = np.linspace(0, duration_seconds, num_points)
    intensity_func = interp1d(profile_times, scaled_profile, kind='cubic', bounds_error=False,
                              fill_value=(scaled_profile[0], scaled_profile[-1]))
    print(f"Generated shared intensity profile: {len(profile)} points over {duration_seconds:.1f}s (Range: {np.min(scaled_profile):.2f}-{np.max(scaled_profile):.2f})")
    return intensity_func


# --- Visual Glitch Functions ---

def apply_perlin_noise(frame, alpha_range=(0.1, 0.9), scale_range=(2.0, 80.0), oct_range=(3, 9)):
    alpha = random.uniform(*alpha_range)
    scale = random.uniform(*scale_range)
    octaves = random.randint(*oct_range)
    persistence = random.uniform(0.2, 0.7)
    lacunarity = random.uniform(1.8, 3.5)
    seed = random.randint(0, 2000)
    height, width = frame.shape[:2]
    
    gray_noise = np.zeros((height, width), dtype=np.float32)
    # Generating Perlin noise per pixel (can be slow for large frames)
    # For performance, one might explore vectorized noise if library supports or precomputed noise textures.
    for i in range(height):
        for j in range(width):
            gray_noise[i, j] = noise.pnoise2(i / scale, j / scale, 
                                             octaves=octaves, persistence=persistence, 
                                             lacunarity=lacunarity, base=seed)
    
    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX)
    colored_noise = cv2.cvtColor(colored_noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if random.random() < 0.4: colored_noise = 255 - colored_noise
    return cv2.addWeighted(frame, 1 - alpha, colored_noise, alpha, 0)

def apply_block_shift(frame, max_shift_range=(10, WIDTH // 3), block_size_range=(5, HEIGHT // 2), num_blocks_range=(20, 100)):
    max_shift = random.randint(*max_shift_range)
    bs_min = max(1, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size_max_val = random.randint(bs_min, bs_max) # Renamed to avoid conflict
    num_blocks = random.randint(*num_blocks_range)
    height, width = frame.shape[:2]
    temp_frame = frame.copy()

    for _ in range(num_blocks):
        bh = random.randint(bs_min, block_size_max_val)
        bw = random.randint(bs_min, block_size_max_val)
        
        if height - bh <= 0 or width - bw <= 0: continue # Ensure valid source block dimensions
        y = random.randint(0, height - bh -1) # Corrected upper bound if bh is large
        x = random.randint(0, width - bw -1)  # Corrected upper bound if bw is large

        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)

        target_y_start = np.clip(y + shift_y, 0, height - bh)
        target_x_start = np.clip(x + shift_x, 0, width - bw)
        
        try:
            block = frame[y:y+bh, x:x+bw]
            temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block
        except ValueError: # Catches potential shape mismatches if logic is flawed
            # print(f"Block shift slice error, skipping block. Block shape: {block.shape}, Target slice: ({bh}, {bw})")
            pass
    return temp_frame

def apply_color_channel_shift(frame, max_shift_range=(5, 60)):
    max_shift = random.randint(*max_shift_range)
    temp_frame = frame.copy()
    for i in range(3): # B, G, R
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        channel = frame[:,:,i]
        shifted_channel = np.roll(np.roll(channel, shift_y, axis=0), shift_x, axis=1)
        temp_frame[:,:,i] = shifted_channel
    return temp_frame

def apply_warp(frame, amplitude_range=(5, 80), freq_range=(0.002, 0.1)):
    rows, cols = frame.shape[:2]
    amplitude_x = random.uniform(*amplitude_range)
    frequency_x = random.uniform(*freq_range)
    amplitude_y = random.uniform(*amplitude_range) * random.uniform(0.3, 1.7)
    frequency_y = random.uniform(*freq_range) * random.uniform(0.3, 1.7)
    phase_x = random.uniform(0, 2 * math.pi)
    phase_y = random.uniform(0, 2 * math.pi)

    x_mesh, y_mesh = np.meshgrid(np.arange(cols), np.arange(rows))
    
    offset_x = (amplitude_x * np.sin(2 * math.pi * y_mesh * frequency_x + phase_x)).astype(np.float32)
    offset_y = (amplitude_y * np.cos(2 * math.pi * x_mesh * frequency_y + phase_y)).astype(np.float32)
    
    map_x = np.clip(x_mesh.astype(np.float32) + offset_x, 0, cols - 1)
    map_y = np.clip(y_mesh.astype(np.float32) + offset_y, 0, rows - 1)
    
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def apply_feedback(frame, prev_frame, alpha_range=(0.02, 0.6)):
    if prev_frame is None: return frame
    alpha = random.uniform(*alpha_range)
    modified_prev = prev_frame.copy()

    if random.random() < 0.2:
        rows, cols = modified_prev.shape[:2]
        angle = random.uniform(-5, 5)
        scale = random.uniform(0.95, 1.05)
        tx = random.uniform(-10, 10)
        ty = random.uniform(-10, 10)
        center = (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        modified_prev = cv2.warpAffine(modified_prev, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.1:
        shift_b = random.randint(-10, 10)
        shift_g = random.randint(-10, 10)
        shift_r = random.randint(-10, 10)
        # This performs uint8 arithmetic (wraps around)
        modified_prev = modified_prev + np.array([shift_b, shift_g, shift_r], dtype=np.uint8) 
        # For explicit clipping:
        # shift_arr = np.array([shift_b, shift_g, shift_r], dtype=np.int16)
        # modified_prev = np.clip(modified_prev.astype(np.int16) + shift_arr, 0, 255).astype(np.uint8)

    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)

def apply_pixelation(frame, block_size_range=(4, 96)):
    height, width = frame.shape[:2]
    bs_min = max(2, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size = random.randint(bs_min, bs_max)

    pixel_w, pixel_h = max(1, width // block_size), max(1, height // block_size)
    temp = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def apply_scanlines(frame, intensity_range=(0.4, 1.0), thickness_range=(1, 4), color_variation=40):
    intensity = random.uniform(*intensity_range)
    thickness = random.randint(*thickness_range)
    scanline_layer = np.zeros_like(frame)
    height, width = frame.shape[:2]
    base_color_val = random.randint(0, 70)

    for y_pos in range(0, height, thickness * 2): # Corrected variable name
        line_color_val = base_color_val + random.randint(-color_variation, color_variation)
        line_color = tuple(np.clip([line_color_val]*3, 0, 255).astype(np.uint8)) # Ensure uint8 for color
        cv2.line(scanline_layer, (0, y_pos), (width, y_pos), line_color, thickness)

    return cv2.addWeighted(frame, 1.0, scanline_layer, intensity * 1.1, -int(255 * intensity * 0.3))

def apply_solarize(frame, threshold_range=(60, 200)):
    threshold = random.randint(*threshold_range)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    solarized_frame = frame.copy()
    solarized_frame[mask == 255] = 255 - solarized_frame[mask == 255]
    return solarized_frame

def apply_extreme_contrast(frame, alpha_range=(1.0, 6.0), beta_range=(-100, 100)):
    alpha = random.uniform(*alpha_range) # Contrast
    beta = random.randint(*beta_range)   # Brightness
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_datamosh_sim(frame, prev_frame, hold_prob=0.3, smear_prob=0.5, block_size_range=(32, 128)):
    if prev_frame is None: return frame
    height, width = frame.shape[:2]
    output = frame.copy()
    bs_min = max(16, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size = random.randint(bs_min, bs_max)

    if random.random() < hold_prob:
        num_hold_blocks = random.randint(1, max(1, (width * height) // (block_size**2) // 3)) # Approx blocks
        for _ in range(num_hold_blocks):
            if width // block_size <=0 or height // block_size <=0: continue
            bx_idx = random.randint(0, max(0, width // block_size - 1))
            by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            bh = min(block_size, height - by)
            bw = min(block_size, width - bx)
            if bh > 0 and bw > 0:
                output[by:by+bh, bx:bx+bw] = prev_frame[by:by+bh, bx:bx+bw]

    if random.random() < smear_prob:
        num_smear_blocks = random.randint(1, max(1, (width * height) // (block_size**2) // 4))
        for _ in range(num_smear_blocks):
            if width // block_size <=0 or height // block_size <=0: continue
            bx_idx = random.randint(0, max(0, width // block_size - 1))
            by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            bh = min(block_size, height - by)
            bw = min(block_size, width - bx)
            if bh <= 0 or bw <= 0: continue

            vx = random.randint(-block_size // 2, block_size // 2)
            vy = random.randint(-block_size // 2, block_size // 2)
            
            smear_block = prev_frame[by:by+bh, bx:bx+bw]
            
            target_y = np.clip(by + vy, 0, height - bh)
            target_x = np.clip(bx + vx, 0, width - bw)
            
            # Ensure smear_block can fit into the target slice
            # This might happen if bh/bw were reduced by min() and target is at edge
            target_bh = min(bh, height - target_y)
            target_bw = min(bw, width - target_x)
            if target_bh > 0 and target_bw > 0 :
                 output[target_y:target_y+target_bh, target_x:target_x+target_bw] = smear_block[:target_bh, :target_bw]
    return output

def apply_ascii_sim(frame, block_size_range=(8, 16), char_set=" .:-=+*#%@", invert=False):
    height, width = frame.shape[:2]
    bs_min = max(4, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size = random.randint(bs_min, bs_max)
    output = frame.copy()
    font_scale = max(0.1, block_size / 15.0)
    thickness = 1
    chars = list(char_set)
    if invert: chars = chars[::-1]
    num_chars = len(chars)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            block_gray = gray[y:y+block_size, x:x+block_size] # Use gray block
            avg_brightness = np.mean(block_gray)
            char_index = int(np.clip((avg_brightness / 255.0) * num_chars, 0, num_chars - 1))
            char = chars[char_index]
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255)
            
            (text_w, text_h), _ = cv2.getTextSize(char, FONT_FACE_TO_USE, font_scale, thickness)
            pos = (x + (block_size - text_w) // 2, y + (block_size + text_h) // 2)
            try:
                cv2.putText(output, char, pos, FONT_FACE_TO_USE, font_scale, text_color, thickness, cv2.LINE_AA)
            except Exception: # Ignore errors from putText (e.g. bad char with Hershey)
                pass
    return output

def apply_crt_ghost_sim(frame, prev_frame):
    if prev_frame is not None:
        return cv2.addWeighted(frame, 0.85, prev_frame, 0.15, 0)
    return frame

def apply_slit_scan_sim(frame):
    rows, cols, _ = frame.shape
    output = frame.copy()
    for r_idx in range(rows): # Renamed variable
        smear_amount = int(math.sin(r_idx * 0.1 + random.uniform(0, math.pi)) * random.uniform(5,15)) 
        output[r_idx, :] = np.roll(output[r_idx, :], smear_amount, axis=0) # Smear along columns for each row
    return output

def apply_vector_field_sim(frame):
    return apply_warp(frame, amplitude_range=(2,10), freq_range=(0.05, 0.2))


# --- Audio Generation & Effects ---

def apply_distortion(data, intensity_range=(1.5, 5.0)):
    intensity = random.uniform(*intensity_range)
    return np.clip(data * intensity, -0.98, 0.98)

def apply_bitcrush(data, bit_depth_range=(4, 12)):
    bd_min = max(2, bit_depth_range[0])
    bd_max = max(bd_min + 1, bit_depth_range[1])
    bits = random.randint(bd_min, bd_max)
    if bits >= 16: return data
    steps = 2**(bits - 1)
    quantized = np.round(data * steps) / steps
    return quantized

def apply_spectral_glitch(data, intensity=0.5):
    n_samples = len(data)
    if n_samples < 1024: return data.astype(np.float32) # Ensure float output

    data_float = data.astype(np.float32)
    spectrum = fft(data_float)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    n_freqs = len(spectrum)

    num_glitches = int(n_freqs * 0.05 * intensity * random.random())
    # Ensure k for random.sample is not larger than population
    population_size = n_freqs // 2 -1
    if population_size <=0 : return data_float # Not enough frequencies to glitch

    k_sample = min(num_glitches, population_size)
    if k_sample <=0 : return data_float

    indices_to_glitch = random.sample(range(1, n_freqs // 2), k_sample) 

    for idx in indices_to_glitch:
        if random.random() < 0.6: 
            magnitude[idx] = 0
            if idx < n_freqs: magnitude[n_freqs - idx] = 0 # Symmetric for real signal
        else: 
            if idx + 1 < n_freqs // 2:
                phase[idx], phase[idx+1] = phase[idx+1], phase[idx]
                # Mirror phase changes for real signal (conjugate symmetry)
                # phase of H[-k] is -phase[k]
                if n_freqs - idx < n_freqs : phase[n_freqs - idx] = -phase[idx] 
                if n_freqs - (idx+1) < n_freqs : phase[n_freqs - (idx+1)] = -phase[idx+1]

    new_spectrum = magnitude * np.exp(1j * phase)
    glitched_data = np.real(ifft(new_spectrum))

    max_abs_orig = np.max(np.abs(data_float))
    max_abs_glitched = np.max(np.abs(glitched_data))
    if max_abs_glitched > 1e-6 and max_abs_orig > 1e-6:
        glitched_data *= (max_abs_orig / max_abs_glitched)
    elif max_abs_glitched > 1e-6: # If original was silent
        glitched_data /= max_abs_glitched
    
    return glitched_data.astype(data.dtype)


def apply_convolution_reverb(data, impulse_response):
    n_data = len(data)
    n_ir = len(impulse_response)

    if n_ir == 0: return data
    if n_ir >= n_data and n_data > 0: # If IR is too long
        impulse_response_eff = impulse_response[:n_data//2 if n_data//2 > 0 else 1] # Truncate
    else:
        impulse_response_eff = impulse_response
    
    if len(impulse_response_eff) == 0 and n_data > 0: # If truncation resulted in empty IR
        return data
    elif n_data == 0:
        return data


    reverbed_data = np.convolve(data, impulse_response_eff, mode='same')

    max_abs_orig = np.max(np.abs(data)) if n_data > 0 else 0
    max_abs_rev = np.max(np.abs(reverbed_data)) if len(reverbed_data) > 0 else 0
    
    if max_abs_rev > 1e-6 and max_abs_orig > 1e-6:
        reverbed_data *= (max_abs_orig / max_abs_rev)
    elif max_abs_rev > 1e-6:
         reverbed_data /= max_abs_rev
         
    return reverbed_data

# Simple impulse responses (can be expanded)
IR_SMALL_ROOM = np.exp(-np.arange(SAMPLE_RATE // 20) / (SAMPLE_RATE / 200)) * \
                (np.random.rand(SAMPLE_RATE // 20) - 0.5)
if len(IR_SMALL_ROOM) > 0: IR_SMALL_ROOM /= np.max(np.abs(IR_SMALL_ROOM))

IR_DAMPED_SPACE = np.exp(-np.arange(SAMPLE_RATE // 5) / (SAMPLE_RATE / 50)) * \
                  (np.random.rand(SAMPLE_RATE // 5) - 0.5)
if len(IR_DAMPED_SPACE) > 0: IR_DAMPED_SPACE /= np.max(np.abs(IR_DAMPED_SPACE))


def generate_melody_samples(duration_samples, theme_data, intensity, sample_rate):
    """Generates a simple, possibly distorted, melodic sequence."""
    if duration_samples <= 0:
        return np.array([])
        
    scale_name = theme_data.get("melody_scale", "minor_pentatonic")
    scale_intervals = SCALES.get(scale_name, SCALES["minor_pentatonic"])
    base_freq = random.uniform(50, 200) # Lower base frequency
    num_notes = random.randint(max(1, duration_samples // (sample_rate // 2)), max(2, duration_samples // (sample_rate // 8))) # Fewer, longer notes
    if num_notes == 0 : return np.zeros(duration_samples)
    
    note_duration_samples = duration_samples // num_notes
    if note_duration_samples == 0: return np.zeros(duration_samples) # Avoid issues with tiny durations

    melody = np.array([])
    for _ in range(num_notes):
        # Choose a note from the scale
        semitone_offset = random.choice(scale_intervals)
        # Add random octave shift for more variation
        octave_shift = random.choice([-1, 0, 0, 1]) * 12 
        current_freq = base_freq * (2**((semitone_offset + octave_shift) / 12.0))
        current_freq = np.clip(current_freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF / 2) # Clip freq

        t_note = np.linspace(0, note_duration_samples / sample_rate, note_duration_samples, endpoint=False)
        note_wave = 0.3 * np.sin(2 * np.pi * current_freq * t_note) # Sine wave base

        # Simple ADSR-like envelope
        attack_len = min(note_duration_samples // 10, sample_rate // 100) # Shorter attack
        decay_len = min(note_duration_samples // 5, sample_rate // 50)
        sustain_level = 0.7
        release_len = min(note_duration_samples // 10, sample_rate // 100)
        
        if attack_len > 0 : note_wave[:attack_len] *= np.linspace(0, 1, attack_len)
        if decay_len > 0 and attack_len + decay_len <= note_duration_samples:
             note_wave[attack_len : attack_len+decay_len] *= np.linspace(1, sustain_level, decay_len)
        if attack_len + decay_len < note_duration_samples - release_len: # Sustain part
            note_wave[attack_len+decay_len : note_duration_samples - release_len] *= sustain_level
        if release_len > 0 : note_wave[note_duration_samples-release_len:] *= np.linspace(sustain_level, 0, release_len)


        melody = np.concatenate((melody, note_wave))

    # Apply distortion based on theme
    distortion_intensity = theme_data.get("melody_distortion", 0) * intensity
    if distortion_intensity > 0.1 and len(melody) > 0:
        melody = np.clip(melody * (1 + distortion_intensity), -0.98, 0.98) # Simple distortion

    # Ensure melody is correct length if concat results in slight mismatch
    if len(melody) > duration_samples:
        melody = melody[:duration_samples]
    elif len(melody) < duration_samples and len(melody) > 0 : # Pad if too short
        melody = np.pad(melody, (0, duration_samples - len(melody)), 'constant')
    elif len(melody) == 0 and duration_samples > 0: # If melody generation failed
        melody = np.zeros(duration_samples)
        
    return melody


def generate_placeholder_noise(duration_samples, vol, features=None):
    """Generic placeholder for various noise types."""
    if duration_samples <= 0: return np.array([])
    return np.random.uniform(-0.1, 0.1, duration_samples) * vol

# --- Actual Pink and Brown Noise Generation ---
def generate_pink_noise(N, scale=1.0):
    """Generates pink noise (1/f power spectrum)."""
    if N <= 0:
        return np.array([], dtype=np.float64)
    
    white_noise = np.random.randn(N)
    fft_white = np.fft.rfft(white_noise) # Real FFT
    num_freqs = len(fft_white)

    # Create filter: amplitude proportional to 1/sqrt(f)
    # Power is 1/f. Indices k are used as proxy for f.
    pink_filter = np.ones(num_freqs)
    if num_freqs > 1: # Handle DC component (index 0) separately
        # For k=1,2,... filter is 1/sqrt(k)
        indices = np.arange(1, num_freqs) 
        pink_filter[1:] /= np.sqrt(indices) 
    
    fft_pink = fft_white * pink_filter
    pink_noise_samples = np.fft.irfft(fft_pink, n=N) # Inverse real FFT
    
    max_abs = np.max(np.abs(pink_noise_samples))
    if max_abs > 1e-6: # Avoid division by zero
        pink_noise_samples /= max_abs # Normalize to +/-1
        
    return pink_noise_samples * scale

def generate_brown_noise(N, scale=1.0):
    """Generates brown noise (1/f^2 power spectrum) by integrating white noise."""
    if N <= 0:
        return np.array([], dtype=np.float64)
        
    white_noise = np.random.randn(N)
    brown_noise_samples = np.cumsum(white_noise)
    
    # Remove DC offset and normalize
    if N > 0 : brown_noise_samples -= np.mean(brown_noise_samples) 
    max_abs = np.max(np.abs(brown_noise_samples))
    if max_abs > 1e-6:
        brown_noise_samples /= max_abs # Normalize to +/-1
        
    return brown_noise_samples * scale


def generate_noise(noise_choice, event_duration_samples, event_vol, features):
    """Generates a short segment of a specific noise type."""
    noise = None # Initialize noise as None for fallback logic
    if event_duration_samples <= 0:
        return np.array([])

    if noise_choice == "white":
        noise = (np.random.rand(event_duration_samples) * 2 - 1) * event_vol
    elif noise_choice == "static":
        noise = np.zeros(event_duration_samples)
        # *** BUG FIX HERE ***
        # Ensure we don't try to write past the buffer or use more random numbers than available samples
        assign_len = min(100, event_duration_samples)
        if assign_len > 0: # Only assign if there's space and length is positive
            noise[0:assign_len] = (np.random.rand(assign_len) - 0.5) * 0.1 * event_vol # Scale by event_vol too
        # Add more continuous static for the rest
        if event_duration_samples > assign_len:
            noise[assign_len:] = (np.random.rand(event_duration_samples - assign_len) - 0.5) * 0.05 * event_vol

    elif noise_choice == "glitch":
        noise = np.zeros(event_duration_samples)
        num_clicks = random.randint(1, max(1, event_duration_samples // 200 if event_duration_samples >= 200 else 1))
        for _ in range(num_clicks):
            if event_duration_samples == 0: continue
            click_pos = random.randint(0, event_duration_samples - 1)
            noise[click_pos] = (random.random() * 2 - 1) * event_vol * 1.5 # Louder clicks
    elif noise_choice == "digital_artifact":
        noise = np.zeros(event_duration_samples)
        glitch_len = random.randint(1, max(2, event_duration_samples // 10 if event_duration_samples >= 10 else 2))
        if event_duration_samples - glitch_len < 0 : # Not enough space for glitch
             noise = generate_placeholder_noise(event_duration_samples, event_vol, features)
        else:
            start_pos = random.randint(0, event_duration_samples - glitch_len)
            glitch_val = (random.random() * 2 - 1) * event_vol
            noise[start_pos : start_pos + glitch_len] = glitch_val
            # Add some rapid changes within the glitch
            for i in range(start_pos, min(start_pos + glitch_len, event_duration_samples)):
                 if random.random() < 0.3: noise[i] *= random.uniform(-1,1)


    elif noise_choice == "screech":
        t = np.linspace(0, event_duration_samples / SAMPLE_RATE, event_duration_samples, endpoint=False)
        freq = random.uniform(1000, HIGH_FREQ_CUTOFF / 1.5) * (1 + np.sin(t * random.uniform(5,50)) * 0.5) # Modulated freq
        noise = np.sin(2 * np.pi * freq * t) * event_vol * 0.7

    elif noise_choice == "digital_clipping_sim":
        noise = np.zeros(event_duration_samples)
        pulse_len = SAMPLE_RATE // random.randint(100, 500) # e.g. 88 to 441 samples
        if pulse_len == 0: pulse_len = 1
        num_pulses = event_duration_samples // (pulse_len * 2) # Ensure space between pulses
        for i in range(num_pulses):
            start = i * pulse_len * 2
            if start + pulse_len <= event_duration_samples:
                 noise[start : start + pulse_len] = random.choice([-1, 1]) * event_vol
    
    elif noise_choice == "feedback_screech":
        t = np.linspace(0, event_duration_samples / SAMPLE_RATE, event_duration_samples, endpoint=False)
        base_f = random.uniform(500, 3000)
        mod_f = random.uniform(10, 100) * np.sin(t * random.uniform(0.1, 5)) # Slow modulation
        freq = base_f + mod_f + (t * random.uniform(1000, 5000)) # Rising pitch
        freq = np.clip(freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF * 0.8)
        noise = np.sin(2 * np.pi * freq * t)
        # Apply distortion for harsher sound
        noise = np.clip(noise * random.uniform(1.5, 5.0) * event_vol, -event_vol, event_vol)

    # --- Using new noise generators ---
    elif noise_choice == "pink":
        noise = generate_pink_noise(event_duration_samples, scale=event_vol)
    elif noise_choice == "brown":
        noise = generate_brown_noise(event_duration_samples, scale=event_vol)
    
    # --- Placeholder for other specific noises from themes ---
    elif noise_choice in ["squelch", "heartbeat", "breathing", "wet_clicks", "sub_bass", 
                           "granular_flesh", "choking_sim", "modem_sim", "filtered_noise", 
                           "granular_synth", "tts_fragments"]: # TTS would need specific library
        # print(f"Using placeholder for specific noise: {noise_choice}")
        noise = generate_placeholder_noise(event_duration_samples, event_vol, features)


    # Fallback for any unimplemented or None noise types
    if noise is None:
        # print(f"Warning: Noise type '{noise_choice}' not fully implemented or failed, using placeholder.")
        noise = generate_placeholder_noise(event_duration_samples, event_vol, features if 'features' in locals() else {})

    # Ensure output is numpy array
    if not isinstance(noise, np.ndarray):
        noise = np.array(noise if noise is not None else [], dtype=np.float32)
    if len(noise) != event_duration_samples and event_duration_samples > 0 : # Ensure correct length
        if len(noise) > event_duration_samples: noise = noise[:event_duration_samples]
        else: noise = np.pad(noise, (0, event_duration_samples - len(noise)), 'constant')
    elif event_duration_samples == 0:
        return np.array([])

    return noise.astype(np.float32)


def generate_audio_enhanced(duration_samples, fps, theme_data, intensity_function, global_params=None):
    """Generates complex, layered audio based on theme and intensity."""
    # global_params is unused in this snippet but kept for signature consistency
    print(f"Generating {duration_samples / SAMPLE_RATE:.1f}s audio, Theme: {theme_data.get('name', 'Unknown Theme')}...")

    audio_data_stereo = np.zeros((duration_samples, 2), dtype=np.float32)
    
    # Layer 1: Buried Melody (if theme supports)
    if "melody_scale" in theme_data:
        melody_samples = generate_melody_samples(duration_samples, theme_data, 
                                                 intensity_function(duration_samples / SAMPLE_RATE / 2), # Avg intensity
                                                 SAMPLE_RATE)
        if len(melody_samples) == duration_samples:
            audio_data_stereo[:, 0] += melody_samples * 0.2 # Pan left slightly
            audio_data_stereo[:, 1] += melody_samples * 0.2 # Pan right slightly
        else:
            print(f"Warning: Melody length mismatch. Expected {duration_samples}, got {len(melody_samples)}")


    # Layer 2: Dynamic Noise Events
    num_audio_events = int((duration_samples / SAMPLE_RATE) * random.uniform(5, 25)) # More events
    
    for i in range(num_audio_events):
        current_time_audio = (i / num_audio_events) * (duration_samples / SAMPLE_RATE)
        intensity_factor = intensity_function(current_time_audio)
        intensity_factor = np.clip(intensity_factor, 0.01, 5.0) # Ensure intensity_factor is positive

        event_start_sample = random.randint(0, max(0, duration_samples - SAMPLE_RATE // 10)) # Ensure event can fit
        
        # Shorter, sharper events, scaled by intensity
        event_duration = random.uniform(0.005, 0.15) * intensity_factor 
        event_duration = max(0.001, min(event_duration, 1.0)) # Clamp duration
        event_duration_samples = int(event_duration * SAMPLE_RATE)
        if event_duration_samples <= 0: continue # Skip if event is too short

        # Ensure event does not exceed total duration
        if event_start_sample + event_duration_samples > duration_samples:
            event_duration_samples = duration_samples - event_start_sample
            if event_duration_samples <= 0: continue


        noise_choice = random.choice(theme_data.get("audio_noise_types", ["white"]))
        event_vol = random.uniform(0.05, 0.5) * intensity_factor # Volume scaled by intensity
        event_vol = np.clip(event_vol, 0.01, 1.0) # Clip volume

        features = {"intensity": intensity_factor} # Pass intensity as a feature
        # Add more features based on theme if needed
        if "convolution_reverb_damp" in theme_data.get("audio_features", []) and random.random() < 0.1:
            features["reverb_ir"] = IR_DAMPED_SPACE
        elif "reverb" in theme_data.get("audio_features", []) and random.random() < 0.05:
            features["reverb_ir"] = IR_SMALL_ROOM


        segment_mono = generate_noise(noise_choice, event_duration_samples, event_vol, features)
        
        if len(segment_mono) == 0: continue

        # Apply audio features/effects from theme
        for feature_name in theme_data.get("audio_features", []):
            if len(segment_mono) == 0: break # Stop if segment becomes empty
            if feature_name == "stutter" and random.random() < 0.2 * intensity_factor:
                stutter_len = min(len(segment_mono)//2, SAMPLE_RATE // random.randint(50,200))
                if stutter_len > 0 and len(segment_mono) > stutter_len:
                    stutter_segment = segment_mono[:stutter_len]
                    num_repeats = random.randint(2,5)
                    original_len = len(segment_mono)
                    stuttered_part = np.tile(stutter_segment, num_repeats)
                    segment_mono = np.concatenate((stuttered_part, segment_mono[stutter_len:]))
                    if len(segment_mono) > original_len: segment_mono = segment_mono[:original_len]


            elif feature_name == "bitcrush" and random.random() < 0.3 * intensity_factor:
                segment_mono = apply_bitcrush(segment_mono, bit_depth_range=(3, 8 + int(intensity_factor)))
            elif feature_name == "distortion" and random.random() < 0.25 * intensity_factor:
                segment_mono = apply_distortion(segment_mono, intensity_range=(1.0 + intensity_factor, 5.0 + intensity_factor * 2))
            elif feature_name == "spectral_glitch" and random.random() < 0.15 * intensity_factor:
                segment_mono = apply_spectral_glitch(segment_mono, intensity=intensity_factor * 0.5)
            
            # Convolution reverb example (can be slow)
            if "reverb_ir" in features and random.random() < 0.5: # Apply if IR was chosen
                 segment_mono = apply_convolution_reverb(segment_mono, features["reverb_ir"])


        # Panning
        pan = random.uniform(-1, 1) # -1 full left, 0 center, 1 full right
        if "extreme_panning" in theme_data.get("audio_features", []) and random.random() < 0.4:
            pan = random.choice([-1, 1]) * random.uniform(0.8, 1.0) # Hard pan

        left_gain = (1 - pan) / 2
        right_gain = (1 + pan) / 2
        
        # Ensure gains are not negative due to float precision with pan around -1 or 1
        left_gain = max(0, min(1, left_gain))
        right_gain = max(0, min(1, right_gain))


        # Add segment to main audio data, ensuring bounds
        end_sample = event_start_sample + len(segment_mono)
        actual_len = len(segment_mono)
        if end_sample > duration_samples:
            actual_len = duration_samples - event_start_sample
            if actual_len <= 0: continue
            segment_mono = segment_mono[:actual_len]
            end_sample = duration_samples
        
        if actual_len > 0:
            audio_data_stereo[event_start_sample:end_sample, 0] += segment_mono * left_gain
            audio_data_stereo[event_start_sample:end_sample, 1] += segment_mono * right_gain

    # Normalize final audio to prevent clipping
    max_abs_val = np.max(np.abs(audio_data_stereo))
    if max_abs_val > 1.0:
        audio_data_stereo /= max_abs_val
    elif max_abs_val < 0.1 and max_abs_val > 1e-6 : # Boost if too quiet but not silent
        audio_data_stereo /= (max_abs_val * 2) # Gentle boost
        audio_data_stereo = np.clip(audio_data_stereo, -1.0, 1.0)


    # Convert to 16-bit PCM for WAV
    audio_pcm = (audio_data_stereo * 32767).astype(np.int16)
    
    temp_audio_file = "temp_audio_extreme.wav"
    with wave.open(temp_audio_file, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_pcm.tobytes())
    
    print(f"Audio generation complete: {temp_audio_file}")
    return temp_audio_file


# --- Main Video Generation Logic ---
def generate_video():
    actual_duration = random.uniform(MIN_DURATION, MAX_DURATION)
    actual_fps = random.randint(MIN_FPS, MAX_FPS)
    total_frames = int(actual_duration * actual_fps)
    actual_duration_samples = int(actual_duration * SAMPLE_RATE)

    # Choose a theme
    theme_name = random.choice(list(THEMES.keys()))
    current_theme_data = copy.deepcopy(THEMES[theme_name]) # Use a copy for potential modifications
    current_theme_data['name'] = theme_name # Store name for logging
    print(f"Selected Theme: {theme_name}")
    print(f"Generating video: {actual_duration:.2f}s @ {actual_fps} FPS ({total_frames} frames)")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4
    video_out = cv2.VideoWriter(OUTPUT_FILE, fourcc, actual_fps, (WIDTH, HEIGHT))

    # Shared intensity profile for visuals and audio cues
    # Use actual_duration_samples for audio-synced intensity, map to video time
    intensity_function = generate_intensity_profile(actual_duration_samples, SAMPLE_RATE)

    prev_frame = None # For effects like feedback or datamosh_sim

    # --- Visual Element Generation ---
    base_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) # Base black frame
    
    # Pre-generate some text elements to vary
    num_word_elements = random.randint(5, 15)
    text_elements = []
    for _ in range(num_word_elements):
        word = random.choice(current_theme_data["words"])
        if random.random() < current_theme_data.get("zalgo_words_chance", 0):
            word = zalgo_text(word)
        
        symbol_prefix = random.choice(current_theme_data["symbols"]) if current_theme_data["symbols"] and random.random() < 0.7 else ""
        symbol_suffix = random.choice(current_theme_data["symbols"]) if current_theme_data["symbols"] and random.random() < 0.7 else ""
        
        # Try to encode some words for variety
        if random.random() < 0.1: # Base64
            try: word = base64.b64encode(word.encode('utf-8')).decode('utf-8')[:random.randint(8,20)]
            except: pass
        elif random.random() < 0.05: # Hex
            try: word = binascii.hexlify(word.encode('utf-8')).decode('utf-8')[:random.randint(8,20)]
            except: pass

        display_text = f"{symbol_prefix}{word}{symbol_suffix}"
        font_scale = random.uniform(0.5, 3.5) # Wider range
        color_idx = random.randint(1, len(current_theme_data["colors"][0]) -1) # Avoid bg color for text
        text_color = random.choice(current_theme_data["colors"])[color_idx]
        thickness = random.randint(1, 4)
        pos_x = random.randint(WIDTH // 10, WIDTH - WIDTH // 5)
        pos_y = random.randint(HEIGHT // 10, HEIGHT - HEIGHT // 5)
        text_elements.append({"text": display_text, "scale": font_scale, "color": text_color, "thick": thickness, "pos": (pos_x, pos_y)})


    # --- Frame Generation Loop ---
    for frame_num in range(total_frames):
        current_time_video = frame_num / actual_fps
        # Get intensity for current video time (map from audio intensity timeline)
        intensity = intensity_function(current_time_video) # intensity_function expects time in seconds
        intensity = np.clip(intensity, 0.01, 5.0) # Ensure positive for multipliers

        # Start with a base color or simple gradient from theme
        palette = random.choice(current_theme_data["colors"])
        frame = base_frame.copy()
        frame[:,:] = palette[0] # Background color

        # Add some basic shapes or noise based on intensity
        if random.random() < 0.2 * intensity: # More activity with higher intensity
            num_shapes = random.randint(1, int(5 * intensity))
            for _ in range(num_shapes):
                shape_type = random.choice(["rect", "line", "circle"])
                color = random.choice(palette[1:]) # Foreground colors
                x1, y1 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
                x2, y2 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
                thick = random.randint(1, int(10 * intensity))
                if shape_type == "rect":
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, random.choice([-1, thick])) # Filled or outline
                elif shape_type == "line":
                    cv2.line(frame, (x1,y1), (x2,y2), color, thick)
                elif shape_type == "circle" and thick > 0: # Ensure thick > 0 for cv2.circle
                     cv2.circle(frame, (x1,y1), random.randint(5, int(100*intensity)), color, random.choice([-1, thick]))


        # Apply visual effects chosen from the theme
        # More effects applied or stronger effects with higher intensity
        num_effects_to_apply = random.randint(1, 2 + int(intensity)) 
        
        # Shuffle effects for variety each frame
        available_effects = current_theme_data.get("visual_effects", [])
        random.shuffle(available_effects)
        
        applied_effects_count = 0
        for effect_name in available_effects:
            if applied_effects_count >= num_effects_to_apply: break
            
            # Some effects might need prev_frame
            effect_func = globals().get(effect_name)
            if effect_func:
                # Check if function requires prev_frame
                sig = inspect.signature(effect_func)
                if 'prev_frame' in sig.parameters:
                    frame = effect_func(frame, prev_frame)
                else:
                    frame = effect_func(frame)
                applied_effects_count +=1


        # Overlay text elements, vary visibility/position slightly
        num_visible_texts = random.randint(0, 2 + int(intensity*2))
        visible_texts = random.sample(text_elements, min(num_visible_texts, len(text_elements)))
        
        for el in visible_texts:
            if random.random() < 0.05 * intensity : # Chance to flicker text off
                continue
            
            # Slight position jitter
            jit_x = random.randint(-int(10*intensity), int(10*intensity))
            jit_y = random.randint(-int(10*intensity), int(10*intensity))
            current_pos = (el["pos"][0] + jit_x, el["pos"][1] + jit_y)
            
            # Use FONT_FACE_TO_USE (Hershey) for cv2.putText
            try:
                cv2.putText(frame, el["text"], current_pos, FONT_FACE_TO_USE, 
                            el["scale"] * (0.8 + 0.4*random.random()), # Slight scale jitter
                            el["color"], el["thick"], cv2.LINE_AA)
            except Exception as e:
                # print(f"Error drawing text '{el['text']}': {e}") # In case of weird chars with Hershey
                pass


        video_out.write(frame)
        prev_frame = frame.copy() # Store for next iteration's feedback effects

        if frame_num % (actual_fps * 5) == 0: # Log progress every 5 seconds
            print(f"Video processing: Frame {frame_num}/{total_frames} ({current_time_video:.1f}s / {actual_duration:.1f}s)")

    video_out.release()
    print(f"Video generation complete: {OUTPUT_FILE}")
    print(f"Video generation yielded: Duration={actual_duration:.2f}s, FPS={actual_fps}")

    # --- Audio Generation ---
    print(f"Generating {actual_duration:.1f}s audio, Theme: {theme_name}...")
    # Pass actual_duration_samples for audio generation
    audio_temp_file = generate_audio_enhanced(actual_duration_samples, actual_fps, current_theme_data, intensity_function, {}) # Pass empty dict for global_params if not used

    # --- Combine Audio and Video using FFmpeg ---
    final_output_file = f"final_{OUTPUT_FILE}"
    # Ensure OUTPUT_FILE (video only) and audio_temp_file exist
    if os.path.exists(OUTPUT_FILE) and os.path.exists(audio_temp_file):
        ffmpeg_command = [
            'ffmpeg', '-y', # Overwrite output file if it exists
            '-i', OUTPUT_FILE,         # Input video
            '-i', audio_temp_file,     # Input audio
            '-c:v', 'copy',            # Copy video stream (no re-encoding)
            '-c:a', 'aac',             # Encode audio to AAC (common for MP4)
            '-b:a', '192k',            # Audio bitrate
            '-shortest',               # Finish encoding when the shortest input stream ends
            final_output_file
        ]
        print(f"Combining video and audio with FFmpeg: {' '.join(ffmpeg_command)}")
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            print(f"Successfully created final video: {final_output_file}")
            # Clean up intermediate files
            os.remove(OUTPUT_FILE)
            os.remove(audio_temp_file)
            print(f"Cleaned up temporary files.")
        except subprocess.CalledProcessError as e:
            print(f"Error during FFmpeg processing:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            print(f"Final video might not be created or complete. Video-only at {OUTPUT_FILE}, Audio-only at {audio_temp_file}")
        except FileNotFoundError:
            print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")
            print(f"Video-only file is at: {OUTPUT_FILE}")
            print(f"Audio-only file is at: {audio_temp_file}")

    else:
        print("Error: Video or audio file missing, cannot combine.")
        if os.path.exists(OUTPUT_FILE): print(f"Video-only file is at: {OUTPUT_FILE}")
        if os.path.exists(audio_temp_file): print(f"Audio-only file is at: {audio_temp_file}")


if __name__ == '__main__':
    # Ensure FFmpeg is available (optional check, actual error handled later)
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True)
        print("FFmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: FFmpeg not found or not working. Video and audio will be separate files.")
        print("Please install FFmpeg and add it to your PATH to combine them automatically.")

    generate_video()















