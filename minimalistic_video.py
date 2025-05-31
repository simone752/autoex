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
LOW_FREQ_CUTOFF = 18      # Allow slightly lower frequencies
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
    for char_val in text: # Renamed char to char_val to avoid conflict with char_set
        output += char_val
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
# FONT_TO_USE is for systems that might use TTF with PIL/Pillow, not directly with cv2.putText here.
FONT_TO_USE = SELECTED_TTF_FONT if SELECTED_TTF_FONT else DEFAULT_FONT 
if SELECTED_TTF_FONT:
    print(f"Selected TTF font (for potential use with PIL/FreeType): {SELECTED_TTF_FONT}")
else:
    print("No TTF fonts found or selected from the predefined paths.")

# For cv2.putText, we primarily use Hershey fonts due to dependency simplicity.
FONT_FACE_TO_USE = DEFAULT_FONT # This is what cv2.putText will use.
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
    # Corrected print statement to use scaled_profile for min/max display
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
    # Generating Perlin noise per pixel
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
    block_size_max_val = random.randint(bs_min, bs_max) 
    num_blocks = random.randint(*num_blocks_range)
    height, width = frame.shape[:2]
    temp_frame = frame.copy()

    for _ in range(num_blocks):
        bh = random.randint(bs_min, block_size_max_val)
        bw = random.randint(bs_min, block_size_max_val)
        
        if height - bh <= 0 or width - bw <= 0: continue 
        y = random.randint(0, height - bh -1) 
        x = random.randint(0, width - bw -1)  

        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)

        target_y_start = np.clip(y + shift_y, 0, height - bh)
        target_x_start = np.clip(x + shift_x, 0, width - bw)
        
        try:
            block = frame[y:y+bh, x:x+bw]
            temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block
        except ValueError: 
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
        # Using np.int16 for intermediate sum to prevent uint8 overflow before clipping
        modified_prev_int = modified_prev.astype(np.int16)
        modified_prev_int[:,:,0] += shift_b
        modified_prev_int[:,:,1] += shift_g
        modified_prev_int[:,:,2] += shift_r
        modified_prev = np.clip(modified_prev_int, 0, 255).astype(np.uint8)


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

    for y_pos in range(0, height, thickness * 2): 
        line_color_val = base_color_val + random.randint(-color_variation, color_variation)
        line_color = tuple(np.clip([line_color_val]*3, 0, 255).astype(np.uint8))
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
    alpha = random.uniform(*alpha_range) 
    beta = random.randint(*beta_range)   
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_datamosh_sim(frame, prev_frame, hold_prob=0.3, smear_prob=0.5, block_size_range=(32, 128)):
    if prev_frame is None: return frame
    height, width = frame.shape[:2]
    output = frame.copy()
    bs_min = max(16, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size = random.randint(bs_min, bs_max)

    if random.random() < hold_prob:
        num_hold_blocks = random.randint(1, max(1, (width * height) // (block_size**2) // 3 if block_size > 0 else 1)) 
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
        num_smear_blocks = random.randint(1, max(1, (width * height) // (block_size**2) // 4 if block_size > 0 else 1))
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
    chars = list(char_set) # Use the provided char_set parameter
    if invert: chars = chars[::-1]
    num_chars = len(chars)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            block_gray = gray[y:y+block_size, x:x+block_size] 
            avg_brightness = np.mean(block_gray)
            char_index = int(np.clip((avg_brightness / 255.0) * num_chars, 0, num_chars - 1))
            # Ensure char_index is valid for the 'chars' list
            char_to_draw = chars[char_index if num_chars > 0 else 0] # Renamed char to char_to_draw
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255)
            
            (text_w, text_h), _ = cv2.getTextSize(char_to_draw, FONT_FACE_TO_USE, font_scale, thickness)
            pos = (x + (block_size - text_w) // 2, y + (block_size + text_h) // 2)
            try:
                cv2.putText(output, char_to_draw, pos, FONT_FACE_TO_USE, font_scale, text_color, thickness, cv2.LINE_AA)
            except Exception: 
                pass
    return output

def apply_crt_ghost_sim(frame, prev_frame):
    if prev_frame is not None:
        return cv2.addWeighted(frame, 0.85, prev_frame, 0.15, 0)
    return frame

def apply_slit_scan_sim(frame):
    rows, cols, _ = frame.shape
    output = frame.copy()
    for r_idx in range(rows): 
        smear_amount = int(math.sin(r_idx * 0.1 + random.uniform(0, math.pi)) * random.uniform(5,15)) 
        output[r_idx, :] = np.roll(output[r_idx, :], smear_amount, axis=0) 
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
    if n_samples < 1024: return data.astype(np.float32) 

    data_float = data.astype(np.float32)
    spectrum = fft(data_float)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    n_freqs = len(spectrum)

    num_glitches = int(n_freqs * 0.05 * intensity * random.random())
    population_size = n_freqs // 2 -1
    if population_size <=0 : return data_float 

    k_sample = min(num_glitches, population_size)
    if k_sample <=0 : return data_float

    indices_to_glitch = random.sample(range(1, n_freqs // 2), k_sample) 

    for idx in indices_to_glitch:
        if random.random() < 0.6: 
            magnitude[idx] = 0
            if idx < n_freqs: magnitude[n_freqs - idx] = 0 
        else: 
            if idx + 1 < n_freqs // 2:
                phase[idx], phase[idx+1] = phase[idx+1], phase[idx]
                if n_freqs - idx < n_freqs : phase[n_freqs - idx] = -phase[idx] 
                if n_freqs - (idx+1) < n_freqs : phase[n_freqs - (idx+1)] = -phase[idx+1]

    new_spectrum = magnitude * np.exp(1j * phase)
    glitched_data = np.real(ifft(new_spectrum))

    max_abs_orig = np.max(np.abs(data_float)) if len(data_float) > 0 else 0
    max_abs_glitched = np.max(np.abs(glitched_data)) if len(glitched_data) > 0 else 0

    if max_abs_glitched > 1e-6 and max_abs_orig > 1e-6:
        glitched_data *= (max_abs_orig / max_abs_glitched)
    elif max_abs_glitched > 1e-6: 
        glitched_data /= max_abs_glitched
    
    return glitched_data.astype(data.dtype)


def apply_convolution_reverb(data, impulse_response):
    n_data = len(data)
    n_ir = len(impulse_response)

    if n_ir == 0 or n_data == 0: return data # Return original data if no impulse or no data

    impulse_response_eff = impulse_response
    if n_ir >= n_data : # If IR is too long or data is very short
        # Truncate IR to be shorter than data, or a minimum sensible length
        target_ir_len = max(1, n_data // 2 if n_data > 1 else 1)
        impulse_response_eff = impulse_response[:target_ir_len]
        if len(impulse_response_eff) == 0: return data # If IR becomes empty after truncation
    
    reverbed_data = np.convolve(data, impulse_response_eff, mode='same')

    max_abs_orig = np.max(np.abs(data)) 
    max_abs_rev = np.max(np.abs(reverbed_data)) if len(reverbed_data) > 0 else 0
    
    if max_abs_rev > 1e-6 and max_abs_orig > 1e-6:
        reverbed_data *= (max_abs_orig / max_abs_rev)
    elif max_abs_rev > 1e-6:
         reverbed_data /= max_abs_rev
         
    return reverbed_data

# Simple impulse responses for reverb (pre-normalized)
# Using the IMPULSE_RESPONSES dictionary structure from the user's latest paste
IMPULSE_RESPONSES = {
    "damp": np.exp(-np.linspace(0, 10, int(SAMPLE_RATE * 0.2))) * np.random.randn(int(SAMPLE_RATE * 0.2)), 
    "metal_hit": np.sin(2*np.pi*1500*np.linspace(0,0.1,int(SAMPLE_RATE*0.1))) * np.exp(-np.linspace(0, 20, int(SAMPLE_RATE * 0.1))), 
    "noise_burst": np.random.randn(int(SAMPLE_RATE * 0.1)) * np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * 0.1)))
}
# Normalize impulses to have max amplitude of 1
for k_ir in IMPULSE_RESPONSES: # Renamed k to k_ir
    if len(IMPULSE_RESPONSES[k_ir]) > 0: # Check if impulse is not empty
        max_abs = np.max(np.abs(IMPULSE_RESPONSES[k_ir]))
        if max_abs > 1e-6:
            IMPULSE_RESPONSES[k_ir] /= max_abs
    else: # Handle empty impulse responses if they occur from generation
        IMPULSE_RESPONSES[k_ir] = np.array([0.0]) # Replace with a single zero to avoid errors


# --- Tone and Noise Generation (Continuing from user's paste point) ---

def generate_tone(freq, duration_samples, vol, fm_chance=0.5, harmonic_chance=0.4): # Copied from existing artifact
    """Generates a tone with potential FM or harmonics and sharp envelope."""
    if duration_samples <= 0: return np.array([], dtype=np.float32) 
    t_vals = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False) # Renamed t to t_vals
    wave = np.sin(2 * np.pi * freq * t_vals) # Base sine wave

    # Frequency Modulation (FM)
    if random.random() < fm_chance:
        mod_freq = random.uniform(1, 20) * (1 + intensity_function(duration_samples / (2*SAMPLE_RATE)) * 2) # Mod freq scales with intensity
        mod_depth = random.uniform(0.1, 1.0) * freq * 0.5 # Mod depth relative to base freq
        wave = np.sin(2 * np.pi * (freq + mod_depth * np.sin(2 * np.pi * mod_freq * t_vals)) * t_vals)

    # Add Harmonics
    if random.random() < harmonic_chance:
        num_harmonics = random.randint(1,3)
        for i in range(2, num_harmonics + 2): # Add 2nd, 3rd, 4th harmonics
            wave += (random.random() * 0.5) * np.sin(2 * np.pi * freq * i * t_vals) / i
    
    # Sharp Attack/Decay Envelope
    attack_len = min(duration_samples // 20, SAMPLE_RATE // 100) # Very short attack
    decay_len = min(duration_samples // 5, SAMPLE_RATE // 20)   # Short decay
    
    if attack_len > 0 : wave[:attack_len] *= np.linspace(0, 1, attack_len)
    # Apply decay to the rest of the wave after attack
    if duration_samples > attack_len and decay_len > 0:
        decay_actual_len = min(decay_len, duration_samples - attack_len)
        wave[attack_len : attack_len+decay_actual_len] *= np.linspace(1, 0, decay_actual_len)
        if duration_samples > attack_len + decay_actual_len: # Zero out the rest if decay is shorter than remaining
            wave[attack_len+decay_actual_len:] = 0


    wave_normalized = wave / (np.max(np.abs(wave)) if np.max(np.abs(wave)) > 1e-6 else 1.0)
    return (wave_normalized * vol).astype(np.float32)


def generate_placeholder_noise(duration_samples, vol, features=None): # Copied from existing artifact
    """Generic placeholder for various noise types."""
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    return (np.random.uniform(-0.1, 0.1, duration_samples) * vol).astype(np.float32)

def generate_pink_noise(N, scale=1.0): # Copied from existing artifact
    """Generates pink noise (1/f power spectrum)."""
    if N <= 0:
        return np.array([], dtype=np.float64)
    
    white_noise = np.random.randn(N)
    fft_white = np.fft.rfft(white_noise) 
    num_freqs = len(fft_white)

    pink_filter = np.ones(num_freqs)
    if num_freqs > 1: 
        indices = np.arange(1, num_freqs) 
        pink_filter[1:] /= np.sqrt(indices) 
    
    fft_pink = fft_white * pink_filter
    pink_noise_samples = np.fft.irfft(fft_pink, n=N) 
    
    max_abs = np.max(np.abs(pink_noise_samples)) if N > 0 else 0
    if max_abs > 1e-6: 
        pink_noise_samples /= max_abs 
        
    return (pink_noise_samples * scale).astype(np.float32)

def generate_brown_noise(N, scale=1.0): # Copied from existing artifact
    """Generates brown noise (1/f^2 power spectrum) by integrating white noise."""
    if N <= 0:
        return np.array([], dtype=np.float64)
        
    white_noise = np.random.randn(N)
    brown_noise_samples = np.cumsum(white_noise)
    
    if N > 0 : brown_noise_samples -= np.mean(brown_noise_samples) 
    max_abs = np.max(np.abs(brown_noise_samples)) if N > 0 else 0
    if max_abs > 1e-6:
        brown_noise_samples /= max_abs 
        
    return (brown_noise_samples * scale).astype(np.float32)


def generate_noise(noise_choice, event_duration_samples, event_vol, features):
    """Generates a short segment of a specific noise type."""
    noise = None 
    if event_duration_samples <= 0:
        return np.array([], dtype=np.float32)

    if noise_choice == "white":
        noise = (np.random.rand(event_duration_samples) * 2 - 1) * event_vol
    elif noise_choice == "static":
        noise = np.zeros(event_duration_samples)
        # *** BUG FIX APPLIED HERE ***
        assign_len = min(100, event_duration_samples)
        if assign_len > 0: 
            noise[0:assign_len] = (np.random.rand(assign_len) - 0.5) * 0.1 * event_vol 
        if event_duration_samples > assign_len: # Add more continuous static for the rest
            noise[assign_len:] = (np.random.rand(event_duration_samples - assign_len) - 0.5) * 0.05 * event_vol

    elif noise_choice == "glitch":
        noise = np.zeros(event_duration_samples)
        num_clicks = random.randint(1, max(1, event_duration_samples // 200 if event_duration_samples >= 200 else 1))
        for _ in range(num_clicks):
            if event_duration_samples == 0: continue
            click_pos = random.randint(0, event_duration_samples - 1)
            noise[click_pos] = (random.random() * 2 - 1) * event_vol * 1.5 
    elif noise_choice == "digital_artifact":
        noise = np.zeros(event_duration_samples)
        glitch_len = random.randint(1, max(2, event_duration_samples // 10 if event_duration_samples >= 10 else 2))
        if event_duration_samples > glitch_len : # Check if there is enough space for glitch
            start_pos = random.randint(0, event_duration_samples - glitch_len -1 if event_duration_samples - glitch_len >0 else 0)
            glitch_val = (random.random() * 2 - 1) * event_vol
            noise[start_pos : start_pos + glitch_len] = glitch_val
            for i_glitch in range(start_pos, min(start_pos + glitch_len, event_duration_samples)): # Renamed i to i_glitch
                 if random.random() < 0.3: noise[i_glitch] *= random.uniform(-1,1)
        else: # Not enough space, generate placeholder
            noise = generate_placeholder_noise(event_duration_samples, event_vol, features)


    elif noise_choice == "screech":
        t_screech = np.linspace(0, event_duration_samples / SAMPLE_RATE, event_duration_samples, endpoint=False) # Renamed t
        freq = random.uniform(1000, HIGH_FREQ_CUTOFF / 1.5) * (1 + np.sin(t_screech * random.uniform(5,50)) * 0.5) 
        noise = np.sin(2 * np.pi * freq * t_screech) * event_vol * 0.7

    elif noise_choice == "digital_clipping_sim":
        noise = np.zeros(event_duration_samples)
        pulse_len = SAMPLE_RATE // random.randint(100, 500) 
        if pulse_len == 0: pulse_len = 1
        num_pulses = event_duration_samples // (pulse_len * 2) if pulse_len > 0 else 0
        for i_pulse in range(num_pulses): # Renamed i
            start = i_pulse * pulse_len * 2
            if start + pulse_len <= event_duration_samples:
                 noise[start : start + pulse_len] = random.choice([-1, 1]) * event_vol
    
    elif noise_choice == "feedback_screech":
        t_feedback = np.linspace(0, event_duration_samples / SAMPLE_RATE, event_duration_samples, endpoint=False) # Renamed t
        base_f = random.uniform(500, 3000)
        mod_f = random.uniform(10, 100) * np.sin(t_feedback * random.uniform(0.1, 5)) 
        freq = base_f + mod_f + (t_feedback * random.uniform(1000, 5000)) 
        freq = np.clip(freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF * 0.8)
        noise = np.sin(2 * np.pi * freq * t_feedback)
        noise = np.clip(noise * random.uniform(1.5, 5.0) * event_vol, -event_vol, event_vol)

    elif noise_choice == "pink":
        noise = generate_pink_noise(event_duration_samples, scale=event_vol)
    elif noise_choice == "brown":
        noise = generate_brown_noise(event_duration_samples, scale=event_vol)
    
    elif noise_choice in ["squelch", "heartbeat", "breathing", "wet_clicks", "sub_bass", 
                           "granular_flesh", "choking_sim", "modem_sim", "filtered_noise", 
                           "granular_synth", "tts_fragments"]: 
        noise = generate_placeholder_noise(event_duration_samples, event_vol, features)

    if noise is None:
        noise = generate_placeholder_noise(event_duration_samples, event_vol, features if 'features' in locals() else {})

    if not isinstance(noise, np.ndarray):
        noise = np.array(noise if noise is not None else [], dtype=np.float32)
    if len(noise) != event_duration_samples and event_duration_samples > 0 : 
        if len(noise) > event_duration_samples: noise = noise[:event_duration_samples]
        else: noise = np.pad(noise, (0, event_duration_samples - len(noise)), 'constant', constant_values=0)
    elif event_duration_samples == 0:
        return np.array([], dtype=np.float32)

    return noise.astype(np.float32)


def generate_audio_enhanced(duration_samples, fps, theme_data, intensity_func_ref, global_params=None): # Renamed intensity_function
    """Generates complex, layered audio based on theme and intensity."""
    print(f"Generating {duration_samples / SAMPLE_RATE:.1f}s audio, Theme: {theme_data.get('name', 'Unknown Theme')}...")
    global intensity_function # Declare that we are using the global intensity_function for generate_tone
    intensity_function = intensity_func_ref # Assign the passed function to the global scope for generate_tone

    audio_data_stereo = np.zeros((duration_samples, 2), dtype=np.float32)
    
    if "melody_scale" in theme_data:
        # Pass the intensity function correctly
        melody_intensity = intensity_func_ref(duration_samples / (SAMPLE_RATE * 2.0)) # Get intensity at midpoint of melody
        melody_samples = generate_melody_samples(duration_samples, theme_data, 
                                                 melody_intensity, 
                                                 SAMPLE_RATE)
        if len(melody_samples) == duration_samples:
            audio_data_stereo[:, 0] += melody_samples * 0.2 
            audio_data_stereo[:, 1] += melody_samples * 0.2 
        # else:
            # print(f"Warning: Melody length mismatch. Expected {duration_samples}, got {len(melody_samples)}")


    num_audio_events = int((duration_samples / SAMPLE_RATE) * random.uniform(5, 25)) 
    
    for i_event in range(num_audio_events): # Renamed i
        current_time_audio = (i_event / num_audio_events if num_audio_events > 0 else 0) * (duration_samples / SAMPLE_RATE)
        intensity_factor = intensity_func_ref(current_time_audio) # Use the passed intensity function
        intensity_factor = np.clip(intensity_factor, 0.01, 5.0) 

        event_start_sample = random.randint(0, max(0, duration_samples - (SAMPLE_RATE // 10) -1)) 
        
        event_duration = random.uniform(0.005, 0.15) * intensity_factor 
        event_duration = max(0.001, min(event_duration, 1.0)) 
        event_duration_samples = int(event_duration * SAMPLE_RATE)
        if event_duration_samples <= 0: continue 

        if event_start_sample + event_duration_samples > duration_samples:
            event_duration_samples = duration_samples - event_start_sample
            if event_duration_samples <= 0: continue


        noise_choice = random.choice(theme_data.get("audio_noise_types", ["white"]))
        event_vol = random.uniform(0.05, 0.5) * intensity_factor 
        event_vol = np.clip(event_vol, 0.01, 1.0) 

        features = {"intensity": intensity_factor} 
        # Use IMPULSE_RESPONSES dictionary
        if "convolution_reverb_damp" in theme_data.get("audio_features", []) and random.random() < 0.1:
            features["reverb_ir"] = IMPULSE_RESPONSES.get("damp", np.array([0.0]))
        elif "reverb" in theme_data.get("audio_features", []) and random.random() < 0.05: # Generic reverb might use a noise burst or similar
            features["reverb_ir"] = IMPULSE_RESPONSES.get("noise_burst", np.array([0.0]))


        segment_mono = generate_noise(noise_choice, event_duration_samples, event_vol, features)
        
        if len(segment_mono) == 0: continue

        for feature_name in theme_data.get("audio_features", []):
            if len(segment_mono) == 0: break 
            if feature_name == "stutter" and random.random() < 0.2 * intensity_factor:
                stutter_len = min(len(segment_mono)//2 if len(segment_mono)>0 else 0, SAMPLE_RATE // random.randint(50,200))
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
            
            if "reverb_ir" in features and random.random() < 0.5 and len(features["reverb_ir"]) > 0: 
                 segment_mono = apply_convolution_reverb(segment_mono, features["reverb_ir"])


        pan = random.uniform(-1, 1) 
        if "extreme_panning" in theme_data.get("audio_features", []) and random.random() < 0.4:
            pan = random.choice([-1, 1]) * random.uniform(0.8, 1.0) 

        left_gain = (1 - pan) / 2
        right_gain = (1 + pan) / 2
        
        left_gain = max(0, min(1, left_gain))
        right_gain = max(0, min(1, right_gain))

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

    max_abs_val = np.max(np.abs(audio_data_stereo)) if duration_samples > 0 else 0
    if max_abs_val > 1.0:
        audio_data_stereo /= max_abs_val
    elif max_abs_val < 0.1 and max_abs_val > 1e-6 : 
        audio_data_stereo /= (max_abs_val * 2) 
        audio_data_stereo = np.clip(audio_data_stereo, -1.0, 1.0)

    audio_pcm = (audio_data_stereo * 32767).astype(np.int16)
    
    temp_audio_file = "temp_audio_extreme.wav"
    try:
        with wave.open(temp_audio_file, 'w') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2) 
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_pcm.tobytes())
    except Exception as e:
        print(f"Error writing WAV file {temp_audio_file}: {e}")
        return None # Return None if WAV writing fails
    
    print(f"Audio generation complete: {temp_audio_file}")
    return temp_audio_file

# Global intensity_function for generate_tone, will be set in generate_audio_enhanced
intensity_function = None 

def generate_melody_samples(duration_samples, theme_data, current_intensity, sample_rate): # Added current_intensity
    """Generates a simple, possibly distorted, melodic sequence."""
    if duration_samples <= 0:
        return np.array([], dtype=np.float32)
        
    scale_name = theme_data.get("melody_scale", "minor_pentatonic")
    scale_intervals = SCALES.get(scale_name, SCALES["minor_pentatonic"])
    base_freq = random.uniform(50, 200) 
    # Make num_notes dependent on duration_samples to avoid issues with very short durations
    min_notes = 1
    max_notes = max(min_notes + 1, duration_samples // (sample_rate // 8)) # Ensure max_notes is at least min_notes + 1
    num_notes = random.randint(min_notes, max_notes)
    
    if num_notes == 0 : return np.zeros(duration_samples, dtype=np.float32)
    
    note_duration_samples = duration_samples // num_notes
    if note_duration_samples == 0: return np.zeros(duration_samples, dtype=np.float32)

    melody = np.array([], dtype=np.float32)
    for _ in range(num_notes):
        semitone_offset = random.choice(scale_intervals)
        octave_shift = random.choice([-1, 0, 0, 1]) * 12 
        current_freq = base_freq * (2**((semitone_offset + octave_shift) / 12.0))
        current_freq = np.clip(current_freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF / 2) 

        t_note = np.linspace(0, note_duration_samples / sample_rate, note_duration_samples, endpoint=False)
        note_wave = 0.3 * np.sin(2 * np.pi * current_freq * t_note) 

        attack_len = min(note_duration_samples // 10, sample_rate // 100) 
        decay_len = min(note_duration_samples // 5, sample_rate // 50)
        sustain_level = 0.7
        release_len = min(note_duration_samples // 10, sample_rate // 100)
        
        # Ensure lengths are positive
        attack_len = max(0, attack_len)
        decay_len = max(0, decay_len)
        release_len = max(0, release_len)

        if attack_len > 0 : note_wave[:attack_len] *= np.linspace(0, 1, attack_len)
        
        # Ensure indices are valid for sustain and release
        sustain_start = attack_len + decay_len
        release_start = note_duration_samples - release_len

        if decay_len > 0 and attack_len < sustain_start and sustain_start <= note_duration_samples :
             note_wave[attack_len : sustain_start] *= np.linspace(1, sustain_level, sustain_start - attack_len)
        
        if sustain_start < release_start and release_start <= note_duration_samples: 
            note_wave[sustain_start : release_start] *= sustain_level
        
        if release_len > 0 and release_start < note_duration_samples : 
            note_wave[release_start:] *= np.linspace(sustain_level, 0, note_duration_samples - release_start)


        melody = np.concatenate((melody, note_wave))

    distortion_intensity_val = theme_data.get("melody_distortion", 0) * current_intensity # Renamed
    if distortion_intensity_val > 0.1 and len(melody) > 0:
        melody = np.clip(melody * (1 + distortion_intensity_val), -0.98, 0.98) 

    if len(melody) > duration_samples:
        melody = melody[:duration_samples]
    elif len(melody) < duration_samples and duration_samples > 0 : 
        melody = np.pad(melody, (0, duration_samples - len(melody)), 'constant', constant_values=0)
    elif len(melody) == 0 and duration_samples > 0: 
        melody = np.zeros(duration_samples, dtype=np.float32)
        
    return melody.astype(np.float32)


# --- Main Video Generation Logic ---
def generate_video():
    global intensity_function # To be used by generate_tone via generate_audio_enhanced
    actual_duration = random.uniform(MIN_DURATION, MAX_DURATION)
    actual_fps = random.randint(MIN_FPS, MAX_FPS)
    total_frames = int(actual_duration * actual_fps)
    actual_duration_samples = int(actual_duration * SAMPLE_RATE)

    theme_name = random.choice(list(THEMES.keys()))
    current_theme_data = copy.deepcopy(THEMES[theme_name]) 
    current_theme_data['name'] = theme_name 
    print(f"Selected Theme: {theme_name}")
    print(f"Generating video: {actual_duration:.2f}s @ {actual_fps} FPS ({total_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_out = cv2.VideoWriter(OUTPUT_FILE, fourcc, actual_fps, (WIDTH, HEIGHT))

    # This is the primary intensity function for the whole video/audio
    # It's passed to generate_audio_enhanced, which then makes it available globally for generate_tone
    video_intensity_func = generate_intensity_profile(actual_duration_samples, SAMPLE_RATE)
    intensity_function = video_intensity_func # Set the global reference

    prev_frame = None 

    base_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8) 
    
    num_word_elements = random.randint(5, 15)
    text_elements = []
    for _ in range(num_word_elements):
        word = random.choice(current_theme_data["words"])
        if random.random() < current_theme_data.get("zalgo_words_chance", 0):
            word = zalgo_text(word)
        
        symbol_prefix = random.choice(current_theme_data["symbols"]) if current_theme_data["symbols"] and random.random() < 0.7 else ""
        symbol_suffix = random.choice(current_theme_data["symbols"]) if current_theme_data["symbols"] and random.random() < 0.7 else ""
        
        if random.random() < 0.1: 
            try: word = base64.b64encode(word.encode('utf-8')).decode('utf-8')[:random.randint(8,20)]
            except: pass
        elif random.random() < 0.05: 
            try: word = binascii.hexlify(word.encode('utf-8')).decode('utf-8')[:random.randint(8,20)]
            except: pass

        display_text = f"{symbol_prefix}{word}{symbol_suffix}"
        font_scale = random.uniform(0.5, 3.5) 
        
        # Ensure palette choice is valid
        current_palette_choices = current_theme_data["colors"]
        if not current_palette_choices: # Fallback if colors list is empty for some reason
            text_color_val = (255,255,255)
        else:
            chosen_palette = random.choice(current_palette_choices)
            if len(chosen_palette) > 1:
                 color_idx = random.randint(1, len(chosen_palette) -1) 
                 text_color_val = chosen_palette[color_idx] # Renamed text_color
            elif len(chosen_palette) == 1: # Only one color in palette
                 text_color_val = chosen_palette[0]
            else: # Empty palette
                 text_color_val = (255,255,255)


        thickness = random.randint(1, 4)
        pos_x = random.randint(WIDTH // 10, WIDTH - WIDTH // 5 if WIDTH // 5 > WIDTH // 10 else WIDTH // 10 +1)
        pos_y = random.randint(HEIGHT // 10, HEIGHT - HEIGHT // 5 if HEIGHT //5 > HEIGHT // 10 else HEIGHT //10 +1)
        text_elements.append({"text": display_text, "scale": font_scale, "color": text_color_val, "thick": thickness, "pos": (pos_x, pos_y)})


    for frame_num in range(total_frames):
        current_time_video = frame_num / actual_fps
        current_intensity_val = video_intensity_func(current_time_video) # Renamed intensity
        current_intensity_val = np.clip(current_intensity_val, 0.01, 5.0) 

        palette = random.choice(current_theme_data["colors"]) if current_theme_data["colors"] else [[(0,0,0)]] # Handle empty palette list
        frame = base_frame.copy()
        frame[:,:] = palette[0] if palette and palette[0] else (0,0,0)

        if random.random() < 0.2 * current_intensity_val: 
            num_shapes = random.randint(1, int(5 * current_intensity_val))
            for _ in range(num_shapes):
                shape_type = random.choice(["rect", "line", "circle"])
                # Ensure palette has foreground colors
                fg_colors = palette[1:] if len(palette) > 1 else [(255,255,255)] 
                color = random.choice(fg_colors)
                x1, y1 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
                x2, y2 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
                thick = random.randint(1, int(10 * current_intensity_val))
                if shape_type == "rect":
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, random.choice([-1, thick])) 
                elif shape_type == "line":
                    cv2.line(frame, (x1,y1), (x2,y2), color, thick)
                elif shape_type == "circle" and thick > 0: 
                     cv2.circle(frame, (x1,y1), random.randint(5, max(6, int(100*current_intensity_val))), color, random.choice([-1, thick]))


        num_effects_to_apply = random.randint(1, 2 + int(current_intensity_val)) 
        
        available_effects = current_theme_data.get("visual_effects", [])
        random.shuffle(available_effects)
        
        applied_effects_count = 0
        for effect_name in available_effects:
            if applied_effects_count >= num_effects_to_apply: break
            
            effect_func = globals().get(effect_name)
            if effect_func:
                sig = inspect.signature(effect_func)
                if 'prev_frame' in sig.parameters:
                    frame = effect_func(frame, prev_frame)
                else:
                    frame = effect_func(frame)
                applied_effects_count +=1


        num_visible_texts = random.randint(0, 2 + int(current_intensity_val*2))
        # Ensure sample size is not greater than population
        sample_k = min(num_visible_texts, len(text_elements))
        visible_texts = random.sample(text_elements, sample_k) if len(text_elements) > 0 else []
        
        for el in visible_texts:
            if random.random() < 0.05 * current_intensity_val : 
                continue
            
            jit_x = random.randint(-int(10*current_intensity_val), int(10*current_intensity_val))
            jit_y = random.randint(-int(10*current_intensity_val), int(10*current_intensity_val))
            current_pos = (el["pos"][0] + jit_x, el["pos"][1] + jit_y)
            
            try:
                cv2.putText(frame, el["text"], current_pos, FONT_FACE_TO_USE, 
                            el["scale"] * (0.8 + 0.4*random.random()), 
                            el["color"], el["thick"], cv2.LINE_AA)
            except Exception:
                pass


        video_out.write(frame)
        prev_frame = frame.copy() 

        if frame_num % (actual_fps * 5) == 0 and actual_fps > 0 : 
            print(f"Video processing: Frame {frame_num}/{total_frames} ({current_time_video:.1f}s / {actual_duration:.1f}s)")

    video_out.release()
    print(f"Video generation complete: {OUTPUT_FILE}")
    print(f"Video generation yielded: Duration={actual_duration:.2f}s, FPS={actual_fps}")

    print(f"Generating {actual_duration:.1f}s audio, Theme: {theme_name}...")
    audio_temp_file = generate_audio_enhanced(actual_duration_samples, actual_fps, current_theme_data, video_intensity_func, {}) 

    final_output_file = f"final_{OUTPUT_FILE}"
    if audio_temp_file and os.path.exists(OUTPUT_FILE) and os.path.exists(audio_temp_file): # Check if audio_temp_file is not None
        ffmpeg_command = [
            'ffmpeg', '-y', 
            '-i', OUTPUT_FILE,         
            '-i', audio_temp_file,     
            '-c:v', 'copy',            
            '-c:a', 'aac',             
            '-b:a', '192k',            
            '-shortest',               
            final_output_file
        ]
        print(f"Combining video and audio with FFmpeg: {' '.join(ffmpeg_command)}")
        try:
            process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            print(f"Successfully created final video: {final_output_file}")
            os.remove(OUTPUT_FILE)
            os.remove(audio_temp_file)
            print(f"Cleaned up temporary files.")
        except subprocess.CalledProcessError as e:
            print(f"Error during FFmpeg processing:")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return code: {e.returncode}")
            print(f"Stdout: {e.stdout}") # Ensure stdout and stderr are printed
            print(f"Stderr: {e.stderr}")
            print(f"Final video might not be created or complete. Video-only at {OUTPUT_FILE}, Audio-only at {audio_temp_file}")
        except FileNotFoundError:
            print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")
            print(f"Video-only file is at: {OUTPUT_FILE}")
            print(f"Audio-only file is at: {audio_temp_file}")
    else:
        print("Error: Video or audio file missing (or audio generation failed), cannot combine.")
        if os.path.exists(OUTPUT_FILE): print(f"Video-only file is at: {OUTPUT_FILE}")
        if audio_temp_file and os.path.exists(audio_temp_file): print(f"Audio-only file is at: {audio_temp_file}")
        elif not audio_temp_file: print("Audio generation failed, no temporary audio file was created.")


if __name__ == '__main__':
    try:
        # Check FFmpeg version and capture output to suppress it unless there's an error
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, text=True)
        print("FFmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("Warning: FFmpeg not found or not working. Video and audio will be separate files.")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"FFmpeg error: {e.stderr}")
        print("Please install FFmpeg and add it to your PATH to combine them automatically.")

    generate_video()
















