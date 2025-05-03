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

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720 # Increased resolution for better quality
MIN_DURATION, MAX_DURATION = 15, 40 # Slightly shorter range, focus on impact
# << INCREASED FPS RANGE >>
MIN_FPS, MAX_FPS = 15, 30 # Higher FPS for smoother motion, but still allowing some glitchiness
OUTPUT_FILE = "avant_garde_output.mp4"
HIGH_FREQ_CUTOFF = 22000 # Allow slightly higher frequencies
LOW_FREQ_CUTOFF = 18      # Allow slightly lower frequencies
SAMPLE_RATE = 44100
time_step = 1.0 / SAMPLE_RATE

# --- Enhanced Thematic Elements ---
# Added new themes and expanded existing ones with more abrasive elements

# Helper function for Zalgo text (more abrasive text)
def zalgo_text(text):
    zalgo_chars = [chr(i) for i in range(0x0300, 0x036F + 1)] # Combining Diacritical Marks
    output = ""
    for char in text:
        output += char
        num_zalgo = random.randint(1, 8) # Add more chaos
        for _ in range(num_zalgo):
            output += random.choice(zalgo_chars)
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
        "visual_effects": ["perlin", "block_shift", "color_shift", "warp", "pixelation", "scanlines", "contrast", "feedback", "datamosh_sim", "ascii_sim"], # Added new effects
        "audio_freq_range": (30, 18000),
        "audio_noise_types": ["white", "glitch", "static", "digital_artifact", "screech", "digital_clipping_sim", "feedback_screech"], # Added harsher noises
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
        "visual_effects": ["perlin", "block_shift", "warp", "feedback", "crt_ghost_sim", "contrast", "solarize", "slit_scan_sim"], # Different effect focus
        "audio_freq_range": (18, 2000), # Even lower floor
        "audio_noise_types": ["brown", "squelch", "heartbeat", "breathing", "wet_clicks", "sub_bass", "granular_flesh", "choking_sim"], # Added granular/choking
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
        "visual_effects": ["perlin", "warp", "feedback", "color_shift", "vector_field_sim", "ascii_sim", "contrast", "pixelation"],
        "audio_freq_range": (50, 16000),
        "audio_noise_types": ["pink", "static", "digital_artifact", "modem_sim", "filtered_noise", "granular_synth"],
        "audio_features": ["stutter", "bitcrush", "delay", "reverb", "extreme_panning", "spectral_glitch", "tts_fragments"], # Added TTS
        "melody_scale": "chromatic", # All notes possible
        "melody_distortion": 5.0, # Less harsh distortion
    },
    # Add more themes here based on brainstorming (Quantum Foam, Urban Signal Ghosts etc.)
}

# --- Scales for Buried Melody ---
SCALES = {
    "minor_pentatonic": [0, 3, 5, 7, 10], # Root, m3, p4, p5, m7
    "phrygian": [0, 1, 3, 5, 7, 8, 10], # Root, m2, m3, p4, p5, m6, m7 (dissonant)
    "chromatic": list(range(12)), # All notes
    "major": [0, 2, 4, 5, 7, 9, 11],
    # Add more scales as needed
}

# --- Font Configuration (Optional - Ensure good Unicode support) ---
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux
    "/System/Library/Fonts/Arial Unicode.ttf", # MacOS
    "C:/Windows/Fonts/arialuni.ttf", # Windows (if installed)
    "C:/Windows/Fonts/arial.ttf",
    # Add path to a specific broad Unicode font if available
    # "NotoSans-Regular.ttf",
]
AVAILABLE_FONTS = [f for f in FONT_PATHS if os.path.exists(f)]
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX # Fallback
SELECTED_TTF_FONT = None
if AVAILABLE_FONTS:
     # Try to find a font with broad support if possible, otherwise pick one
    broad_fonts = [f for f in AVAILABLE_FONTS if 'unicode' in f.lower() or 'dejavu' in f.lower() or 'noto' in f.lower()]
    if broad_fonts:
         SELECTED_TTF_FONT = random.choice(broad_fonts)
    else:
         SELECTED_TTF_FONT = random.choice(AVAILABLE_FONTS)

print(f"Available TTF fonts found: {AVAILABLE_FONTS}")
# Use TTF font if found, otherwise default. TTF is better for special symbols.
FONT_TO_USE = SELECTED_TTF_FONT if SELECTED_TTF_FONT else DEFAULT_FONT
# Note: Using TTF fonts with OpenCV's putText requires FreeType support compiled in OpenCV.
# If TTF fails, the script might default to SIMPLEX or error.
# For simplicity in this example, we'll mostly stick to DEFAULT_FONT, but TTF is preferred.
print(f"Font selected for rendering (prefer TTF if available/compatible): {FONT_TO_USE}")
# If using TTF: font_face = cv2.freetype.createFreeType2()
# font_face.loadFontData(fontFileName=FONT_TO_USE, id=0)
# font_face.putText(...)
# For now, we stick to default font for broader compatibility without FreeType hassle.
FONT_FACE_TO_USE = DEFAULT_FONT


# Initialize pygame mixer
try:
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=2048) # Larger buffer maybe?
except pygame.error as e:
    print(f"Pygame mixer init failed (non-critical for file output): {e}")


# --- Intensity Profile Generation (Unchanged) ---
def generate_intensity_profile(duration_samples, sample_rate, scale=25.0, octaves=5, persistence=0.55, lacunarity=2.2):
    """Generates a more dynamic Perlin noise-based intensity profile."""
    duration_seconds = duration_samples / sample_rate
    num_points = int(duration_seconds * 15) # More points for finer control
    if num_points < 2: num_points = 2
    profile = np.zeros(num_points)
    base = random.randint(0, 500) # Wider seed range

    for i in range(num_points):
        profile[i] = noise.pnoise1(i / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base)

    min_val, max_val = np.min(profile), np.max(profile)
    if max_val - min_val > 1e-6:
        normalized_profile = (profile - min_val) / (max_val - min_val)
        # << MORE ABRASIVE: Wider intensity range >>
        intensity_min = 0.1 # Allow deeper lows
        intensity_max = 2.5 # Allow higher peaks
        scaled_profile = intensity_min + normalized_profile * (intensity_max - intensity_min)
    else:
        scaled_profile = np.full(num_points, 1.0)

    profile_times = np.linspace(0, duration_seconds, num_points)
    intensity_func = interp1d(profile_times, scaled_profile, kind='cubic', bounds_error=False, # Smoother cubic interpolation
                              fill_value=(scaled_profile[0], scaled_profile[-1]))
    print(f"Generated shared intensity profile: {len(profile)} points over {duration_seconds:.1f}s (Range: {intensity_min:.2f}-{intensity_max:.2f})")
    return intensity_func


# --- Visual Glitch Functions (Optimized & New Additions) ---

# (Keep apply_perlin_noise, apply_block_shift, apply_color_channel_shift as they are reasonably efficient)
def apply_perlin_noise(frame, alpha_range=(0.1, 0.9), scale_range=(2.0, 80.0), oct_range=(3, 9)): # Wider ranges
    alpha = random.uniform(*alpha_range)
    scale = random.uniform(*scale_range)
    octaves = random.randint(*oct_range)
    persistence = random.uniform(0.2, 0.7) # More variability
    lacunarity = random.uniform(1.8, 3.5) # More variability
    seed = random.randint(0, 2000)
    height, width = frame.shape[:2]
    # Generate noise coordinates using meshgrid for potential vectorization (though pnoise2 isn't directly vectorized)
    x_coords = np.arange(width) / scale
    y_coords = np.arange(height) / scale
    # Direct loop remains necessary for noise.pnoise2 currently
    gray_noise = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            gray_noise[i, j] = noise.pnoise2(y_coords[i], x_coords[j], octaves=octaves,
                                             persistence=persistence, lacunarity=lacunarity, base=seed)
    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX)
    colored_noise = cv2.cvtColor(colored_noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if random.random() < 0.4: colored_noise = 255 - colored_noise # Higher invert chance
    return cv2.addWeighted(frame, 1 - alpha, colored_noise, alpha, 0)

def apply_block_shift(frame, max_shift_range=(10, WIDTH // 3), block_size_range=(5, HEIGHT // 2), num_blocks_range=(20, 100)): # More blocks, wider ranges
    max_shift = random.randint(*max_shift_range)
    block_size_max = random.randint(*block_size_range)
    num_blocks = random.randint(*num_blocks_range)
    height, width = frame.shape[:2]
    temp_frame = frame.copy()
    for _ in range(num_blocks):
        bh = random.randint(3, block_size_max) # Smaller min block size
        bw = random.randint(3, block_size_max)
        if height - bh <= 0 or width - bw <= 0: continue
        y = random.randint(0, height - bh - 1)
        x = random.randint(0, width - bw - 1)
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        target_y_start = np.clip(y + shift_y, 0, height - bh)
        target_x_start = np.clip(x + shift_x, 0, width - bw)
        try: # Add try-except for potential slicing issues with extreme values
            block = frame[y:y+bh, x:x+bw]
            temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block
        except ValueError as e:
            # print(f"Block shift slice error: {e}, skipping block.") # Optional debug
            pass
    return temp_frame

def apply_color_channel_shift(frame, max_shift_range=(5, 60)): # Wider shift range
    max_shift = random.randint(*max_shift_range)
    temp_frame = frame.copy()
    height, width = frame.shape[:2]
    for i in range(3):
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        channel = frame[:,:,i]
        shifted_channel = np.roll(np.roll(channel, shift_y, axis=0), shift_x, axis=1)
        temp_frame[:,:,i] = shifted_channel
    return temp_frame

# apply_warp: Using remap is already efficient. Increase amplitude/frequency ranges.
def apply_warp(frame, amplitude_range=(5, 80), freq_range=(0.002, 0.1)): # Wider ranges
    rows, cols = frame.shape[:2]
    amplitude_x = random.uniform(*amplitude_range)
    frequency_x = random.uniform(*freq_range)
    amplitude_y = random.uniform(*amplitude_range) * random.uniform(0.3, 1.7) # More variation
    frequency_y = random.uniform(*freq_range) * random.uniform(0.3, 1.7)
    phase_x = random.uniform(0, 2 * math.pi)
    phase_y = random.uniform(0, 2 * math.pi)
    # Generate meshgrid for coordinates
    x_mesh, y_mesh = np.meshgrid(np.arange(cols), np.arange(rows))
    # Calculate offsets using numpy operations for potential speedup
    offset_x = (amplitude_x * np.sin(2 * math.pi * y_mesh * frequency_x + phase_x)).astype(np.float32)
    offset_y = (amplitude_y * np.cos(2 * math.pi * x_mesh * frequency_y + phase_y)).astype(np.float32) # Use x_mesh here? Cosine of j
    # Calculate source map, clipping included
    map_x = np.clip(x_mesh.astype(np.float32) + offset_x, 0, cols - 1)
    map_y = np.clip(y_mesh.astype(np.float32) + offset_y, 0, rows - 1)
    # Apply remap
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# apply_feedback: Increase alpha range for stronger feedback
def apply_feedback(frame, prev_frame, alpha_range=(0.02, 0.6)): # Higher max alpha
    if prev_frame is None: return frame
    alpha = random.uniform(*alpha_range)
    modified_prev = prev_frame
    # Add more aggressive random transformations
    if random.random() < 0.2: # Increased chance
        rows, cols = modified_prev.shape[:2]
        angle = random.uniform(-5, 5) # Wider angle
        scale = random.uniform(0.95, 1.05) # Wider scale
        tx = random.uniform(-10, 10)
        ty = random.uniform(-10, 10)
        center = (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx # Add translation
        M[1, 2] += ty
        modified_prev = cv2.warpAffine(modified_prev, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    # Maybe add a chance for slight color shift in feedback?
    if random.random() < 0.1:
         modified_prev = cv2.add(modified_prev, (random.randint(-10,10),random.randint(-10,10),random.randint(-10,10), 0))

    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)

# apply_pixelation: Efficient resizing. Increase block size range.
def apply_pixelation(frame, block_size_range=(4, 96)): # Wider range
    height, width = frame.shape[:2]
    block_size = random.randint(*block_size_range)
    block_size = max(2, block_size)
    pixel_w, pixel_h = max(1, width // block_size), max(1, height // block_size)
    temp = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

# apply_scanlines: Efficient. Increase intensity range.
def apply_scanlines(frame, intensity_range=(0.4, 1.0), thickness_range=(1, 4), color_variation=40): # Higher max intensity, thickness
    intensity = random.uniform(*intensity_range)
    thickness = random.randint(*thickness_range)
    temp_frame = np.zeros_like(frame)
    height, width = frame.shape[:2]
    base_color_val = random.randint(0, 70) # Slightly brighter dark lines possible
    for y in range(0, height, thickness * 2):
        line_color_val = base_color_val + random.randint(-color_variation, color_variation)
        line_color = tuple(np.clip([line_color_val]*3, 0, 255))
        cv2.line(temp_frame, (0, y), (width, y), line_color, thickness)
    # More aggressive blend
    return cv2.addWeighted(frame, 1.0, temp_frame, intensity * 1.1, -int(255 * intensity * 0.3))

# apply_solarize: Efficient. Keep as is or slightly widen threshold.
def apply_solarize(frame, threshold_range=(60, 200)): # Wider range
    threshold = random.randint(*threshold_range)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    solarized_frame = frame.copy()
    solarized_frame[mask == 255] = 255 - solarized_frame[mask == 255]
    return solarized_frame

# apply_extreme_contrast: Efficient. Increase alpha/beta ranges.
def apply_extreme_contrast(frame, alpha_range=(1.0, 6.0), beta_range=(-100, 100)): # Much wider ranges
    alpha = random.uniform(*alpha_range)
    beta = random.randint(*beta_range)
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# << NEW VISUAL EFFECTS >>
def apply_datamosh_sim(frame, prev_frame, hold_prob=0.3, smear_prob=0.5, block_size_range=(32, 128)):
    """Simulates datamoshing effects like I-frame holds and P-frame smearing."""
    if prev_frame is None: return frame
    height, width = frame.shape[:2]
    output = frame.copy()
    block_size = random.randint(*block_size_range)

    # I-Frame Hold Simulation
    if random.random() < hold_prob:
        num_hold_blocks = random.randint(1, (width // block_size) * (height // block_size) // 3)
        for _ in range(num_hold_blocks):
            bx = random.randint(0, width // block_size - 1) * block_size
            by = random.randint(0, height // block_size - 1) * block_size
            output[by:by+block_size, bx:bx+block_size] = prev_frame[by:by+block_size, bx:bx+block_size]

    # P-Frame Smear Simulation
    if random.random() < smear_prob:
        num_smear_blocks = random.randint(1, (width // block_size) * (height // block_size) // 4)
        for _ in range(num_smear_blocks):
            bx = random.randint(0, width // block_size - 1) * block_size
            by = random.randint(0, height // block_size - 1) * block_size
            # Simulate motion vector
            vx = random.randint(-block_size // 2, block_size // 2)
            vy = random.randint(-block_size // 2, block_size // 2)
            # Smear the block from previous frame
            smear_block = prev_frame[by:by+block_size, bx:bx+block_size]
            # Simple smear: just repeat edge pixels (crude but fast)
            if abs(vx) > abs(vy): # Horizontal smear
                edge = smear_block[:, -1 if vx > 0 else 0, :]
                for k in range(1, abs(vx)):
                    target_x = bx + (k * np.sign(vx))
                    if 0 <= target_x < width - bw: # Check bounds
                         output[by:by+block_size, target_x:target_x+bw] = edge # Incorrect slicing here needs fix
                         # TODO: Fix smearing logic - this part is tricky to get right efficiently
                         pass # Placeholder fix
            else: # Vertical smear
                 # TODO: Implement vertical smearing logic
                 pass # Placeholder fix
            # A simpler smear: just copy the block with offset
            target_y = np.clip(by + vy, 0, height - block_size)
            target_x = np.clip(bx + vx, 0, width - block_size)
            output[target_y:target_y+block_size, target_x:target_x+block_size] = smear_block


    return output

def apply_ascii_sim(frame, block_size_range=(8, 16), char_set=" .:-=+*#%@", invert=False):
    """Overlays a crude ASCII representation."""
    height, width = frame.shape[:2]
    block_size = random.randint(*block_size_range)
    output = frame.copy()
    font_scale = block_size / 15.0 # Adjust font scale based on block size
    thickness = 1
    chars = list(char_set)
    if invert: chars = chars[::-1]
    num_chars = len(chars)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            avg_brightness = np.mean(block)
            char_index = int(np.clip((avg_brightness / 255.0) * num_chars, 0, num_chars - 1))
            char = chars[char_index]
            # Draw character - choose contrasting color
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255)
            # Position text roughly in the center of the block
            pos = (x + block_size // 4, y + block_size * 3 // 4)
            try:
                cv2.putText(output, char, pos, FONT_FACE_TO_USE, font_scale, text_color, thickness, cv2.LINE_AA)
            except Exception as e:
                 # print(f"ASCII sim text error: {e}") # Optional debug
                 pass # Ignore if font fails
    return output

# Add other new visual effects (CRT Ghost, Vector Field, Slit Scan) similarly...


# --- Audio Generation & Effects (Optimized & New Additions) ---

# (Keep apply_distortion, apply_bitcrush as they are)

# << NEW AUDIO EFFECTS >>
def apply_spectral_glitch(data, intensity=0.5):
    """Applies glitches in the frequency domain."""
    if len(data) < 1024: return data # Need enough data for FFT
    spectrum = fft(data)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    n = len(spectrum)

    # Glitch: Randomly zero out frequency bins or swap phases
    num_glitches = int(n * 0.1 * intensity * random.random()) # More glitches with intensity
    for _ in range(num_glitches):
        idx = random.randint(1, n // 2) # Affect positive frequencies
        if random.random() < 0.5: # Zero out bin
             magnitude[idx] = 0
             if idx < n: magnitude[n - idx] = 0 # Mirror for real signal
        else: # Swap phases of adjacent bins (causes phase smearing)
             if idx + 1 < n // 2:
                  phase[idx], phase[idx+1] = phase[idx+1], phase[idx]
                  # Mirror phase swaps (conjugate symmetry)
                  if n - idx > 0: phase[n - idx] = -phase[idx]
                  if n - (idx+1) > 0 : phase[n - (idx+1)] = -phase[idx+1]

    # Reconstruct signal
    new_spectrum = magnitude * np.exp(1j * phase)
    glitched_data = np.real(ifft(new_spectrum))

    # Normalize to prevent clipping after IFFT
    max_abs = np.max(np.abs(glitched_data))
    if max_abs > 1e-6: glitched_data /= max_abs

    return glitched_data * np.max(np.abs(data)) # Restore original amplitude scale roughly

def apply_convolution_reverb(data, impulse_response):
    """Applies reverb using convolution (can be slow for long impulses)."""
    # Ensure impulse response is shorter than data
    if len(impulse_response) >= len(data):
        impulse_response = impulse_response[:len(data)//2]

    reverbed_data = np.convolve(data, impulse_response, mode='same') # 'same' keeps original length

    # Normalize - convolution can change amplitude significantly
    max_abs_orig = np.max(np.abs(data))
    max_abs_rev = np.max(np.abs(reverbed_data))
    if max_abs_rev > 1e-6 and max_abs_orig > 1e-6:
        reverbed_data *= (max_abs_orig / max_abs_rev) # Try to match original peak level

    return reverbed_data

# Simple impulse responses for reverb
IMPULSE_RESPONSES = {
    "damp": np.exp(-np.linspace(0, 10, int(SAMPLE_RATE * 0.2))) * np.random.randn(int(SAMPLE_RATE * 0.2)), # Short, noisy decay
    "metal_hit": np.sin(2*np.pi*1500*np.linspace(0,0.1,int(SAMPLE_RATE*0.1))) * np.exp(-np.linspace(0, 20, int(SAMPLE_RATE * 0.1))), # Metallic ring
    "noise_burst": np.random.randn(int(SAMPLE_RATE * 0.1)) * np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * 0.1)))
}
IMPULSE_RESPONSES["damp"] /= np.max(np.abs(IMPULSE_RESPONSES["damp"])) # Normalize impulses
IMPULSE_RESPONSES["metal_hit"] /= np.max(np.abs(IMPULSE_RESPONSES["metal_hit"]))
IMPULSE_RESPONSES["noise_burst"] /= np.max(np.abs(IMPULSE_RESPONSES["noise_burst"]))


# --- Melody Generation ---
def generate_melody_fragment(duration_samples, vol, theme_data):
    """Generates a short, simple melodic fragment based on theme scale."""
    scale_name = theme_data.get("melody_scale", "minor_pentatonic")
    scale_intervals = SCALES.get(scale_name, SCALES["minor_pentatonic"])
    base_freq = random.uniform(80, 440) # Base note frequency (A2 to A4)
    notes_in_fragment = random.randint(3, 8)
    note_duration_samples = duration_samples // notes_in_fragment

    melody = np.zeros(duration_samples)
    current_pos = 0

    for _ in range(notes_in_fragment):
        if current_pos >= duration_samples: break
        # Choose note from scale
        scale_degree = random.choice(scale_intervals)
        octave_shift = random.choice([0, 0, 0, 1, -1]) # Chance to jump octave
        note_freq = base_freq * (2**((scale_degree + octave_shift * 12) / 12.0))
        note_freq = np.clip(note_freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)

        # Generate tone for the note
        actual_note_len = min(note_duration_samples, duration_samples - current_pos)
        if actual_note_len <= 0: break
        # Use simpler tone gen for melody? Or same complex one? Let's use complex for texture.
        note_tone = generate_tone(note_freq, actual_note_len, vol, fm_chance=0.2, harmonic_chance=0.6) # More harmonics

        melody[current_pos : current_pos + actual_note_len] = note_tone
        current_pos += actual_note_len

    # Apply heavy distortion specific to melody
    distortion_amount = theme_data.get("melody_distortion", 10.0)
    melody = apply_distortion(melody, intensity_range=(distortion_amount * 0.8, distortion_amount * 1.2))
    # Maybe add bitcrushing too?
    if random.random() < 0.5:
         melody = apply_bitcrush(melody, bit_depth_range=(3, 7))

    return melody


# (Keep generate_tone, generate_noise - update generate_noise with new types if needed)
# ... Ensure generate_noise handles new types like "digital_clipping_sim", "feedback_screech", "granular_flesh" etc. ...
# Example addition to generate_noise:
    # elif noise_type == "digital_clipping_sim":
    #     # Generate loud noise/tone and clip it harshly
    #     base_sound = generate_tone(random.uniform(100, 5000), duration_samples, vol * 5.0) # Generate loud sound
    #     return np.clip(base_sound, -0.99, 0.99) # Hard clip


# --- Frame Generation (Updated for new effects, abrasiveness) ---
def generate_frames_enhanced(theme_data, intensity_func, global_params):
    """Generates video frames, more abrasive, faster, uses new effects."""
    DURATION = random.randint(MIN_DURATION, MAX_DURATION)
    FPS = random.randint(MIN_FPS, MAX_FPS) # Using updated higher FPS range
    FRAME_COUNT = DURATION * FPS

    # Use the passed global_params which might have changed due to instability
    current_params = global_params["visual"]
    instability_chance = global_params["instability_chance"] * 1.5 # Increase instability impact

    print(f"Generating {DURATION}s video ({FPS} FPS), Theme: {global_params['theme_name']}...")
    print(f"Instability Chance: {instability_chance:.4f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_temp_path = f"video_temp_{int(time.time())}.mp4" # Unique temp name
    video = cv2.VideoWriter(video_temp_path, fourcc, FPS, (WIDTH, HEIGHT))

    prev_frame = None
    frame_hold_counter = 0
    held_frame = None
    effect_history = [] # Track recent effects to avoid too much repetition

    # Persistent texts - make more chaotic
    persistent_texts = []
    num_persistent = random.randint(5, 15) # More text elements
    for _ in range(num_persistent):
        word = random.choice(theme_data["words"])
        symbol = random.choice(theme_data["symbols"])
        # Higher chance for Zalgo text based on theme
        if random.random() < theme_data.get("zalgo_words_chance", 0.3):
             word = zalgo_text(word)
        if random.random() < 0.2: word = base64.b64encode(word.encode()).decode()
        elif random.random() < 0.15: word = binascii.hexlify(word.encode()).decode()

        persistent_texts.append({
            "text": word + symbol,
            "pos": (random.randint(-50, WIDTH - 50), random.randint(-50, HEIGHT - 20)), # Allow off-screen start
            "font_size": random.uniform(0.5, 4.5), # Wider size range
            "color": random.choice(random.choice(theme_data["colors"])), # Use theme colors
            "lifetime": random.randint(int(FPS * 0.1), int(FPS * 8)), # Wider lifetime
            "frame_count": 0,
            "move_speed": (random.uniform(-8, 8), random.uniform(-6, 6)), # Faster movement
            "rotation_speed": random.uniform(-5, 5) # Add rotation
        })

    # --- Main Frame Loop ---
    for i in range(FRAME_COUNT):
        current_time_sec = i / FPS
        current_intensity = intensity_func(current_time_sec)
        current_intensity = np.clip(current_intensity, 0.05, 3.0) # Wider intensity clamp

        # --- Parameter Instability Check (Visual) ---
        if random.random() < instability_chance:
            param_key = random.choice(list(current_params.keys()))
            if param_key == "effect_probabilities":
                effect_func_name = random.choice(list(current_params["effect_probabilities"].keys()))
                old_prob = current_params["effect_probabilities"][effect_func_name]
                new_prob = np.clip(old_prob + random.uniform(-0.3, 0.3), 0.01, 0.99) # Wider change
                current_params["effect_probabilities"][effect_func_name] = new_prob
            elif isinstance(current_params[param_key], tuple) and len(current_params[param_key]) == 2:
                current_min, current_max = current_params[param_key]
                range_width = abs(current_max - current_min) + 1e-6
                mid_shift = random.uniform(-range_width * 0.2, range_width * 0.2) # Wider shift
                scale_factor = random.uniform(0.7, 1.3) # Wider scale change
                new_mid = (current_min + current_max) / 2 + mid_shift
                new_width = range_width * scale_factor
                new_min = new_mid - new_width / 2
                new_max = new_mid + new_width / 2
                # Add specific bounds checks if needed
                current_params[param_key] = (new_min, new_max)

        # --- Frame Stutter/Hold --- (Keep logic, maybe increase hold chance/duration slightly)
        if frame_hold_counter > 0:
            # ... (same as before) ...
            continue
        if random.random() < 0.03 and frame_hold_counter == 0: # Slightly higher chance
            frame_hold_counter = random.randint(1, int(FPS * 0.7)) # Potentially longer hold
            held_frame = None

        # --- Base Frame ---
        # More variation: occasional gradients or noise base
        base_roll = random.random()
        if base_roll < 0.3: # Solid Color
             bg_color = random.choice(random.choice(theme_data["colors"]))
             frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
        elif base_roll < 0.5: # Gradient
             color1 = random.choice(random.choice(theme_data["colors"]))
             color2 = random.choice(random.choice(theme_data["colors"]))
             frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
             for k in range(HEIGHT):
                  ratio = k / HEIGHT
                  color = tuple(int(c1 * (1 - ratio) + c2 * ratio) for c1, c2 in zip(color1, color2))
                  cv2.line(frame, (0, k), (WIDTH, k), color, 1)
             if random.random() < 0.5: frame = cv2.rotate(frame, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])) # Random orientation
        else: # Noise Base
             frame = np.random.randint(0, 50, (HEIGHT, WIDTH, 3), dtype=np.uint8) # Dark noise

        # --- Apply Visual Effects ---
        # Get available effects for the current theme
        available_effects = theme_data.get("visual_effects", list(current_params["effect_probabilities"].keys()))
        effect_candidates = [globals()[name] for name in available_effects if name in globals()] # Map names to functions

        # More effects, weighted by intensity and avoiding too much repetition
        num_effects_to_apply = min(len(effect_candidates), random.randint(2, 5) + int(current_intensity * 2.5))

        applied_effects_count = 0
        random.shuffle(effect_candidates)
        applied_this_frame = []

        for effect_func in effect_candidates:
            if applied_effects_count >= num_effects_to_apply: break
            effect_name = effect_func.__name__

            # Check history to avoid applying the same effect many times in a row
            recent_count = effect_history.count(effect_name)
            if recent_count > 2 and random.random() < 0.6: continue # Skip if used >2 times recently

            # Calculate probability
            base_probability = current_params["effect_probabilities"].get(effect_name, 0.6) # Default higher prob
            probability_modifier = 0.4 + current_intensity * 0.6 # Intensity has strong effect
            final_probability = np.clip(base_probability * probability_modifier, 0.05, 0.95) # Higher min/max prob

            if random.random() < final_probability:
                try:
                    # --- Apply effect, scaling parameters by intensity ---
                    # (Similar scaling logic as before, but potentially more aggressive ranges)
                    # Example: Contrast scaling
                    if effect_func == apply_extreme_contrast:
                        a_min, a_max = current_params["contrast_alpha"]
                        b_min, b_max = current_params["contrast_beta"]
                        alpha_range = (a_min * (0.5 + current_intensity), a_max * (1.0 + current_intensity))
                        beta_range = (int(b_min * (1.0 + current_intensity*0.5)), int(b_max * (1.0 + current_intensity*0.5)))
                        frame = apply_extreme_contrast(frame, alpha_range=np.clip(alpha_range, 0.1, 15.0), # Wider alpha clamp
                                                       beta_range=np.clip(beta_range, -200, 200)) # Wider beta clamp
                    # Example: Feedback scaling
                    elif effect_func == apply_feedback:
                         f_min, f_max = current_params["feedback_alpha"]
                         alpha_range = (f_min * current_intensity, f_max * (1.0 + current_intensity))
                         frame = apply_feedback(frame, prev_frame, alpha_range=np.clip(alpha_range, 0.01, 0.9)) # Higher max feedback
                    # Example: Datamosh needs prev_frame
                    elif effect_func == apply_datamosh_sim:
                         frame = apply_datamosh_sim(frame, prev_frame) # Use default probs for now
                    # Add scaling for other effects based on current_params...
                    # Fallback for unscaled effects
                    else:
                         # Check if effect takes ranges and pass from current_params if so
                         # This requires more detailed handling per effect
                         # Simple fallback: call without args if possible
                         sig = inspect.signature(effect_func)
                         if len(sig.parameters) == 1: # Only takes frame
                             frame = effect_func(frame)
                         elif 'prev_frame' in sig.parameters: # Needs prev_frame
                              frame = effect_func(frame, prev_frame)
                         else: # Assume it takes ranges, try to pass defaults (less ideal)
                              # TODO: Improve this fallback by checking param names
                              frame = effect_func(frame)


                    applied_effects_count += 1
                    applied_this_frame.append(effect_name)
                except Exception as e:
                    print(f"Error applying {effect_name} at frame {i}: {e}")

        # Update effect history (keep last N effects)
        effect_history.extend(applied_this_frame)
        effect_history = effect_history[-20:] # Keep history of last 20 applied effects


        # --- Subliminal Flash (More Abrasive) ---
        flash_probability = 0.15 * current_intensity # Higher base chance
        if random.random() < flash_probability:
             # ... (Flash logic - maybe add new flash types like pure noise or ASCII flash) ...
             flash_type = random.choice(["invert", "text", "symbol", "color", "noise", "contrast", "ascii_flash"])
             overlay = frame.copy()
             try:
                 # ... (handle existing types) ...
                 if flash_type == "ascii_flash":
                      overlay = apply_ascii_sim(frame, block_size_range=(6,12), invert=random.choice([True,False]))

                 # Blend flash more strongly
                 frame = cv2.addWeighted(frame, random.uniform(0.0, 0.3), overlay, random.uniform(0.7, 1.0), 0)
             except Exception as e:
                  print(f"Error during subliminal flash ({flash_type}): {e}")


        # --- Text Rendering (More Chaotic) ---
        current_persistent_texts = []
        frame_overlay = frame.copy() # Draw on overlay
        # Update and draw persistent texts
        for text_info in persistent_texts:
            # ... (Update position, bounce logic - maybe add rotation update) ...
            text_info["pos"] = (np.clip(text_info["pos"][0], -WIDTH//2, WIDTH + WIDTH//2), # Allow more off-screen
                                np.clip(text_info["pos"][1], -HEIGHT//2, HEIGHT + HEIGHT//2))
            text_info["frame_count"] += 1

            if text_info["frame_count"] < text_info["lifetime"]:
                try:
                    display_font_size = text_info["font_size"] * (0.7 + current_intensity * 0.3)
                    display_font_size = max(0.1, display_font_size) # Ensure minimum size
                    thickness = max(1, int(display_font_size / 2))
                    # Use theme color directly
                    color = text_info["color"]
                    # Add slight jitter to position
                    jitter_x = random.randint(-2, 2)
                    jitter_y = random.randint(-2, 2)
                    pos = (text_info["pos"][0] + jitter_x, text_info["pos"][1] + jitter_y)

                    # Simple rotation not feasible with default putText, skip for now
                    cv2.putText(frame_overlay, text_info["text"], pos, FONT_FACE_TO_USE,
                                display_font_size, color, thickness, cv2.LINE_AA)
                    current_persistent_texts.append(text_info)
                except Exception as e:
                    current_persistent_texts.append(text_info) # Keep trying
            else: # Respawn logic (maybe change color/speed on respawn)
                if random.random() < 0.6: # Higher respawn chance
                    # ... (Respawn logic - maybe change text content more drastically) ...
                    current_persistent_texts.append(text_info)
        persistent_texts = current_persistent_texts

        # Fleeting text (more frequent)
        if random.random() < 0.6 + current_intensity * 0.2: # Much more likely
             num_fleet = random.randint(1, int(5 + current_intensity * 5)) # Many fleeting texts
             for _ in range(num_fleet):
                  # ... (Fleeting text logic - use Zalgo more often?) ...
                  fleet_text = random.choice(theme_data["words"]) + random.choice(theme_data["symbols"])
                  if random.random() < theme_data.get("zalgo_words_chance", 0.3) + 0.2:
                      fleet_text = zalgo_text(fleet_text)
                  # ... render fleeting text ...


        # Blend text overlay (maybe vary alpha?)
        text_alpha = np.clip(0.6 + current_intensity * 0.2, 0.3, 1.0)
        frame = cv2.addWeighted(frame, 1.0 - text_alpha * 0.5, frame_overlay, text_alpha, 0) # Blend differently


        # --- Store frame & Write ---
        prev_frame = frame.copy()
        if frame_hold_counter > 0 and held_frame is None: held_frame = frame.copy()
        video.write(frame)

        # --- Progress ---
        if (i + 1) % (FPS * 2) == 0 and frame_hold_counter == 0:
            print(f"Video Frame {i+1}/{FRAME_COUNT} (Intensity: {current_intensity:.2f})...")

    video.release()
    print("Video generation complete.")
    return DURATION, FPS, video_temp_path


# --- Audio Generation (Updated for melody, abrasiveness) ---
def generate_audio_enhanced(video_duration, video_fps, theme_data, intensity_func, global_params):
    """Generates audio with buried melody, more abrasive, linked to intensity."""
    DURATION = video_duration
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    NUM_CHANNELS = 2
    samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32)

    # Use global audio params
    current_audio_params = global_params["audio"]
    audio_instability_chance = global_params["instability_chance"] * 1.2 # Slightly different instability

    print(f"Generating {DURATION:.1f}s audio, Theme: {global_params['theme_name']}...")

    # Audio Elements
    min_freq, max_freq = theme_data["audio_freq_range"]
    noise_types = theme_data["audio_noise_types"]
    features = theme_data["audio_features"]

    # --- Generate Audio Events ---
    last_event_end_sample = 0
    melody_track = np.zeros(NUM_SAMPLES, dtype=np.float32) # Separate track for melody

    while last_event_end_sample < NUM_SAMPLES:
        current_sample_time = last_event_end_sample / SAMPLE_RATE
        current_intensity = intensity_func(current_sample_time)
        current_intensity = np.clip(current_intensity, 0.05, 3.0) # Wider clamp

        # --- Optional: Audio Parameter Instability ---
        if random.random() < audio_instability_chance:
             # ... (Logic to modify current_audio_params) ...
             pass

        # --- Determine Event ---
        # Make events more overlapping and denser with intensity
        event_max_duration = max(0.03, 2.0 / (current_intensity + 0.8)) # Shorter max duration
        event_min_duration = 0.005 # Allow very short events
        event_duration_sec = random.uniform(event_min_duration, event_max_duration)
        event_duration_samples = int(event_duration_sec * SAMPLE_RATE)
        event_duration_samples = min(event_duration_samples, NUM_SAMPLES - last_event_end_sample)
        if event_duration_samples <= 1: break

        event_start_sample = last_event_end_sample
        event_end_sample = event_start_sample + event_duration_samples

        # Volume scaling - more aggressive
        base_vol = random.uniform(0.03, 0.4) # Higher max base
        event_vol = np.clip(base_vol * (0.3 + current_intensity * 1.2), 0.005, 1.0) # Stronger scaling, allow full volume

        # Choose event type: Noise, Tone, Melody, Silence
        type_roll = random.random()
        silence_thresh = max(0.01, 0.15 * (1.8 - current_intensity)) # Less silence overall
        melody_thresh = silence_thresh + 0.15 # Fixed chance for melody? Or intensity based? Let's try fixed for now.
        noise_thresh = melody_thresh + (0.5 + 0.3 * current_intensity) # More noise when intense

        segment = np.zeros((event_duration_samples, NUM_CHANNELS), dtype=np.float32)
        segment_mono = np.zeros(event_duration_samples, dtype=np.float32)
        is_melody = False

        if type_roll < silence_thresh:
            event_type = "silence"
        elif type_roll < melody_thresh:
            event_type = "melody"
            is_melody = True
            melody_vol = event_vol * 0.4 # Keep melody quieter initially
            segment_mono = generate_melody_fragment(event_duration_samples, melody_vol, theme_data)
        elif type_roll < noise_thresh:
            event_type = "noise"
            noise_choice = random.choice(noise_types)
            # Bias towards harsher noises
            if current_intensity > 1.5 and random.random() < 0.6:
                 harsh_noises = [n for n in noise_types if "screech" in n or "clip" in n or "glitch" in n or "artifact" in n]
                 if harsh_noises: noise_choice = random.choice(harsh_noises)
            segment_mono = generate_noise(noise_choice, event_duration_samples, event_vol, features)
        else:
            event_type = "tone"
            current_max_freq = max_freq * (0.7 + current_intensity * 0.6)
            freq = random.uniform(min_freq, current_max_freq)
            freq = np.clip(freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)
            # More FM/Harmonics for tones when intense
            fm_chance = 0.4 + current_intensity * 0.3
            harmonic_chance = 0.3 + current_intensity * 0.2
            segment_mono = generate_tone(freq, event_duration_samples, event_vol, fm_chance, harmonic_chance)

        if event_type != "silence":
            # --- Apply Audio Effects (More Abrasive) ---
            effect_chance_mod = 0.5 + current_intensity # Higher base chance for effects

            # Distortion more likely and intense
            if "distortion" in features and random.random() < 0.5 * effect_chance_mod:
                d_min, d_max = current_audio_params["distortion_intensity"]
                dist_range = (d_min * (0.8 + current_intensity), d_max * (1.0 + current_intensity))
                segment_mono = apply_distortion(segment_mono, intensity_range=np.clip(dist_range, 1.0, 25.0)) # Higher max distortion

            # Bitcrush more likely and intense
            if "bitcrush" in features and random.random() < 0.4 * effect_chance_mod:
                 b_min, b_max = current_audio_params["bitcrush_depth"]
                 target_bits = int(max(2, b_min + (b_max - b_min) * (1.8 - current_intensity))) # Lower bits when intense
                 segment_mono = apply_bitcrush(segment_mono, bit_depth_range=(2, max(3, target_bits))) # Allow 2-bit

            # Spectral Glitch
            if "spectral_glitch" in features and random.random() < 0.3 * effect_chance_mod:
                 segment_mono = apply_spectral_glitch(segment_mono, intensity=current_intensity)

            # Convolution Reverb (Use sparingly, can be slow)
            if "convolution_reverb_damp" in features and random.random() < 0.1 * effect_chance_mod: # Low chance
                 impulse_name = random.choice(list(IMPULSE_RESPONSES.keys()))
                 segment_mono = apply_convolution_reverb(segment_mono, IMPULSE_RESPONSES[impulse_name])


            # --- Panning ---
            pan_extremity = np.clip(0.3 + current_intensity * 0.7, 0.1, 1.0) # Wider panning with intensity
            pan = random.uniform(-pan_extremity, pan_extremity)
            gain_l = np.sqrt(0.5 * (1 - pan))
            gain_r = np.sqrt(0.5 * (1 + pan))

            # Add to appropriate track
            end_idx = min(event_end_sample, NUM_SAMPLES)
            length = end_idx - event_start_sample
            if is_melody:
                 # Add melody direct to mono track for now, will mix later
                 melody_track[event_start_sample:end_idx] += segment_mono[:length]
            else:
                 # Add noise/tone to stereo samples
                 samples[event_start_sample:end_idx, 0] += segment_mono[:length] * gain_l
                 samples[event_start_sample:end_idx, 1] += segment_mono[:length] * gain_r


        # Move time forward allowing more overlap
        overlap_factor = np.clip(0.2 + current_intensity * 0.4, 0.05, 0.8) # More overlap
        advance_samples = int(event_duration_samples * (1.0 - overlap_factor))
        last_event_end_sample += max(1, advance_samples)

    # --- Mix Melody Track ---
    # Normalize melody track separately first? Or just mix low?
    max_melody_abs = np.max(np.abs(melody_track))
    if max_melody_abs > 1e-6:
         melody_track /= max_melody_abs # Normalize melody
    melody_mix_level = 0.3 # How loud the melody is in the final mix
    samples[:, 0] += melody_track * melody_mix_level
    samples[:, 1] += melody_track * melody_mix_level # Add mono melody to both channels


    # --- Final Normalization ---
    max_abs_amplitude = np.max(np.abs(samples))
    if max_abs_amplitude > 1e-6:
        print(f"Normalizing audio (Max amplitude before: {max_abs_amplitude:.3f})")
        samples /= max_abs_amplitude # Normalize the final mix
    else:
        print("Warning: Final audio mix appears silent.")

    samples_int16 = (samples * 32767).astype(np.int16)

    # --- Write WAV ---
    audio_temp_path = f"audio_temp_{int(time.time())}.wav" # Unique temp name
    try:
        with wave.open(audio_temp_path, 'wb') as wf:
            wf.setnchannels(NUM_CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples_int16.tobytes())
        print(f"Temporary audio saved to {audio_temp_path}")
        return audio_temp_path
    except Exception as e:
        print(f"Error writing WAV file: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # 1. Select Initial Theme
    # << SURPRISE: Theme can change mid-generation >>
    initial_theme_name = random.choice(list(THEMES.keys()))
    current_theme_data = copy.deepcopy(THEMES[initial_theme_name]) # Use deep copy

    print(f"--- Starting Avant-Garde Generation ---")
    print(f"Initial Theme: {initial_theme_name}")

    # 2. Setup Global Parameters (for instability and potential theme switching)
    global_params = {
        "theme_name": initial_theme_name,
        "visual": { # Visual parameters subject to instability
             "perlin_alpha": (0.1, 0.9), "perlin_scale": (2.0, 80.0),
             "block_shift_max": (10, WIDTH // 3), "block_size": (5, HEIGHT // 2),
             "color_shift_max": (5, 60),
             "warp_amp": (5, 80), "warp_freq": (0.002, 0.1),
             "pixel_block": (4, 96),
             "contrast_alpha": (1.0, 6.0), "contrast_beta": (-100, 100),
             "feedback_alpha": (0.02, 0.6),
             "solarize_thresh": (60, 200),
             "scanline_intensity": (0.4, 1.0),
             "effect_probabilities": { # Base probabilities (can change)
                  'apply_perlin_noise': 0.7, 'apply_block_shift': 0.6, 'apply_color_channel_shift': 0.6,
                  'apply_warp': 0.5, 'apply_pixelation': 0.4, 'apply_scanlines': 0.3,
                  'apply_solarize': 0.2, 'apply_extreme_contrast': 0.4, 'apply_feedback': 0.5,
                  'apply_datamosh_sim': 0.2, 'apply_ascii_sim': 0.15, # Add new effect probs
                  # Add probs for other new effects here
             }
        },
        "audio": { # Audio parameters subject to instability
            "distortion_intensity": (1.0, 15.0), # Wider range
            "bitcrush_depth": (2, 10), # Lower max bits possible
            # Add other audio params like reverb mix, delay time etc. if implemented
        },
        "instability_chance": 0.008, # Slightly higher base instability
        "style_break_chance": 0.001 # Small chance per frame/event to switch theme temporarily
    }

    # 3. Generate Intensity Profile
    # Use a longer estimated duration to ensure profile covers potential video length
    estimated_max_duration_sec = MAX_DURATION + 10
    estimated_max_samples = int(estimated_max_duration_sec * SAMPLE_RATE)
    intensity_function = generate_intensity_profile(estimated_max_samples, SAMPLE_RATE)

    # << SURPRISE: Implement Style Break Logic within generation functions >>
    # The generation functions will now need access to global_params and THEMES
    # to potentially switch theme data mid-run. This adds significant complexity
    # and is omitted here for clarity, but the framework is set up.
    # A simple version could be added in the main loop of generate_frames/audio:
    # if random.random() < global_params["style_break_chance"]:
    #     new_theme_name = random.choice(list(THEMES.keys()))
    #     print(f"STYLE BREAK! Switching to {new_theme_name} temporarily...")
    #     theme_data = THEMES[new_theme_name] # Switch theme_data used for this frame/event
    #     # Need logic to switch back after a short duration

    # 4. Generate Video
    actual_duration, actual_fps, video_temp_file = generate_frames_enhanced(current_theme_data, intensity_function, global_params)
    print(f"Video generation yielded: Duration={actual_duration:.2f}s, FPS={actual_fps}")

    # 5. Generate Audio
    audio_temp_file = generate_audio_enhanced(actual_duration, actual_fps, current_theme_data, intensity_function, global_params)

    # 6. Combine Video and Audio
    if video_temp_file and audio_temp_file:
        print("Combining video and audio using ffmpeg...")
        if os.path.exists(OUTPUT_FILE):
            try: os.remove(OUTPUT_FILE)
            except OSError as e: print(f"Error removing existing output file: {e}")

        ffmpeg_command = [
            'ffmpeg', '-y',
            '-i', video_temp_file,
            '-i', audio_temp_file,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', # Re-encode video for compatibility, fast preset
            '-c:a', 'aac', '-b:a', '256k', # Higher audio bitrate
            '-shortest',
            OUTPUT_FILE
        ]
        print(f"Running ffmpeg: {' '.join(ffmpeg_command)}")
        try:
            result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            # print("ffmpeg Output:", result.stdout) # Can be very verbose
            # print("ffmpeg Error (if any):", result.stderr)
            print(f"Successfully combined streams into '{OUTPUT_FILE}'")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed with error code {e.returncode}")
            # print("ffmpeg Output:", e.stdout)
            print("ffmpeg Error:", e.stderr)
            print("Check if ffmpeg is installed and in your system's PATH.")
        except FileNotFoundError:
             print("Error: ffmpeg command not found. Please install ffmpeg and ensure it's in your PATH.")
        except Exception as e:
            print(f"An unexpected error occurred during ffmpeg execution: {e}")
    else:
        print("Skipping ffmpeg combination due to errors in video or audio generation.")

    # 7. Clean up temporary files
    print("Cleaning up temporary files...")
    for temp_file in [video_temp_file, audio_temp_file]:
        if temp_file and os.path.exists(temp_file):
            try: os.remove(temp_file)
            except OSError as e: print(f"Error removing temp file {temp_file}: {e}")
    print("Temporary files removed.")

    end_time = time.time()
    print(f"--- Generation Process Finished in {end_time - start_time:.2f} seconds ---")


