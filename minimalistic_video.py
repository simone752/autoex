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
OUTPUT_FILE = "avant_garde_output.mp4"
HIGH_FREQ_CUTOFF = 22000 # Allow slightly higher frequencies
LOW_FREQ_CUTOFF = 18      # Allow slightly lower frequencies
SAMPLE_RATE = 44100
time_step = 1.0 / SAMPLE_RATE

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
        "visual_effects": ["apply_perlin_noise", "apply_block_shift", "apply_warp", "apply_feedback", "apply_crt_ghost_sim", "apply_extreme_contrast", "apply_solarize", "apply_slit_scan_sim"], # Different effect focus (ensure functions exist)
        "audio_freq_range": (18, 2000), # Even lower floor
        "audio_noise_types": ["brown", "squelch", "heartbeat", "breathing", "wet_clicks", "sub_bass", "granular_flesh", "choking_sim"], # Added granular/choking (ensure functions exist)
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
        "audio_noise_types": ["pink", "static", "digital_artifact", "modem_sim", "filtered_noise", "granular_synth"], # Ensure functions exist
        "audio_features": ["stutter", "bitcrush", "delay", "reverb", "extreme_panning", "spectral_glitch", "tts_fragments"], # Added TTS (TTS needs external lib)
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
# For simplicity in this example, we'll mostly stick to DEFAULT_FONT.
print(f"Font selected for rendering (prefer TTF if available/compatible): {FONT_TO_USE}")
FONT_FACE_TO_USE = DEFAULT_FONT


# Initialize pygame mixer
try:
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=2048) # Larger buffer maybe?
except pygame.error as e:
    # This error is expected in environments without a proper audio device (like GitHub Actions)
    # It's non-critical as we are only writing to a file.
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

def apply_perlin_noise(frame, alpha_range=(0.1, 0.9), scale_range=(2.0, 80.0), oct_range=(3, 9)): # Wider ranges
    """Applies a Perlin noise overlay with randomized parameters."""
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
    gray_noise = np.zeros((height, width), dtype=np.float32) # Use float for noise calculation
    for i in range(height):
        for j in range(width):
            gray_noise[i, j] = noise.pnoise2(y_coords[i], x_coords[j], octaves=octaves,
                                             persistence=persistence, lacunarity=lacunarity, base=seed)
    # Normalize noise, convert to BGR
    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX)
    colored_noise = cv2.cvtColor(colored_noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if random.random() < 0.4: colored_noise = 255 - colored_noise # Higher invert chance
    # Blend with the original frame
    return cv2.addWeighted(frame, 1 - alpha, colored_noise, alpha, 0)

def apply_block_shift(frame, max_shift_range=(10, WIDTH // 3), block_size_range=(5, HEIGHT // 2), num_blocks_range=(20, 100)): # More blocks, wider ranges
    """Shifts random rectangular blocks within the frame."""
    max_shift = random.randint(*max_shift_range)
    # Ensure block size range is valid
    bs_min = max(1, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size_max = random.randint(bs_min, bs_max)
    num_blocks = random.randint(*num_blocks_range)
    height, width = frame.shape[:2]
    temp_frame = frame.copy()
    for _ in range(num_blocks):
        bh = random.randint(bs_min, block_size_max) # Smaller min block size possible
        bw = random.randint(bs_min, block_size_max)
        # Ensure block dimensions are valid and within frame bounds for selection
        if height - bh <= 0 or width - bw <= 0: continue
        y = random.randint(0, height - bh - 1)
        x = random.randint(0, width - bw - 1)

        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)

        # Calculate target coordinates, ensuring they stay within bounds
        target_y_start = np.clip(y + shift_y, 0, height - bh)
        target_x_start = np.clip(x + shift_x, 0, width - bw)

        try: # Add try-except for potential slicing issues with extreme values
            # Extract the block using calculated coordinates
            block = frame[y:y+bh, x:x+bw]
            # Place the block at the target location
            temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block
        except ValueError as e:
            # This might happen if calculated dimensions mismatch somehow
            # print(f"Block shift slice error: {e}, skipping block.") # Optional debug
            pass
    return temp_frame

def apply_color_channel_shift(frame, max_shift_range=(5, 60)): # Wider shift range
    """Shifts B, G, R color channels independently using np.roll."""
    max_shift = random.randint(*max_shift_range)
    temp_frame = frame.copy()
    height, width = frame.shape[:2]
    for i in range(3): # Iterate through B, G, R
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        channel = frame[:,:,i]
        # Apply circular shift using np.roll
        shifted_channel = np.roll(np.roll(channel, shift_y, axis=0), shift_x, axis=1)
        temp_frame[:,:,i] = shifted_channel
    return temp_frame

def apply_warp(frame, amplitude_range=(5, 80), freq_range=(0.002, 0.1)): # Wider ranges
    """Applies wave-like distortion using cv2.remap for efficiency."""
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
    # Ensure calculation uses float32 for remap compatibility
    offset_x = (amplitude_x * np.sin(2 * math.pi * y_mesh * frequency_x + phase_x)).astype(np.float32)
    offset_y = (amplitude_y * np.cos(2 * math.pi * x_mesh * frequency_y + phase_y)).astype(np.float32)
    # Calculate source map, clipping included
    map_x = np.clip(x_mesh.astype(np.float32) + offset_x, 0, cols - 1)
    map_y = np.clip(y_mesh.astype(np.float32) + offset_y, 0, rows - 1)
    # Apply remap using linear interpolation and replicating border pixels
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def apply_feedback(frame, prev_frame, alpha_range=(0.02, 0.6)): # Higher max alpha
    """Blends the current frame with a transformed version of the previous frame."""
    if prev_frame is None: return frame # No feedback on first frame
    alpha = random.uniform(*alpha_range)
    modified_prev = prev_frame.copy() # Work on a copy

    # Add more aggressive random transformations to the previous frame
    if random.random() < 0.2: # Increased chance
        rows, cols = modified_prev.shape[:2]
        angle = random.uniform(-5, 5) # Wider angle
        scale = random.uniform(0.95, 1.05) # Wider scale
        tx = random.uniform(-10, 10) # Translation X
        ty = random.uniform(-10, 10) # Translation Y
        center = (cols/2, rows/2)
        # Get rotation matrix and add translation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        # Apply affine transformation
        modified_prev = cv2.warpAffine(modified_prev, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    # Add a chance for slight color shift in feedback
    if random.random() < 0.1:
         shift_b = random.randint(-10, 10)
         shift_g = random.randint(-10, 10)
         shift_r = random.randint(-10, 10)
         # Add shift using numpy, clipping automatically handled by uint8 overflow/underflow
         modified_prev = modified_prev + np.array([shift_b, shift_g, shift_r], dtype=np.int16).astype(np.uint8)


    # Blend the current frame with the modified previous frame
    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)

def apply_pixelation(frame, block_size_range=(4, 96)): # Wider range
    """Applies pixelation effect by resizing down and up."""
    height, width = frame.shape[:2]
    # Ensure block size range is valid
    bs_min = max(2, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size = random.randint(bs_min, bs_max)

    # Calculate target pixelated dimensions, ensuring they are at least 1x1
    pixel_w, pixel_h = max(1, width // block_size), max(1, height // block_size)
    # Resize down using nearest neighbor interpolation (fastest for pixelation)
    temp = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_NEAREST)
    # Resize back up to original size using nearest neighbor
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def apply_scanlines(frame, intensity_range=(0.4, 1.0), thickness_range=(1, 4), color_variation=40): # Higher max intensity, thickness
    """Overlays horizontal scanlines onto the frame."""
    intensity = random.uniform(*intensity_range)
    thickness = random.randint(*thickness_range)
    # Create a separate layer for scanlines (avoids modifying original directly in loop)
    scanline_layer = np.zeros_like(frame)
    height, width = frame.shape[:2]
    base_color_val = random.randint(0, 70) # Base color for dark lines

    # Draw lines on the separate layer
    for y in range(0, height, thickness * 2): # Draw lines every other step based on thickness
        # Add random variation to line color
        line_color_val = base_color_val + random.randint(-color_variation, color_variation)
        # Ensure color value is within 0-255 and create BGR tuple
        line_color = tuple(np.clip([line_color_val]*3, 0, 255))
        cv2.line(scanline_layer, (0, y), (width, y), line_color, thickness)

    # Blend the scanline layer over the original frame using addWeighted
    # Adjust gamma (last parameter) for blend darkness
    return cv2.addWeighted(frame, 1.0, scanline_layer, intensity * 1.1, -int(255 * intensity * 0.3)) # More aggressive blend

def apply_solarize(frame, threshold_range=(60, 200)): # Wider range
    """Inverts pixel values above a certain threshold."""
    threshold = random.randint(*threshold_range)
    # Create a mask where pixel intensity is above the threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    solarized_frame = frame.copy()
    # Apply inversion only where the mask is white (255)
    solarized_frame[mask == 255] = 255 - solarized_frame[mask == 255]
    return solarized_frame

def apply_extreme_contrast(frame, alpha_range=(1.0, 6.0), beta_range=(-100, 100)): # Much wider ranges
    """Applies strong contrast and brightness adjustments."""
    alpha = random.uniform(*alpha_range) # Contrast factor
    beta = random.randint(*beta_range)   # Brightness shift
    # Use convertScaleAbs for efficiency and automatic clipping to 0-255 range
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# << NEW VISUAL EFFECTS >>
def apply_datamosh_sim(frame, prev_frame, hold_prob=0.3, smear_prob=0.5, block_size_range=(32, 128)):
    """Simulates datamoshing effects like I-frame holds and P-frame smearing."""
    if prev_frame is None: return frame
    height, width = frame.shape[:2]
    output = frame.copy()
    # Ensure block size range is valid
    bs_min = max(16, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size = random.randint(bs_min, bs_max)

    # I-Frame Hold Simulation: Replace random blocks with blocks from previous frame
    if random.random() < hold_prob:
        num_hold_blocks = random.randint(1, max(1, (width // block_size) * (height // block_size) // 3))
        for _ in range(num_hold_blocks):
            # Calculate block coordinates, ensuring they align with grid
            bx_idx = random.randint(0, max(0, width // block_size - 1))
            by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            # Ensure block dimensions don't exceed frame boundaries
            bh = min(block_size, height - by)
            bw = min(block_size, width - bx)
            if bh > 0 and bw > 0: # Check if block has valid dimensions
                output[by:by+bh, bx:bx+bw] = prev_frame[by:by+bh, bx:bx+bw]

    # P-Frame Smear Simulation: Copy blocks from prev_frame with a simulated motion vector
    if random.random() < smear_prob:
        num_smear_blocks = random.randint(1, max(1, (width // block_size) * (height // block_size) // 4))
        for _ in range(num_smear_blocks):
            # Choose source block coordinates
            bx_idx = random.randint(0, max(0, width // block_size - 1))
            by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            # Ensure block dimensions are valid
            bh = min(block_size, height - by)
            bw = min(block_size, width - bx)
            if bh <= 0 or bw <= 0: continue # Skip if block is invalid

            # Simulate motion vector
            vx = random.randint(-block_size // 2, block_size // 2)
            vy = random.randint(-block_size // 2, block_size // 2)

            # Get the block from the previous frame
            smear_block = prev_frame[by:by+bh, bx:bx+bw]

            # Calculate target coordinates, clipping to frame bounds
            target_y = np.clip(by + vy, 0, height - bh)
            target_x = np.clip(bx + vx, 0, width - bw)

            # Place the smeared block at the target location
            output[target_y:target_y+bh, target_x:target_x+bw] = smear_block

    return output

def apply_ascii_sim(frame, block_size_range=(8, 16), char_set=" .:-=+*#%@", invert=False):
    """Overlays a crude ASCII representation based on block brightness."""
    height, width = frame.shape[:2]
    # Ensure block size range is valid
    bs_min = max(4, block_size_range[0])
    bs_max = max(bs_min + 1, block_size_range[1])
    block_size = random.randint(bs_min, bs_max)

    output = frame.copy()
    # Adjust font scale based on block size for visibility
    font_scale = max(0.1, block_size / 15.0)
    thickness = 1
    chars = list(char_set)
    if invert: chars = chars[::-1] # Reverse characters if inverted brightness mapping is desired
    num_chars = len(chars)

    # Convert frame to grayscale for brightness calculation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Iterate over blocks
    for y in range(0, height - block_size + 1, block_size):
        for x in range(0, width - block_size + 1, block_size):
            # Extract block and calculate average brightness
            block = gray[y:y+block_size, x:x+block_size]
            avg_brightness = np.mean(block)
            # Map brightness to character index
            char_index = int(np.clip((avg_brightness / 255.0) * num_chars, 0, num_chars - 1))
            char = chars[char_index]
            # Choose text color for contrast (black on light, white on dark)
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255)
            # Position text roughly in the center of the block
            # Adjust position slightly for better centering based on font size
            text_size, _ = cv2.getTextSize(char, FONT_FACE_TO_USE, font_scale, thickness)
            pos = (x + (block_size - text_size[0]) // 2, y + (block_size + text_size[1]) // 2)

            try:
                # Draw the character onto the output frame
                cv2.putText(output, char, pos, FONT_FACE_TO_USE, font_scale, text_color, thickness, cv2.LINE_AA)
            except Exception as e:
                 # print(f"ASCII sim text error: {e}") # Optional debug
                 pass # Ignore if font fails or character is problematic
    return output

# Placeholder for other new effects mentioned in themes
def apply_crt_ghost_sim(frame, prev_frame):
    """Placeholder: Simulates CRT monitor ghosting."""
    # Actual implementation would involve keeping a buffer of past frames
    # and blending them with decay. For now, just return frame.
    if prev_frame is not None:
        # Simple blend with previous frame as a basic ghost
        return cv2.addWeighted(frame, 0.85, prev_frame, 0.15, 0)
    return frame

def apply_slit_scan_sim(frame):
    """Placeholder: Simulates slit-scan effect."""
    # Actual implementation requires maintaining a buffer and stacking slices.
    # Simple simulation: horizontal smearing
    rows, cols, _ = frame.shape
    output = frame.copy()
    for r in range(rows):
         smear_amount = int(math.sin(r * 0.1) * 10) # Simple sine wave smear
         output[r, :] = np.roll(output[r, :], smear_amount, axis=0)
    return output

def apply_vector_field_sim(frame):
    """Placeholder: Simulates distortion from a vector field."""
    # Actual implementation needs vector field generation (e.g., Perlin noise)
    # and cv2.remap. Simple warp as placeholder:
    return apply_warp(frame, amplitude_range=(2,10), freq_range=(0.05, 0.2))


# --- Audio Generation & Effects (Optimized & New Additions) ---

def apply_distortion(data, intensity_range=(1.5, 5.0)):
    """Applies hard clipping distortion."""
    intensity = random.uniform(*intensity_range)
    # Clip audio samples, scaling by intensity first
    return np.clip(data * intensity, -0.98, 0.98) # Clip slightly below full scale

def apply_bitcrush(data, bit_depth_range=(4, 12)):
    """Simulates bitcrushing by quantizing audio samples."""
    # Ensure bit depth range is valid
    bd_min = max(2, bit_depth_range[0])
    bd_max = max(bd_min + 1, bit_depth_range[1])
    bits = random.randint(bd_min, bd_max)

    if bits >= 16: return data # No effect if bit depth is high
    # Calculate the number of quantization steps based on bit depth
    steps = 2**(bits - 1) # For signed values centered around zero
    # Quantize: scale to integer range, round, scale back to float range
    # Assuming input data is already in [-1, 1]
    quantized = np.round(data * steps) / steps
    return quantized

# << NEW AUDIO EFFECTS >>
def apply_spectral_glitch(data, intensity=0.5):
    """Applies glitches in the frequency domain using FFT."""
    n_samples = len(data)
    if n_samples < 1024: return data # Need enough data for meaningful FFT

    # Ensure data is float for FFT
    data_float = data.astype(np.float32)
    spectrum = fft(data_float)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    n_freqs = len(spectrum)

    # Glitch: Randomly zero out frequency bins or swap phases
    # Number of glitches scales with intensity and data length
    num_glitches = int(n_freqs * 0.05 * intensity * random.random()) # Adjust glitch density factor
    indices_to_glitch = random.sample(range(1, n_freqs // 2), min(num_glitches, n_freqs // 2 - 1)) # Avoid DC and Nyquist

    for idx in indices_to_glitch:
        if random.random() < 0.6: # Zero out bin (magnitude glitch)
             magnitude[idx] = 0
             if idx < n_freqs: magnitude[n_freqs - idx] = 0 # Mirror for real signal symmetry
        else: # Swap phases of adjacent bins (phase glitch)
             if idx + 1 < n_freqs // 2:
                  # Swap phase values
                  phase[idx], phase[idx+1] = phase[idx+1], phase[idx]
                  # Mirror phase swaps (negative conjugate for real signal symmetry)
                  if n_freqs - idx < n_freqs: phase[n_freqs - idx] = -phase[idx]
                  if n_freqs - (idx+1) < n_freqs : phase[n_freqs - (idx+1)] = -phase[idx+1]

    # Reconstruct signal from modified magnitude and phase
    new_spectrum = magnitude * np.exp(1j * phase)
    glitched_data = np.real(ifft(new_spectrum))

    # Normalize to prevent clipping after IFFT, trying to match original scale
    max_abs_orig = np.max(np.abs(data_float))
    max_abs_glitched = np.max(np.abs(glitched_data))
    if max_abs_glitched > 1e-6 and max_abs_orig > 1e-6:
        glitched_data *= (max_abs_orig / max_abs_glitched) # Restore original amplitude scale roughly
    elif max_abs_glitched > 1e-6: # If original was silent, normalize glitched
        glitched_data /= max_abs_glitched

    return glitched_data.astype(data.dtype) # Return in original dtype

def apply_convolution_reverb(data, impulse_response):
    """Applies reverb using convolution (can be slow for long impulses)."""
    n_data = len(data)
    n_ir = len(impulse_response)

    # Ensure impulse response is not excessively long compared to data
    if n_ir >= n_data:
        impulse_response = impulse_response[:n_data//2] # Truncate impulse if too long
        n_ir = len(impulse_response)
        if n_ir == 0: return data # Cannot convolve with empty impulse

    # Perform convolution
    reverbed_data = np.convolve(data, impulse_response, mode='same') # 'same' keeps original length

    # Normalize - convolution can change amplitude significantly
    max_abs_orig = np.max(np.abs(data))
    max_abs_rev = np.max(np.abs(reverbed_data))
    # Avoid division by zero and normalize only if necessary
    if max_abs_rev > 1e-6 and max_abs_orig > 1e-6:
        # Scale reverb to roughly match the peak level of the original dry signal
        reverbed_data *= (max_abs_orig / max_abs_rev)
    elif max_abs_rev > 1e-6: # If original was silent, just normalize reverb
         reverbed_data /= max_abs_rev


    return reverbed_data

# Simple impulse responses for reverb (pre-normalized)
IMPULSE_RESPONSES = {
    "damp": np.exp(-np.linspace(0, 10, int(SAMPLE_RATE * 0.2))) * np.random.randn(int(SAMPLE_RATE * 0.2)), # Short, noisy decay
    "metal_hit": np.sin(2*np.pi*1500*np.linspace(0,0.1,int(SAMPLE_RATE*0.1))) * np.exp(-np.linspace(0, 20, int(SAMPLE_RATE * 0.1))), # Metallic ring
    "noise_burst": np.random.randn(int(SAMPLE_RATE * 0.1)) * np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * 0.1)))
}
# Normalize impulses to have max amplitude of 1
for k in IMPULSE_RESPONSES:
    max_abs = np.max(np.abs(IMPULSE_RESPONSES[k]))
    if max_abs > 1e-6:
        IMPULSE_RESPONSES[k] /= max_abs


# --- Tone and Noise Generation ---

def generate_tone(freq, duration_samples, vol, fm_chance=0.5, harmonic_chance=0.4):
    """Generates a tone with potential FM or harmonics and sharp envelope."""
    if duration_samples <= 0: return np.array([], dtype=np.float32) # Handle zero duration
    t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
    wave = np.zeros(duration_samples, dtype=np.float32)

    # Add Frequency Modulation (FM)
    if random.random() < fm_chance:
        mod_freq = freq * random.uniform(0.1, 5.0) # Wider FM frequency ratio
        # Ensure mod_depth calculation is safe
        mod_depth = abs(vol) * random.uniform(1.0, 8.0) # Wider FM modulation depth (relative to vol)
        # Add safety check for extreme frequencies causing issues in np.sin
        if abs(freq * t[-1]) < 1e9 and abs(mod_freq * t[-1]) < 1e9:
             wave = np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))
        else: # Fallback to simple sine if frequencies are too high
             wave = np.sin(2 * np.pi * freq * t)

    # Add Harmonics
    elif random.random() < harmonic_chance:
        harmonic_count = random.randint(1, 4)
        wave = np.sin(2 * np.pi * freq * t) # Fundamental frequency
        for h in range(2, harmonic_count + 2): # Add 2nd, 3rd, etc. harmonics
            harmonic_vol = random.uniform(0.1, 0.5) / h # Decrease volume for higher harmonics
            phase_shift = random.uniform(0, np.pi) # Random phase shift
            wave += harmonic_vol * np.sin(2 * np.pi * freq * h * t + phase_shift)
        # Basic normalization attempt for harmonics
        max_abs = np.max(np.abs(wave))
        if max_abs > 1e-6: wave /= max_abs # Normalize before applying volume

    # Default: Simple Sine Wave
    else:
        wave = np.sin(2 * np.pi * freq * t)

    wave *= vol # Apply overall volume

    # Apply a sharp ADSR-like envelope
    attack_len = min(int(SAMPLE_RATE * 0.005), duration_samples // 3) # Very short attack
    decay_len = min(int(SAMPLE_RATE * random.uniform(0.05, 0.3)), max(0, duration_samples - attack_len)) # Short decay
    sustain_level = random.uniform(0.1, 0.7)
    release_len = min(int(SAMPLE_RATE * random.uniform(0.05, 0.4)), max(0, duration_samples - attack_len - decay_len)) # Variable release

    envelope = np.ones(duration_samples)
    if attack_len > 0:
        envelope[:attack_len] = np.linspace(0, 1, attack_len)

    sustain_start = attack_len + decay_len
    if decay_len > 0:
        envelope[attack_len:sustain_start] = np.linspace(1, sustain_level, decay_len)

    # Apply sustain level and exponential release tail
    release_start = sustain_start
    if release_len > 0 and release_start + release_len <= duration_samples:
         envelope[release_start:release_start+release_len] = sustain_level * np.exp(-np.linspace(0, 5, release_len))
         envelope[release_start+release_len:] = 0 # Ensure silence after release
    elif release_start < duration_samples: # If no distinct release phase fits, hold sustain or start exponential decay from sustain level
         envelope[release_start:] = sustain_level * np.exp(-np.linspace(0, 5, duration_samples - release_start))


    return wave * envelope

# << ENSURE generate_noise IS DEFINED >>
def generate_noise(noise_type, duration_samples, vol, features=[]):
    """Generates various types of noise based on the specified type."""
    if duration_samples <= 0: return np.array([], dtype=np.float32) # Handle zero duration

    if noise_type == "white":
        return vol * (2 * np.random.random(duration_samples) - 1).astype(np.float32)

    elif noise_type == "pink":
        # Simple approximation using filtered cumulative sum of white noise
        wn = (2 * np.random.random(duration_samples) - 1).astype(np.float32)
        pink = np.cumsum(wn)
        # Basic high-pass filter to remove DC offset and reduce low-end dominance
        alpha = 0.99 # Filter coefficient
        pink_filtered = np.zeros_like(pink)
        pink_filtered[0] = pink[0]
        for n in range(1, duration_samples):
            pink_filtered[n] = alpha * pink_filtered[n-1] + pink[n] - pink[n-1]
        # Normalize to prevent potential clipping after filtering
        max_abs = np.max(np.abs(pink_filtered))
        if max_abs > 1e-6: pink_filtered /= max_abs
        return vol * pink_filtered

    elif noise_type == "brown":
        wn = (2 * np.random.random(duration_samples) - 1).astype(np.float32)
        brown = np.cumsum(wn)
        # Normalize the Brownian noise
        max_abs = np.max(np.abs(brown))
        if max_abs > 1e-6: brown /= max_abs
        return vol * brown

    elif noise_type in ["glitch", "clicks", "wet_clicks"]:
        noise = np.zeros(duration_samples, dtype=np.float32)
        num_clicks = random.randint(10, 150) # More clicks for intensity
        click_type = random.choice(["sharp", "burst", "resonant"])
        if noise_type == "wet_clicks": click_type = "resonant" # Bias towards resonant for 'wet'

        for _ in range(num_clicks):
            max_pos = duration_samples - 50 # Ensure space for potential resonance tail
            if max_pos <= 0: continue # Skip if duration too short
            pos = random.randint(0, max_pos)
            amp = vol * random.uniform(0.5, 1.5) # Louder clicks possible

            if click_type == "sharp":
                noise[pos] = amp * random.choice([-1, 1])
                if pos + 1 < duration_samples:
                    noise[pos+1] = -noise[pos] * 0.5 # Simple sharp decay
            elif click_type == "burst":
                burst_len = random.randint(2, 10)
                end_pos = min(pos + burst_len, duration_samples)
                actual_len = end_pos - pos
                if actual_len > 0:
                     noise[pos:end_pos] = amp * (2 * np.random.random(actual_len) - 1)
            elif click_type == "resonant": # Short decaying sine wave
                click_freq = random.uniform(500, 8000)
                res_len = 40 # Fixed length for resonant click tail
                if pos + res_len < duration_samples: # Ensure space
                    t_click = np.arange(res_len) / SAMPLE_RATE
                    click_env = np.exp(-np.linspace(0, 15, res_len)) # Faster decay
                    click_wave = amp * np.sin(2 * np.pi * click_freq * t_click) * click_env
                    noise[pos:pos+res_len] += click_wave.astype(np.float32) # Additive

        return noise # No extra vol multiplication, amp handled per click

    elif noise_type == "static":
        # Mix white noise with some filtering and crackle modulation
        wn = (2 * np.random.random(duration_samples) - 1).astype(np.float32)
        # Simple low-pass filter for static character
        alpha = random.uniform(0.1, 0.9) # Filter coefficient
        filtered_static = np.zeros_like(wn)
        for n in range(1, duration_samples):
            filtered_static[n] = alpha * filtered_static[n-1] + (1 - alpha) * wn[n]
        # Add crackle effect by modulating amplitude randomly
        crackle = 1.0 + random.uniform(0.5, 2.5) * (2 * np.random.random(duration_samples) - 1)
        return vol * filtered_static * crackle

    elif noise_type == "digital_artifact":
        # Start with white noise and apply destructive digital-like effects
        noise = vol * (2 * np.random.random(duration_samples) - 1).astype(np.float32)
        # Randomly zero out sections
        if random.random() < 0.5:
            num_zeros = random.randint(5, 20)
            for _ in range(num_zeros):
                if duration_samples < 100: continue
                z_start = random.randint(0, duration_samples - 100)
                z_len = random.randint(10, 100)
                noise[z_start : z_start+z_len] = 0
        # Randomly repeat small chunks (stuttering)
        else:
            num_repeats = random.randint(3, 10)
            chunk_len = random.randint(5, 50)
            if duration_samples > chunk_len * num_repeats + 10:
                r_start = random.randint(0, duration_samples - (chunk_len * num_repeats) - 5)
                chunk = noise[r_start : r_start+chunk_len].copy()
                for r_idx in range(num_repeats):
                    start_idx = r_start + (r_idx * chunk_len)
                    end_idx = start_idx + chunk_len
                    if end_idx <= duration_samples:
                        noise[start_idx : end_idx] = chunk
        # Apply bitcrushing frequently for digital feel
        if random.random() < 0.7:
            noise = apply_bitcrush(noise, bit_depth_range=(3, 8))
        return noise

    elif noise_type == "screech":
        # High-frequency noise with heavy, fast modulation
        freq = random.uniform(4000, HIGH_FREQ_CUTOFF - 1000)
        mod_freq = random.uniform(50, 500) # Fast modulation
        mod_depth = abs(vol) * random.uniform(5, 20) # Deep modulation
        t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
        wave = np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))
        # Add a bit of white noise for harshness
        wave += 0.3 * (2 * np.random.random(duration_samples) - 1)
        # Normalize roughly to prevent excessive clipping
        max_abs = np.max(np.abs(wave))
        if max_abs > 1e-6: wave /= max_abs
        return (vol * wave).astype(np.float32)

    elif noise_type == "sub_bass":
        # Very low frequency tones, often square or sawtooth for rumble
        freq = random.uniform(LOW_FREQ_CUTOFF, 60)
        t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
        if random.random() < 0.5: # Square wave approximation
            wave = np.sign(np.sin(2 * np.pi * freq * t))
        else: # Sawtooth wave approximation
            wave = 2 * (t * freq - np.floor(0.5 + t * freq))
        # Optional: Add slow LFO for 'throb' effect if requested in theme features
        if "sub_bass_throb" in features:
            lfo_freq = random.uniform(0.1, 2) # Very slow LFO frequency
            wave *= (0.7 + 0.3 * np.sin(2 * np.pi * lfo_freq * t)) # Modulate amplitude
        return (vol * wave).astype(np.float32)

    # --- Theme Specific Noises (Examples) ---
    elif noise_type == "heartbeat":
         # Simulate heartbeat rhythm with low-frequency thuds
         bpm = random.uniform(40, 100) # Beats per minute
         bps = bpm / 60.0
         beat_interval_samples = int(SAMPLE_RATE / bps)
         thud_len = int(SAMPLE_RATE * 0.08) # Short thud sound
         thud_freq = random.uniform(40, 80)
         if thud_len <= 0: return np.zeros(duration_samples, dtype=np.float32) # Safety check
         t_thud = np.linspace(0, thud_len * time_step, thud_len, endpoint=False)
         thud_env = np.exp(-np.linspace(0, 8, thud_len)) # Exponential decay
         thud_sound = np.sin(2*np.pi*thud_freq*t_thud) * thud_env

         noise = np.zeros(duration_samples, dtype=np.float32)
         current_sample = 0
         while current_sample + beat_interval_samples < duration_samples:
             # Ensure thud doesn't write past end of array
             write_len = min(thud_len, duration_samples - current_sample)
             if write_len <= 0: break
             noise[current_sample : current_sample + write_len] += thud_sound[:write_len] * vol
             # Add a smaller second beat?
             if random.random() < 0.7:
                 second_beat_offset = int(beat_interval_samples * random.uniform(0.3, 0.5))
                 second_beat_start = current_sample + second_beat_offset
                 write_len_2 = min(thud_len, duration_samples - second_beat_start)
                 if write_len_2 > 0:
                     noise[second_beat_start : second_beat_start + write_len_2] += thud_sound[:write_len_2] * vol * 0.6
             current_sample += beat_interval_samples
         return noise

    elif noise_type == "breathing":
        # Filtered noise modulated slowly
        base_noise = generate_noise("pink", duration_samples, 1.0) # Start with pink noise
        breath_rate = random.uniform(0.1, 0.4) # Breaths per second (slow)
        t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
        amp_mod = 0.5 + 0.5 * np.sin(2 * np.pi * breath_rate * t) # Amplitude modulation
        # Add filter modulation (simple low-pass example)
        filter_mod = 0.5 + 0.5 * np.sin(2 * np.pi * breath_rate * t + np.pi / 2) # Offset phase

        filtered_breath = np.zeros_like(base_noise)
        # Apply simple time-varying low-pass filter
        # This loop can be slow in Python, consider libraries like scipy.signal for faster filtering if needed
        current_alpha = 0.5 # Initial alpha
        for n in range(1, duration_samples):
             target_alpha = np.clip(0.1 + filter_mod[n] * 0.8, 0.01, 0.99) # Target alpha based on modulation
             # Smooth alpha transition slightly (optional)
             current_alpha = 0.95 * current_alpha + 0.05 * target_alpha
             filtered_breath[n] = current_alpha * filtered_breath[n-1] + (1 - current_alpha) * base_noise[n]

        return (vol * filtered_breath * amp_mod).astype(np.float32)

    # --- Add implementations for other new noise types ---
    elif noise_type == "digital_clipping_sim":
        # Generate loud noise/tone and clip it harshly
        base_sound = generate_tone(random.uniform(100, 5000), duration_samples, vol * random.uniform(3.0, 10.0)) # Generate loud sound
        return np.clip(base_sound, -0.99, 0.99) # Hard clip

    elif noise_type == "feedback_screech":
         # Simulate feedback loop with delay and distortion
         delay_samples = random.randint(int(SAMPLE_RATE*0.01), int(SAMPLE_RATE*0.1))
         feedback_amount = random.uniform(0.9, 0.99) # High feedback
         noise = np.zeros(duration_samples, dtype=np.float32)
         # Start with a small noise burst
         noise[0:100] = (np.random.rand(100) - 0.5) * 0.1
         for n in range(delay_samples, duration_samples):
              feedback_sample = noise[n - delay_samples] * feedback_amount
              # Apply distortion to feedback
              feedback_sample = np.clip(feedback_sample * random.uniform(1.5, 5.0), -1.0, 1.0)
              noise[n] = feedback_sample + (random.random() - 0.5) * 0.01 # Add tiny noise
         # Normalize
         max_abs = np.max(np.abs(noise))
         if max_abs > 1e-6: noise /= max_abs
         return vol * noise

    # Add granular_flesh, choking_sim, modem_sim, granular_synth etc.
    # These would require more complex implementations.

    else: # Default fallback to white noise if type is unknown
        print(f"Warning: Unknown noise type '{noise_type}', using white noise.")
        return vol * (2 * np.random.random(duration_samples) - 1).astype(np.float32)


# --- Melody Generation ---
def generate_melody_fragment(duration_samples, vol, theme_data):
    """Generates a short, simple melodic fragment based on theme scale."""
    if duration_samples <= 0: return np.array([], dtype=np.float32) # Handle zero duration

    scale_name = theme_data.get("melody_scale", "minor_pentatonic")
    scale_intervals = SCALES.get(scale_name, SCALES["minor_pentatonic"])
    base_freq = random.uniform(80, 440) # Base note frequency (A2 to A4)
    notes_in_fragment = random.randint(3, 8)
    # Ensure note duration is at least 1 sample
    note_duration_samples = max(1, duration_samples // notes_in_fragment)

    melody = np.zeros(duration_samples, dtype=np.float32)
    current_pos = 0

    for _ in range(notes_in_fragment):
        if current_pos >= duration_samples: break
        # Choose note from scale intervals relative to root (0)
        scale_degree = random.choice(scale_intervals)
        octave_shift = random.choice([0, 0, 0, 1, -1]) # Chance to jump octave
        # Calculate frequency: Root * 2^(semitones/12)
        note_freq = base_freq * (2**((scale_degree + octave_shift * 12) / 12.0))
        note_freq = np.clip(note_freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF) # Clamp frequency

        # Generate tone for the note
        actual_note_len = min(note_duration_samples, duration_samples - current_pos)
        if actual_note_len <= 0: break

        # Use complex tone gen for melody texture
        note_tone = generate_tone(note_freq, actual_note_len, 1.0, fm_chance=0.2, harmonic_chance=0.6) # Generate at full volume first

        # Ensure generated tone length matches expected length
        if len(note_tone) == actual_note_len:
            melody[current_pos : current_pos + actual_note_len] = note_tone
        elif len(note_tone) > 0: # If length mismatch, truncate or pad
             len_to_copy = min(len(note_tone), actual_note_len)
             melody[current_pos : current_pos + len_to_copy] = note_tone[:len_to_copy]


        current_pos += actual_note_len

    # Apply heavy distortion specific to melody
    distortion_amount = theme_data.get("melody_distortion", 10.0)
    melody = apply_distortion(melody, intensity_range=(distortion_amount * 0.8, distortion_amount * 1.2))

    # Maybe add bitcrushing too?
    if random.random() < 0.5:
         melody = apply_bitcrush(melody, bit_depth_range=(3, 7))

    # Apply overall volume to the processed melody
    melody *= vol

    return melody


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
    # Use a unique temp name to avoid conflicts if run multiple times quickly
    video_temp_path = f"video_temp_{int(time.time())}_{random.randint(100,999)}.mp4"
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
        # Encode some words randomly
        if random.random() < 0.2: word = base64.b64encode(str(word).encode('utf-8', 'ignore')).decode()
        elif random.random() < 0.15: word = binascii.hexlify(str(word).encode('utf-8', 'ignore')).decode()

        persistent_texts.append({
            "text": str(word) + str(symbol), # Ensure parts are strings
            "pos": (random.randint(-50, WIDTH - 50), random.randint(-50, HEIGHT - 20)), # Allow off-screen start
            "font_size": random.uniform(0.5, 4.5), # Wider size range
            "color": random.choice(random.choice(theme_data["colors"])), # Use theme colors
            "lifetime": random.randint(int(FPS * 0.1), int(FPS * 8)), # Wider lifetime
            "frame_count": 0,
            "move_speed": (random.uniform(-8, 8), random.uniform(-6, 6)), # Faster movement
            # "rotation_speed": random.uniform(-5, 5) # Rotation adds complexity, skip for now
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
                # Select an effect function name from the probabilities dict
                effect_func_name = random.choice(list(current_params["effect_probabilities"].keys()))
                old_prob = current_params["effect_probabilities"][effect_func_name]
                # Modify probability more significantly
                new_prob = np.clip(old_prob + random.uniform(-0.3, 0.3), 0.01, 0.99) # Wider change range
                current_params["effect_probabilities"][effect_func_name] = new_prob
                # print(f"INSTABILITY (Visual): Changed {effect_func_name} probability to {new_prob:.2f}") # Optional Debug
            elif isinstance(current_params[param_key], tuple) and len(current_params[param_key]) == 2:
                # Modify parameter ranges more significantly
                current_min, current_max = current_params[param_key]
                # Ensure range width calculation is safe
                range_width = abs(current_max - current_min) + 1e-6
                # Shift the midpoint more
                mid_shift = random.uniform(-range_width * 0.2, range_width * 0.2) # Wider shift
                # Scale the range width more
                scale_factor = random.uniform(0.7, 1.3) # Wider scale change
                new_mid = (current_min + current_max) / 2 + mid_shift
                new_width = range_width * scale_factor
                new_min = new_mid - new_width / 2
                new_max = new_mid + new_width / 2
                # Add specific bounds checks if needed (e.g., alpha between 0 and 1)
                if "alpha" in param_key:
                    new_min = max(0.01, new_min)
                    new_max = min(1.0, max(new_min + 0.05, new_max))
                elif "shift" in param_key:
                     new_min = int(new_min)
                     new_max = int(max(new_min + 1, new_max))
                # Update the parameter range
                current_params[param_key] = (new_min, new_max)
                # print(f"INSTABILITY (Visual): Changed {param_key} range to ({new_min:.2f}, {new_max:.2f})") # Optional Debug


        # --- Frame Stutter/Hold ---
        if frame_hold_counter > 0:
            if held_frame is not None:
                video.write(held_frame)
                frame_hold_counter -= 1
                # Optional: print progress less frequently during holds
                # if (i + 1) % (FPS * 10) == 0: print(f"Video Frame {i+1}/{FRAME_COUNT} (Holding)...")
                continue # Skip generating a new frame
            else: # Reset if held_frame is somehow None
                frame_hold_counter = 0

        # Chance to hold the *next* frame to be generated
        if random.random() < 0.03 and frame_hold_counter == 0: # Slightly higher chance
            frame_hold_counter = random.randint(1, int(FPS * 0.7)) # Potentially longer hold
            held_frame = None # Clear previous held frame


        # --- Base Frame ---
        # More variation: occasional gradients or noise base
        base_roll = random.random()
        if base_roll < 0.3: # Solid Color
             # Ensure theme_data['colors'] is not empty and contains valid tuples
             if theme_data.get("colors") and theme_data["colors"][0]:
                  bg_color = random.choice(random.choice(theme_data["colors"]))
                  if isinstance(bg_color, tuple) and len(bg_color) == 3:
                       frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
                  else: # Fallback if color format is wrong
                       frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
             else: # Fallback if theme colors are missing
                  frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        elif base_roll < 0.5: # Gradient
             if theme_data.get("colors") and len(theme_data["colors"]) >= 1 and len(theme_data["colors"][0]) >= 2:
                  color1 = random.choice(theme_data["colors"][0])
                  color2 = random.choice(theme_data["colors"][0])
                  if isinstance(color1, tuple) and len(color1) == 3 and isinstance(color2, tuple) and len(color2) == 3:
                      frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                      c1_arr = np.array(color1)
                      c2_arr = np.array(color2)
                      # Create gradient using broadcasting (faster)
                      ratios = np.linspace(0, 1, HEIGHT)[:, np.newaxis] # Column vector
                      gradient_colors = (c1_arr * (1 - ratios) + c2_arr * ratios).astype(np.uint8)
                      frame[:, :, :] = gradient_colors[:, np.newaxis, :] # Assign to frame
                      # Randomly rotate gradient
                      if random.random() < 0.5: frame = cv2.rotate(frame, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]))
                  else: # Fallback if color format wrong
                       frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
             else: # Fallback if not enough colors for gradient
                  frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        else: # Noise Base
             frame = np.random.randint(0, 50, (HEIGHT, WIDTH, 3), dtype=np.uint8) # Dark noise


        # --- Apply Visual Effects ---
        # Get available effects for the current theme (list of function names)
        available_effect_names = theme_data.get("visual_effects", list(current_params["effect_probabilities"].keys()))
        # Map names to actual function objects, filtering out any names not found
        effect_candidates = [globals()[name] for name in available_effect_names if name in globals() and callable(globals()[name])]

        # More effects, weighted by intensity and avoiding too much repetition
        num_effects_to_apply = min(len(effect_candidates), random.randint(2, 5) + int(current_intensity * 2.5))

        applied_effects_count = 0
        random.shuffle(effect_candidates)
        applied_this_frame = []

        for effect_func in effect_candidates:
            if applied_effects_count >= num_effects_to_apply: break
            effect_name = effect_func.__name__ # Get function name string

            # Check history to avoid applying the same effect many times in a row
            recent_count = effect_history.count(effect_name)
            # Skip if used frequently recently, but allow occasional bursts
            if recent_count > 2 and random.random() < 0.6: continue

            # Calculate probability based on current params and intensity
            base_probability = current_params["effect_probabilities"].get(effect_name, 0.6) # Default higher prob
            probability_modifier = 0.4 + current_intensity * 0.6 # Intensity has strong effect
            final_probability = np.clip(base_probability * probability_modifier, 0.05, 0.95) # Higher min/max prob

            if random.random() < final_probability:
                try:
                    # --- Apply effect, scaling parameters by current_intensity ---
                    # Retrieve current parameter ranges from the (potentially unstable) current_params dict
                    # Apply intensity scaling to the *ranges* before picking a value

                    # Get signature to check parameters needed by the effect function
                    sig = inspect.signature(effect_func)
                    params_to_pass = {}

                    # Handle effects needing prev_frame
                    if 'prev_frame' in sig.parameters:
                        params_to_pass['prev_frame'] = prev_frame

                    # Handle effects with parameter ranges defined in current_params
                    scaled_ranges = {}
                    for param_name, range_tuple in current_params.items():
                        if param_name in sig.parameters and isinstance(range_tuple, tuple) and len(range_tuple) == 2:
                            p_min, p_max = range_tuple
                            # Apply intensity scaling (example: linear scaling)
                            # Adjust scaling logic per parameter as needed
                            scaled_min = p_min * (1.0 + (current_intensity - 1.0) * 0.5) # Scale around 1.0
                            scaled_max = p_max * (1.0 + (current_intensity - 1.0) * 0.5)
                            # Ensure min <= max after scaling
                            if scaled_min > scaled_max: scaled_min, scaled_max = scaled_max, scaled_min
                            # Add clipping/type casting if necessary based on parameter type
                            # This part needs careful handling per parameter type (float, int, etc.)
                            # Example for alpha ranges:
                            if "alpha" in param_name:
                                 scaled_min = np.clip(scaled_min, 0.01, 1.0)
                                 scaled_max = np.clip(scaled_max, scaled_min + 0.01, 1.0)
                            # Example for integer ranges (like shift amounts, block sizes):
                            elif any(k in param_name for k in ["shift", "block", "thresh"]):
                                 scaled_min = int(max(1, scaled_min)) # Ensure positive integer
                                 scaled_max = int(max(scaled_min + 1, scaled_max))
                            # Store the scaled range to pass to the function
                            scaled_ranges[param_name + "_range"] = (scaled_min, scaled_max)

                    # Pass scaled ranges to the function if it accepts them
                    params_to_pass.update(scaled_ranges)

                    # Call the effect function with the frame and any other required params
                    frame = effect_func(frame, **params_to_pass)

                    applied_effects_count += 1
                    applied_this_frame.append(effect_name)
                except Exception as e:
                    print(f"Error applying {effect_name} at frame {i}: {e}")
                    # print(f"Parameters passed: {params_to_pass}") # Debug parameters

        # Update effect history (keep last N effects)
        effect_history.extend(applied_this_frame)
        effect_history = effect_history[-20:] # Keep history of last 20 applied effects


        # --- Subliminal Flash (More Abrasive) ---
        flash_probability = 0.15 * current_intensity # Higher base chance
        if random.random() < flash_probability:
             flash_type = random.choice(["invert", "text", "symbol", "color", "noise", "contrast", "ascii_flash"])
             overlay = frame.copy() # Work on a copy for the flash
             try:
                if flash_type == "invert":
                    overlay = 255 - frame
                elif flash_type == "color":
                    if theme_data.get("colors") and theme_data["colors"][0]:
                         flash_col = random.choice(random.choice(theme_data["colors"]))
                         if isinstance(flash_col, tuple) and len(flash_col) == 3:
                              overlay = np.full((HEIGHT, WIDTH, 3), flash_col, dtype=np.uint8)
                elif flash_type == "noise":
                    noise_overlay = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
                    overlay = cv2.addWeighted(frame, 0.3, noise_overlay, 0.7, 0)
                elif flash_type == "contrast":
                     # Use more extreme contrast for flash
                     overlay = apply_extreme_contrast(frame, alpha_range=(5.0 * current_intensity, 10.0 * current_intensity), beta_range=(-150, 150))
                elif flash_type == "ascii_flash":
                      overlay = apply_ascii_sim(frame, block_size_range=(6,12), invert=random.choice([True,False]))
                else: # text or symbol flash
                    sub_text_content = random.choice(theme_data["words"]) if flash_type == "text" else random.choice(theme_data["symbols"])
                    sub_text = str(sub_text_content) # Ensure string
                    # Scale size with intensity
                    font_scale = random.uniform(5.0, 10.0) * (0.5 + current_intensity * 0.5)
                    thickness = random.randint(5, 10)
                    # Use default font for simplicity here
                    text_size, _ = cv2.getTextSize(sub_text, FONT_FACE_TO_USE, font_scale, thickness)
                    text_x = max(0, (WIDTH - text_size[0]) // 2)
                    text_y = max(0, (HEIGHT + text_size[1]) // 2)
                    # Choose high contrast color
                    flash_col = (0,0,0) if np.mean(frame) > 128 else (255,255,255)
                    shadow_col = (255,255,255) if np.mean(frame) > 128 else (0,0,0)
                    # Add outline/shadow for visibility
                    cv2.putText(overlay, sub_text, (text_x+3, text_y+3), FONT_FACE_TO_USE, font_scale, shadow_col, thickness+2, cv2.LINE_AA)
                    cv2.putText(overlay, sub_text, (text_x, text_y), FONT_FACE_TO_USE, font_scale, flash_col, thickness, cv2.LINE_AA)

                # Blend flash more strongly and randomly
                frame = cv2.addWeighted(frame, random.uniform(0.0, 0.3), overlay, random.uniform(0.7, 1.0), 0)
             except Exception as e:
                  print(f"Error during subliminal flash ({flash_type}): {e}")


        # --- Text Rendering (More Chaotic) ---
        current_persistent_texts = []
        frame_overlay = frame.copy() # Draw text on a separate overlay
        # Update and draw persistent texts
        for text_info in persistent_texts:
            # Update position based on move_speed
            px, py = text_info["pos"]
            mx, my = text_info["move_speed"]
            text_info["pos"] = (int(px + mx), int(py + my))

            # Bounce logic (simplified: reverse speed at edges)
            pos_x, pos_y = text_info["pos"]
            # Rough estimate for text bounds (can be inaccurate for complex fonts/text)
            text_width_estimate = int(text_info["font_size"] * 20)
            text_height_estimate = int(text_info["font_size"] * 40)
            if pos_x < -text_width_estimate or pos_x > WIDTH:
                text_info["move_speed"] = (-mx * random.uniform(0.8, 1.2), my + random.uniform(-2, 2))
            if pos_y < -text_height_estimate or pos_y > HEIGHT:
                 text_info["move_speed"] = (mx + random.uniform(-2, 2), -my * random.uniform(0.8, 1.2))

            # Keep track of lifetime
            text_info["frame_count"] += 1

            # Draw if still alive
            if text_info["frame_count"] < text_info["lifetime"]:
                try:
                    # Scale font size slightly with intensity
                    display_font_size = text_info["font_size"] * (0.7 + current_intensity * 0.3)
                    display_font_size = max(0.1, display_font_size) # Ensure minimum size
                    thickness = max(1, int(display_font_size)) # Thickness scales with size

                    # Use the color stored in the text_info
                    color = text_info["color"]
                    if not (isinstance(color, tuple) and len(color) == 3): # Fallback color
                        color = (255, 255, 255)

                    # Add slight jitter to position for visual noise
                    jitter_x = random.randint(-2, 2)
                    jitter_y = random.randint(-2, 2)
                    pos = (text_info["pos"][0] + jitter_x, text_info["pos"][1] + jitter_y)

                    # Draw the text
                    cv2.putText(frame_overlay, text_info["text"], pos, FONT_FACE_TO_USE,
                                display_font_size, color, thickness, cv2.LINE_AA)
                    current_persistent_texts.append(text_info) # Keep it for next frame
                except Exception as e:
                    # print(f"Error drawing persistent text '{text_info['text']}': {e}") # Optional debug
                    current_persistent_texts.append(text_info) # Keep it anyway, maybe it works next time
            else: # Lifetime expired, decide whether to respawn
                if random.random() < 0.6: # Higher respawn chance
                    # Reset position, lifetime, frame count
                    text_info["pos"] = (random.randint(-50, WIDTH - 50), random.randint(-50, HEIGHT - 20))
                    text_info["frame_count"] = 0
                    text_info["lifetime"] = random.randint(int(FPS * 0.1), int(FPS * 8))
                    # Change text content, color, speed on respawn
                    word = random.choice(theme_data["words"])
                    symbol = random.choice(theme_data["symbols"])
                    if random.random() < theme_data.get("zalgo_words_chance", 0.3): word = zalgo_text(word)
                    if random.random() < 0.1: word = base64.b64encode(str(word).encode('utf-8','ignore')).decode()
                    text_info["text"] = str(word) + str(symbol)
                    text_info["color"] = random.choice(random.choice(theme_data["colors"]))
                    text_info["move_speed"] = (random.uniform(-8, 8), random.uniform(-6, 6))
                    current_persistent_texts.append(text_info) # Add back to list

        persistent_texts = current_persistent_texts # Update the list

        # Fleeting text (more frequent and numerous)
        if random.random() < 0.6 + current_intensity * 0.2: # Much more likely
             num_fleet = random.randint(1, int(5 + current_intensity * 5)) # Many fleeting texts
             for _ in range(num_fleet):
                  try:
                      fleet_text_content = random.choice(theme_data["words"]) + " " + random.choice(theme_data["symbols"])
                      fleet_text = str(fleet_text_content) # Ensure string
                      # Higher chance for Zalgo
                      if random.random() < theme_data.get("zalgo_words_chance", 0.3) + 0.2:
                          fleet_text = zalgo_text(fleet_text)
                      font_size = random.uniform(0.4, 2.5) * (0.8 + current_intensity * 0.2) # Scale size
                      font_size = max(0.1, font_size)
                      thickness = max(1, int(font_size))
                      position = (random.randint(5, WIDTH - 80), random.randint(20, HEIGHT - 20))
                      # Use theme colors
                      text_color = random.choice(random.choice(theme_data["colors"]))
                      if not (isinstance(text_color, tuple) and len(text_color) == 3): text_color = (200,200,200) # Fallback
                      cv2.putText(frame_overlay, fleet_text, position, FONT_FACE_TO_USE, font_size, text_color, thickness, cv2.LINE_AA)
                  except Exception as e:
                      # print(f"Error drawing fleeting text: {e}") # Optional debug
                      pass # Ignore errors for fleeting text

        # Blend text overlay (maybe vary alpha based on intensity?)
        text_alpha = np.clip(0.6 + current_intensity * 0.2, 0.3, 1.0)
        # Blend text overlay onto the main frame
        frame = cv2.addWeighted(frame, 1.0, frame_overlay, text_alpha, 0)


        # --- Store frame for feedback loop & Write to video ---
        prev_frame = frame.copy() # Store state *before* potential hold frame is saved
        if frame_hold_counter > 0 and held_frame is None:
            held_frame = frame.copy() # Store the final frame state if holding starts now

        # Write the final frame (or the held frame if applicable, handled at loop start)
        video.write(frame)

        # --- Progress Indicator ---
        if (i + 1) % (FPS * 2) == 0 and frame_hold_counter == 0: # Update less often
            print(f"Video Frame {i+1}/{FRAME_COUNT} (Intensity: {current_intensity:.2f})...")

    # --- Cleanup ---
    video.release()
    print("Video generation complete.")
    return DURATION, FPS, video_temp_path


# --- Audio Generation (Updated for melody, abrasiveness) ---
def generate_audio_enhanced(video_duration, video_fps, theme_data, intensity_func, global_params):
    """Generates audio with buried melody, more abrasive, linked to intensity."""
    DURATION = video_duration
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    if NUM_SAMPLES <= 0:
        print("Error: Calculated audio duration is zero or negative.")
        return None
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
        # Get intensity, ensuring it's valid
        try:
            current_intensity = intensity_func(current_sample_time)
        except ValueError: # Handle potential issues with interpolation times
            current_intensity = 1.0 # Default intensity if time is out of bounds
        current_intensity = np.clip(current_intensity, 0.05, 3.0) # Wider clamp

        # --- Optional: Audio Parameter Instability ---
        if random.random() < audio_instability_chance:
             param_key = random.choice(list(current_audio_params.keys()))
             if isinstance(current_audio_params[param_key], tuple) and len(current_audio_params[param_key]) == 2:
                  # Modify audio parameter ranges (similar logic to visual instability)
                  current_min, current_max = current_audio_params[param_key]
                  range_width = abs(current_max - current_min) + 1e-6
                  mid_shift = random.uniform(-range_width * 0.15, range_width * 0.15)
                  scale_factor = random.uniform(0.8, 1.2)
                  new_mid = (current_min + current_max) / 2 + mid_shift
                  new_width = range_width * scale_factor
                  new_min = new_mid - new_width / 2
                  new_max = new_mid + new_width / 2
                  # Add bounds checks (e.g., bit depth min 2)
                  if "depth" in param_key: new_min = max(2, int(new_min))
                  current_audio_params[param_key] = (new_min, new_max)
                  # print(f"INSTABILITY (Audio): Changed {param_key} range to ({new_min:.2f}, {new_max:.2f})") # Optional Debug


        # --- Determine Event ---
        # Make events more overlapping and denser with intensity
        event_max_duration = max(0.03, 2.0 / (current_intensity + 0.8)) # Shorter max duration
        event_min_duration = 0.005 # Allow very short events
        event_duration_sec = random.uniform(event_min_duration, event_max_duration)
        event_duration_samples = int(event_duration_sec * SAMPLE_RATE)
        # Ensure duration is positive and doesn't exceed remaining samples
        event_duration_samples = min(max(1, event_duration_samples), NUM_SAMPLES - last_event_end_sample)
        if event_duration_samples <= 0: break # Stop if no room left

        event_start_sample = last_event_end_sample
        event_end_sample = event_start_sample + event_duration_samples

        # Volume scaling - more aggressive
        base_vol = random.uniform(0.03, 0.4) # Higher max base
        # Stronger intensity scaling, allow full volume potential, ensure positive
        event_vol = np.clip(abs(base_vol * (0.3 + current_intensity * 1.2)), 0.005, 1.0)

        # Choose event type: Noise, Tone, Melody, Silence
        type_roll = random.random()
        silence_thresh = max(0.01, 0.15 * (1.8 - current_intensity)) # Less silence overall
        melody_thresh = silence_thresh + 0.15 # Fixed chance for melody? Or intensity based? Let's try fixed.
        noise_thresh = melody_thresh + (0.5 + 0.3 * current_intensity) # More noise when intense

        segment_mono = np.zeros(event_duration_samples, dtype=np.float32)
        is_melody = False

        if type_roll < silence_thresh:
            event_type = "silence"
        elif type_roll < melody_thresh:
            event_type = "melody"
            is_melody = True
            melody_vol = event_vol * 0.4 # Keep melody quieter initially to be buried
            segment_mono = generate_melody_fragment(event_duration_samples, melody_vol, theme_data)
        elif type_roll < noise_thresh:
            event_type = "noise"
            noise_choice = random.choice(noise_types)
            # Bias towards harsher noises based on intensity
            if current_intensity > 1.5 and random.random() < 0.6:
                 harsh_noises = [n for n in noise_types if any(k in n for k in ["screech", "clip", "glitch", "artifact", "feedback"])]
                 if harsh_noises: noise_choice = random.choice(harsh_noises)
            segment_mono = generate_noise(noise_choice, event_duration_samples, event_vol, features)
        else:
            event_type = "tone"
            # Scale frequency range based on intensity
            current_max_freq = max_freq * (0.7 + current_intensity * 0.6)
            freq = random.uniform(min_freq, current_max_freq)
            freq = np.clip(freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF) # Clamp frequency
            # More FM/Harmonics for tones when intense
            fm_chance = np.clip(0.4 + current_intensity * 0.3, 0.1, 0.9)
            harmonic_chance = np.clip(0.3 + current_intensity * 0.2, 0.1, 0.8)
            segment_mono = generate_tone(freq, event_duration_samples, event_vol, fm_chance, harmonic_chance)

        # Ensure segment_mono has the correct length before applying effects
        if len(segment_mono) != event_duration_samples:
             # If length mismatch (e.g., from generate_tone error), pad or truncate
             correct_segment = np.zeros(event_duration_samples, dtype=np.float32)
             len_to_copy = min(len(segment_mono), event_duration_samples)
             if len_to_copy > 0:
                  correct_segment[:len_to_copy] = segment_mono[:len_to_copy]
             segment_mono = correct_segment


        if event_type != "silence":
            # --- Apply Audio Effects (More Abrasive) ---
            effect_chance_mod = 0.5 + current_intensity # Higher base chance for effects

            # Distortion more likely and intense
            if "distortion" in features and random.random() < 0.5 * effect_chance_mod:
                d_min, d_max = current_audio_params["distortion_intensity"]
                # Scale range more aggressively
                dist_range = (d_min * (0.8 + current_intensity), d_max * (1.0 + current_intensity * 1.2))
                segment_mono = apply_distortion(segment_mono, intensity_range=np.clip(dist_range, 1.0, 25.0)) # Higher max distortion

            # Bitcrush more likely and intense
            if "bitcrush" in features and random.random() < 0.4 * effect_chance_mod:
                 b_min, b_max = current_audio_params["bitcrush_depth"]
                 # Calculate target bits, inversely related to intensity, ensure minimum of 2
                 target_bits = int(max(2, b_min + (b_max - b_min) * (1.8 - current_intensity)))
                 segment_mono = apply_bitcrush(segment_mono, bit_depth_range=(2, max(3, target_bits))) # Allow 2-bit

            # Spectral Glitch
            if "spectral_glitch" in features and random.random() < 0.3 * effect_chance_mod:
                 segment_mono = apply_spectral_glitch(segment_mono, intensity=current_intensity)

            # Convolution Reverb (Use sparingly, check feature name)
            reverb_feature = [f for f in features if "convolution_reverb" in f]
            if reverb_feature and random.random() < 0.1 * effect_chance_mod: # Low chance
                 # Extract impulse type from feature name (e.g., "convolution_reverb_damp")
                 try:
                      impulse_name = reverb_feature[0].split("_")[-1]
                      if impulse_name in IMPULSE_RESPONSES:
                           segment_mono = apply_convolution_reverb(segment_mono, IMPULSE_RESPONSES[impulse_name])
                 except Exception as e:
                      # print(f"Error applying convolution reverb: {e}") # Optional debug
                      pass


            # --- Panning ---
            pan_extremity = np.clip(0.3 + current_intensity * 0.7, 0.1, 1.0) # Wider panning with intensity
            pan = random.uniform(-pan_extremity, pan_extremity)
            # Apply panning using square root law for constant power
            gain_l = np.sqrt(0.5 * (1 - pan))
            gain_r = np.sqrt(0.5 * (1 + pan))

            # Add to appropriate track (main stereo or separate melody)
            end_idx = min(event_end_sample, NUM_SAMPLES) # Ensure index is within bounds
            length = end_idx - event_start_sample
            if length <= 0: continue # Skip if length is zero

            if is_melody:
                 # Add processed melody to its dedicated mono track
                 melody_track[event_start_sample:end_idx] += segment_mono[:length]
            else:
                 # Add noise/tone to the main stereo samples array
                 samples[event_start_sample:end_idx, 0] += segment_mono[:length] * gain_l
                 samples[event_start_sample:end_idx, 1] += segment_mono[:length] * gain_r


        # Move time forward allowing more overlap based on intensity
        overlap_factor = np.clip(0.2 + current_intensity * 0.4, 0.05, 0.8) # More overlap allowed
        advance_samples = int(event_duration_samples * (1.0 - overlap_factor))
        last_event_end_sample += max(1, advance_samples) # Ensure progress by at least 1 sample

    # --- Mix Melody Track into Main Stereo Track ---
    # Normalize melody track separately first to control its level better
    max_melody_abs = np.max(np.abs(melody_track))
    if max_melody_abs > 1e-6:
         melody_track /= max_melody_abs # Normalize melody track to [-1, 1]

    melody_mix_level = 0.3 # How loud the normalized melody is in the final mix (adjust as needed)
    samples[:, 0] += melody_track * melody_mix_level
    samples[:, 1] += melody_track * melody_mix_level # Add mono melody equally to both channels


    # --- Final Normalization of the entire mix ---
    max_abs_amplitude = np.max(np.abs(samples))
    if max_abs_amplitude > 1e-6:
        print(f"Normalizing final audio mix (Max amplitude before: {max_abs_amplitude:.3f})")
        # Normalize to prevent clipping, scaling factor ensures peak is at 1.0
        normalization_factor = 1.0 / max_abs_amplitude
        samples *= normalization_factor
    elif np.any(samples): # If not totally silent but max is near zero
         print("Warning: Final audio mix is very quiet.")
    else: # If completely silent
        print("Warning: Final audio mix appears silent.")

    # Convert to 16-bit integers for WAV export
    # Clip values just in case they slightly exceed +/- 1 after operations
    samples_clipped = np.clip(samples, -1.0, 1.0)
    samples_int16 = (samples_clipped * 32767).astype(np.int16)

    # --- Write WAV file ---
    # Use a unique temp name
    audio_temp_path = f"audio_temp_{int(time.time())}_{random.randint(100,999)}.wav"
    try:
        with wave.open(audio_temp_path, 'wb') as wf:
            wf.setnchannels(NUM_CHANNELS)
            wf.setsampwidth(2) # 2 bytes for 16-bit audio
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples_int16.tobytes()) # Write the byte representation
        print(f"Temporary audio saved to {audio_temp_path}")
        return audio_temp_path
    except Exception as e:
        print(f"Error writing WAV file '{audio_temp_path}': {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # 1. Select Initial Theme
    # << SURPRISE: Theme can change mid-generation >>
    initial_theme_name = random.choice(list(THEMES.keys()))
    # Make a deep copy so modifications during generation don't affect original THEMES dict
    current_theme_data = copy.deepcopy(THEMES[initial_theme_name])

    print(f"--- Starting Avant-Garde Generation ---")
    print(f"Initial Theme: {initial_theme_name}")

    # 2. Setup Global Parameters (for instability and potential theme switching)
    # These parameters define the initial state and are modified by instability
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
                  # Map function objects to their names for easier modification lookup
                  func.__name__: prob for func, prob in [
                      (apply_perlin_noise, 0.7), (apply_block_shift, 0.6), (apply_color_channel_shift, 0.6),
                      (apply_warp, 0.5), (apply_pixelation, 0.4), (apply_scanlines, 0.3),
                      (apply_solarize, 0.2), (apply_extreme_contrast, 0.4), (apply_feedback, 0.5),
                      (apply_datamosh_sim, 0.2), (apply_ascii_sim, 0.15),
                      (apply_crt_ghost_sim, 0.1), (apply_slit_scan_sim, 0.1), (apply_vector_field_sim, 0.1) # Add new effect probs
                  ] if callable(func) # Ensure it's a callable function
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

    # << SURPRISE: Implement Style Break Logic >>
    # This is complex to integrate directly into the functions without major refactoring.
    # A simpler approach: Check for style break possibility *before* calling generation functions,
    # pass the potentially modified theme_data and global_params.
    # This is NOT implemented here yet but the framework exists.

    # 4. Generate Video
    # Pass the potentially modified theme_data and global_params
    actual_duration, actual_fps, video_temp_file = generate_frames_enhanced(current_theme_data, intensity_function, global_params)
    if video_temp_file is None:
        print("Video generation failed. Exiting.")
        exit(1)
    print(f"Video generation yielded: Duration={actual_duration:.2f}s, FPS={actual_fps}")

    # 5. Generate Audio
    # Pass the potentially modified theme_data and global_params
    audio_temp_file = generate_audio_enhanced(actual_duration, actual_fps, current_theme_data, intensity_function, global_params)
    if audio_temp_file is None:
        print("Audio generation failed. Exiting.")
        # Clean up video file if audio failed
        if video_temp_file and os.path.exists(video_temp_file):
             try: os.remove(video_temp_file)
             except OSError as e: print(f"Error removing temp video file {video_temp_file}: {e}")
        exit(1)


    # 6. Combine Video and Audio using ffmpeg
    if video_temp_file and audio_temp_file:
        print("Combining video and audio using ffmpeg...")
        # Ensure output file doesn't exist or ffmpeg might hang/fail
        if os.path.exists(OUTPUT_FILE):
            try:
                os.remove(OUTPUT_FILE)
            except OSError as e:
                print(f"Warning: Could not remove existing output file '{OUTPUT_FILE}': {e}")
                # Optionally, create a unique output filename instead
                # OUTPUT_FILE = f"avant_garde_output_{int(time.time())}.mp4"
                # print(f"Using new output filename: {OUTPUT_FILE}")

        # Construct ffmpeg command
        ffmpeg_command = [
            'ffmpeg', '-y', # Overwrite output without asking
            '-i', video_temp_file, # Input video
            '-i', audio_temp_file, # Input audio
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', # Re-encode video for compatibility
            '-c:a', 'aac', '-b:a', '256k', # Encode audio to AAC with higher bitrate
            '-shortest', # Finish encoding when the shortest input ends
            OUTPUT_FILE # Output file path
        ]
        print(f"Running ffmpeg: {' '.join(ffmpeg_command)}") # Print the command being run

        try:
            # Run ffmpeg command, capturing output and errors
            result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            # Optional: Print ffmpeg output if needed (can be very verbose)
            # print("ffmpeg Output:", result.stdout)
            # print("ffmpeg Error (if any):", result.stderr)
            print(f"Successfully combined streams into '{OUTPUT_FILE}'")
        except subprocess.CalledProcessError as e:
            # Handle errors during ffmpeg execution
            print(f"ffmpeg failed with error code {e.returncode}")
            # Print ffmpeg's error output for debugging
            # print("ffmpeg Output:", e.stdout) # stdout might also contain info
            print("ffmpeg Error:", e.stderr)
            print("Check if ffmpeg is installed correctly and accessible in your system's PATH.")
            print("Also check the temporary video/audio files for issues.")
        except FileNotFoundError:
             # Handle case where ffmpeg command itself is not found
             print("Error: ffmpeg command not found. Please install ffmpeg and ensure it's in your PATH.")
        except Exception as e:
            # Catch any other unexpected errors during ffmpeg execution
            print(f"An unexpected error occurred during ffmpeg execution: {e}")

    else:
        print("Skipping ffmpeg combination due to errors in video or audio generation.")

    # 7. Clean up temporary files
    print("Cleaning up temporary files...")
    for temp_file in [video_temp_file, audio_temp_file]:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError as e:
                print(f"Error removing temp file {temp_file}: {e}") # Report error but continue
    print("Temporary files cleanup attempted.")

    end_time = time.time()
    print(f"--- Generation Process Finished in {end_time - start_time:.2f} seconds ---")


