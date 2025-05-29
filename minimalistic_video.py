# -*- coding: utf-8 -*-
# Requires: pip install numpy opencv-python pygame noise scipy qrcode[pil] Pillow
import numpy as np
import cv2
import pygame
import random
import string
import wave
import os
import math
import time
import noise
import base64
import binascii
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft
import subprocess
import copy
import inspect
import qrcode # For ARG QR codes
from PIL import Image, ImageDraw, ImageFont # For QR code image manipulation

# --- Configuration ---
WIDTH, HEIGHT = 1280, 720 # HD Resolution
MIN_DURATION, MAX_DURATION = 25, 60 # Longer videos for more ARG content
MIN_FPS, MAX_FPS = 18, 33 # Faster, more fluid but still glitchable
OUTPUT_FILE = "shardmind_contact.mp4" # Thematic output name
HIGH_FREQ_CUTOFF = 22000
LOW_FREQ_CUTOFF = 18
SAMPLE_RATE = 44100
time_step = 1.0 / SAMPLE_RATE

# --- ARG Content Configuration ---
ARG_QR_SIZE = 100 # pixels
ARG_TEXT_FONT_SCALE_BASE = 0.7
ARG_ERROR_CODE_FONT_SCALE = 1.0

# --- Helper function for Zalgo text ---
def zalgo_text(text_input):
    """Applies combining diacritical marks randomly to text."""
    text = str(text_input) # Ensure input is a string
    # Expanded Zalgo characters for more visual chaos
    zalgo_chars_combined = [chr(i) for i in range(0x0300, 0x036F + 1)] # Combining Diacritical Marks
    # Add more ranges if desired, e.g., Combining Diacritical Marks Supplement (0x1DC0–0x1DFF)
    # zalgo_chars_supplement = [chr(i) for i in range(0x1DC0, 0x1DFF + 1)]
    # all_zalgo = zalgo_chars_combined + zalgo_chars_supplement
    all_zalgo = zalgo_chars_combined

    if not all_zalgo: return text # Should not happen with the current fixed range

    output = ""
    for char_original in text:
        output += char_original
        num_zalgo = random.randint(1, 10) # More Zalgo
        for _ in range(num_zalgo):
            output += random.choice(all_zalgo)
    return output

# --- THEMES (Expanded for ARG and Abrasiveness) ---
THEMES = {
    "Shardmind Protocol": {
        "colors": [
            [(0,0,0), (255,255,255), (0,100,255), (255,0,100)], # Black, White, Glitch Blue, Magenta
            [(10,10,30), (200,220,255), (50,200,50), (100,100,100)], # Dark Blue, Off-White, Terminal Green, Grey
        ],
        "words": ["TRANSMIT", "RECEIVE", "PACKET", "CORRUPT", "SIGNAL", "PROTOCOL", "ENTITY", "SHARD", "MIND", "QUERY", "RESPONSE", "DECODE", "ENCRYPT", "MANIFEST"],
        "symbols": ["::", "->", "||", "&&", "0x", "#>", "%$", "§", "∆", "∑", "Ω", "サイバー"], # Cyber in Japanese Katakana
        "zalgo_words_chance": 0.5,
        "visual_effects": ["apply_perlin_noise", "apply_block_shift", "apply_color_channel_shift", "apply_warp", "apply_pixelation", "apply_scanlines", "apply_extreme_contrast", "apply_feedback", "apply_datamosh_sim", "apply_ascii_sim", "apply_sensor_overload", "apply_flickering_glitch_layer"],
        "audio_freq_range": (20, 20000),
        "audio_noise_types": ["digital_artifact", "static", "modem_sim", "feedback_screech", "glitch", "white_bursts", "data_stream_sim"], # NOTE: Some of these are not fully implemented in generate_noise
        "audio_features": ["stutter", "bitcrush", "extreme_panning", "distortion", "spectral_glitch", "dtmf_bursts", "morse_fragments"],
        "melody_scale": "chromatic",
        "melody_distortion": 20.0,
        "melody_counter_melody_chance": 0.6,
        "arg_elements": {
            "fake_errors": ["ERR_SYNC_FAIL", "PACKET_CRC_MISMATCH", "MEM_ACCESS_VIOLATION_0XFFFE", "UNKNOWN_OPCODE", "ENTITY_STREAM_INTERRUPTED"],
            "cipher_keys": ["ALPHA", "OMEGA", "NULL", "VOID", "777", "13", "3301"], # Example keys
            "qr_data_prefix": "SHARD_LINK_",
            "text_code_types": ["base64", "hex", "binary_short"]
        }
    },
    "Flesh Interface": {
        "colors": [
            [(30,0,0), (150,20,20), (200,180,170), (80,50,40)], # Dark Blood, Flesh, Bone, Bruise
            [(0,0,0), (255,10,10), (100,5,5), (50,50,50)], # Black, Bright Blood, Dark Clot, Grey
        ],
        "words": ["MERGE", "INTEGRATE", "BIOFEEDBACK", "SYNAPSE", "NEURAL", "PULSE", "MUTATE", "CONSUME", "REJECT", "HOST", "PARASITE", "SYMBIOSIS"],
        "symbols": ["⚕️", "🧬", "🧠", "👁️", "〰️", "∬", "𝔅ℑ𝔒", "𝔐𝔈ℭℌ"], # Fraktur for Bio/Mech
        "zalgo_words_chance": 0.7,
        "visual_effects": ["apply_perlin_noise", "apply_warp", "apply_feedback", "apply_data_bleed_sim", "apply_solarize", "apply_crt_ghost_sim", "apply_figurative_noise_sim", "apply_pixelation"],
        "audio_freq_range": (18, 8000),
        "audio_noise_types": ["heartbeat", "breathing", "squelch", "wet_clicks", "bone_cracking_sim", "granular_flesh", "choking_sim", "biometric_rhythm"], # NOTE: Some of these are not fully implemented
        "audio_features": ["wet_sounds", "slow_lfo", "dissonance", "low_rumble", "distortion", "extreme_pitch_bend", "broken_speaker_sim", "convolution_reverb_damp"],
        "melody_scale": "phrygian",
        "melody_distortion": 12.0,
        "melody_counter_melody_chance": 0.4,
        "arg_elements": {
            "fake_errors": ["BIO_SYNC_ERR", "NEURAL_OVERLOAD", "REJECTION_IMMINENT", "HOST_VITALS_CRITICAL", "DNA_RESEQUENCE_FAIL"],
            "cipher_keys": ["SERUM", "GENE", "PULSE", "NERVE", "XENOS"],
            "qr_data_prefix": "SPECIMEN_LOG_",
            "text_code_types": ["hex", "morse_audio_short"]
        }
    },
}

# --- Scales for Buried Melody ---
SCALES = {
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": list(range(12)),
    "major": [0, 2, 4, 5, 7, 9, 11],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "augmented": [0, 4, 8],
    "whole_tone": [0, 2, 4, 6, 8, 10],
}

# --- Font Configuration ---
# DEFAULT_FONT is used for cv2.putText. TTF fonts with cv2.putText require FreeType.
# SELECTED_TTF_FONT was unused with cv2.putText, so removed for now.
# Pillow (for QR codes) can use TTF fonts if specified, but current QR generation doesn't use text.
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_FACE_TO_USE = DEFAULT_FONT # Consistently use this for OpenCV text

# --- Pygame Mixer Init ---
try:
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=4096)
except pygame.error as e:
    print(f"Pygame mixer init failed (non-critical for file output): {e}")

# --- Intensity Profile Generation ---
def generate_intensity_profile(duration_samples, sample_rate, scale=30.0, octaves=6, persistence=0.5, lacunarity=2.0):
    duration_seconds = duration_samples / sample_rate
    num_points = int(duration_seconds * 20)
    if num_points < 2: num_points = 2 # interp1d needs at least 2 points
    profile = np.zeros(num_points)
    base = random.randint(0, 1000)
    for i in range(num_points):
        n1 = noise.pnoise1(i / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base)
        n2 = noise.pnoise1(i / (scale*0.3) + 1000, octaves=3, persistence=0.7, lacunarity=1.8, base=base+1)
        profile[i] = (n1 * 0.7) + (n2 * 0.3)
    min_val, max_val = np.min(profile), np.max(profile)
    if max_val - min_val > 1e-6: # Avoid division by zero if profile is flat
        normalized_profile = (profile - min_val) / (max_val - min_val)
        intensity_min = 0.05 # Min perceived intensity
        intensity_max = 3.5  # Max perceived intensity (can be tuned)
        scaled_profile = intensity_min + normalized_profile * (intensity_max - intensity_min)
    else:
        scaled_profile = np.full(num_points, (0.05 + 3.5) / 2) # Average intensity if flat
    profile_times = np.linspace(0, duration_seconds, num_points)
    intensity_func = interp1d(profile_times, scaled_profile, kind='cubic', bounds_error=False,
                              fill_value=(scaled_profile[0], scaled_profile[-1]))
    # print(f"Generated shared intensity profile: {len(profile)} points over {duration_seconds:.1f}s (Range: {scaled_profile.min():.2f}-{scaled_profile.max():.2f})")
    return intensity_func

# --- ARG Content Generation Functions ---
def generate_qr_code_image(data, size=ARG_QR_SIZE):
    try:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=max(1, size // 20), border=1)
        qr.add_data(data)
        qr.make(fit=True)
        img_pil = qr.make_image(fill_color="white", back_color="black").convert('RGB')
        img_pil = img_pil.resize((size, size), Image.NEAREST) # Use Image.NEAREST for crisp pixels
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error generating QR code for '{data}': {e}")
        return np.zeros((size, size, 3), dtype=np.uint8) # Return blank image on error

def generate_arg_text_code(theme_data, length=16):
    code_type_options = theme_data.get("arg_elements", {}).get("text_code_types", ["hex"])
    if not code_type_options: code_type_options = ["hex"] # Fallback
    code_type = random.choice(code_type_options)

    actual_length = random.randint(max(1, length // 2), length) # Vary length more, ensure min 1

    try:
        if code_type == "base64":
            raw_data = os.urandom(actual_length * 3 // 4 + 1) # Ensure enough bytes for base64
            return base64.b64encode(raw_data).decode('utf-8')[:actual_length]
        elif code_type == "hex":
            raw_data = os.urandom(actual_length // 2 + 1) # Ensure enough bytes
            return binascii.hexlify(raw_data).decode('utf-8')[:actual_length]
        elif code_type == "binary_short":
            raw_data = os.urandom(actual_length // 8 + 1) # Ensure enough bytes for binary string
            return "".join(format(byte, '08b') for byte in raw_data)[:actual_length]
    except Exception as e:
        print(f"Error generating ARG text code type {code_type}: {e}")
    return "NULL_CODE_" + str(random.randint(100,999))


# --- Visual Glitch Functions ---
# Note: The dynamic parameter scaling for these effects based on `global_params["visual"]`
# and intensity might not be fully working as intended due to mismatches between
# parameter names in function signatures (e.g., `alpha_range`) and keys in `global_params`
# (e.g., `perlin_alpha`). This would require a refactor of parameter naming/access.
def apply_perlin_noise(frame, alpha_range=(0.1, 0.9), scale_range=(2.0, 80.0), oct_range=(3, 9)):
    alpha = random.uniform(*alpha_range)
    scale = random.uniform(*scale_range)
    octaves = random.randint(*oct_range)
    persistence = random.uniform(0.2, 0.7)
    lacunarity = random.uniform(1.8, 3.5)
    seed = random.randint(0, 2000)
    height, width = frame.shape[:2]
    
    # Generate coordinates once
    x_coords_norm = np.arange(width) / scale
    y_coords_norm = np.arange(height) / scale
    
    gray_noise = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        y_coord = y_coords_norm[i]
        for j in range(width):
            x_coord = x_coords_norm[j]
            gray_noise[i, j] = noise.pnoise2(y_coord, x_coord, octaves=octaves,
                                             persistence=persistence, lacunarity=lacunarity, base=seed)
    
    # Normalize and convert to BGR
    # cv2.normalize might be faster if noise range is consistent, but min/max handles variance
    min_gn, max_gn = np.min(gray_noise), np.max(gray_noise)
    if max_gn - min_gn > 1e-6:
        normalized_noise = (gray_noise - min_gn) / (max_gn - min_gn)
    else:
        normalized_noise = np.zeros_like(gray_noise)

    colored_noise = (normalized_noise * 255).astype(np.uint8)
    colored_noise = cv2.cvtColor(colored_noise, cv2.COLOR_GRAY2BGR)

    if random.random() < 0.4: colored_noise = 255 - colored_noise
    return cv2.addWeighted(frame, 1 - alpha, colored_noise, alpha, 0)

def apply_block_shift(frame, max_shift_range=(10, WIDTH // 3), block_size_range=(5, HEIGHT // 2), num_blocks_range=(20, 100)):
    max_shift = random.randint(max(1,int(max_shift_range[0])), max(2,int(max_shift_range[1])))
    bs_min = max(1, int(block_size_range[0]))
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    # block_size_max = random.randint(bs_min, bs_max) # This was unused, block_size is bh, bw
    num_blocks = random.randint(*num_blocks_range)
    height, width = frame.shape[:2]
    temp_frame = frame.copy()

    for _ in range(num_blocks):
        bh = random.randint(bs_min, bs_max) # Use bs_max directly from range
        bw = random.randint(bs_min, bs_max) # Use bs_max directly from range
        
        if height - bh <= 0 or width - bw <= 0: continue # Ensure valid dimensions for source block
        
        y = random.randint(0, height - bh -1) # Corrected: -1 to ensure y+bh is within bounds
        x = random.randint(0, width - bw -1)  # Corrected: -1
        
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        # Ensure target coordinates are valid for the block size
        target_y_start = np.clip(y + shift_y, 0, height - bh)
        target_x_start = np.clip(x + shift_x, 0, width - bw)
        
        try:
            block = frame[y:y+bh, x:x+bw]
            temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block
        except ValueError as e:
            # This can happen if block dimensions don't match due to clipping logic,
            # or if source/target slices are not perfectly aligned.
            # print(f"Block shift ValueError: {e}. Block shape: {block.shape}, Target pos: ({target_y_start},{target_x_start}), bh/bw: ({bh},{bw})")
            pass # Silently ignore if a block fails, to keep video generating
    return temp_frame

def apply_color_channel_shift(frame, max_shift_range=(5, 60)):
    max_shift = random.randint(max(1,int(max_shift_range[0])), max(2,int(max_shift_range[1])))
    temp_frame = frame.copy()
    for i in range(3): # Iterate over B, G, R channels
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
    amplitude_y = random.uniform(*amplitude_range) * random.uniform(0.3, 1.7) # Vary y relative to x
    frequency_y = random.uniform(*freq_range) * random.uniform(0.3, 1.7)   # Vary y relative to x
    phase_x = random.uniform(0, 2 * math.pi)
    phase_y = random.uniform(0, 2 * math.pi)

    x_mesh, y_mesh = np.meshgrid(np.arange(cols), np.arange(rows))
    
    offset_x = (amplitude_x * np.sin(2 * math.pi * y_mesh * frequency_x + phase_x)).astype(np.float32)
    offset_y = (amplitude_y * np.cos(2 * math.pi * x_mesh * frequency_y + phase_y)).astype(np.float32) # Using cos for y for variety

    map_x = np.clip(x_mesh.astype(np.float32) + offset_x, 0, cols - 1)
    map_y = np.clip(y_mesh.astype(np.float32) + offset_y, 0, rows - 1)
    
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def apply_feedback(frame, prev_frame, alpha_range=(0.02, 0.6)):
    if prev_frame is None: return frame
    alpha = random.uniform(*alpha_range)
    modified_prev = prev_frame.copy()

    # Apply some transformation to the feedback frame for more interesting effects
    if random.random() < 0.2: # Chance to slightly transform feedback
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
    
    if random.random() < 0.1: # Chance to slightly color shift feedback
         # Ensure addition does not overflow/underflow by using cv2.add with saturation
         shift_b = random.randint(-15,15)
         shift_g = random.randint(-15,15)
         shift_r = random.randint(-15,15)
         color_shift_matrix = np.full_like(modified_prev, (shift_b, shift_g, shift_r), dtype=np.int16) # Use int16 for intermediate
         modified_prev_shifted = np.clip(modified_prev.astype(np.int16) + color_shift_matrix, 0, 255).astype(np.uint8)
         modified_prev = modified_prev_shifted
         # modified_prev = cv2.add(modified_prev, (random.randint(-15,15),random.randint(-15,15),random.randint(-15,15),0), dtype=cv2.CV_8UC3) # Original, check alpha channel if frame has it

    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)

def apply_pixelation(frame, block_size_range=(4, 96)):
    height, width = frame.shape[:2]
    bs_min = max(2, int(block_size_range[0])) 
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    if bs_min >= bs_max: # Ensure bs_max is greater than bs_min
        block_size = bs_min
    else:
        block_size = random.randint(bs_min, bs_max)

    # Ensure block_size is not zero or negative
    block_size = max(1, block_size)

    pixel_w, pixel_h = max(1, width // block_size), max(1, height // block_size)
    temp = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def apply_scanlines(frame, intensity_range=(0.4, 1.0), thickness_range=(1, 4), color_variation=40):
    intensity = random.uniform(*intensity_range)
    thickness = random.randint(*thickness_range)
    scanline_layer = np.zeros_like(frame) # Initialize with zeros
    height, width = frame.shape[:2]
    base_color_val = random.randint(0, 70) # Dark scanlines

    for y_coord in range(0, height, thickness * 2): # thickness * 2 for gap
        line_color_val = base_color_val + random.randint(-color_variation, color_variation)
        line_color = tuple(np.clip([line_color_val]*3, 0, 255)) # Ensure color is valid
        cv2.line(scanline_layer, (0, y_coord), (width, y_coord), line_color, thickness)
    
    # Blend scanlines with original frame
    # The original gamma value was negative, which is unusual for cv2.addWeighted's gamma.
    # A common way to make scanlines darker:
    # frame * (1-intensity) + scanline_layer * intensity
    # Or, if scanline_layer contains dark lines on black, can subtract or use multiply blend.
    # Using addWeighted with adjusted alpha for scanlines:
    return cv2.addWeighted(frame, 1.0, scanline_layer, intensity, 0)


def apply_solarize(frame, threshold_range=(60, 200)):
    threshold = random.randint(*threshold_range)
    # Solarize effect: pixels above threshold are inverted
    solarized_frame = frame.copy()
    mask = frame > threshold # Create a boolean mask for all channels
    solarized_frame[mask] = 255 - solarized_frame[mask]
    return solarized_frame

def apply_extreme_contrast(frame, alpha_range=(1.0, 6.0), beta_range=(-100, 100)):
    alpha = random.uniform(*alpha_range) # Contrast control
    beta = random.randint(*beta_range)   # Brightness control
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_datamosh_sim(frame, prev_frame, hold_prob=0.3, smear_prob=0.5, block_size_range=(32, 128)):
    if prev_frame is None: return frame
    height, width = frame.shape[:2]
    output = frame.copy()
    
    bs_min = max(16, int(block_size_range[0]))
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    if bs_min >= bs_max: block_size = bs_min
    else: block_size = random.randint(bs_min, bs_max)
    block_size = max(1, block_size) # Ensure positive

    # Hold blocks from previous frame
    if random.random() < hold_prob:
        num_hold_blocks = random.randint(1, max(1, (width * height) // (block_size**2 * 3) )) # Approx 1/3 of blocks
        for _ in range(num_hold_blocks):
            if width // block_size <=0 or height // block_size <=0: continue
            bx_idx = random.randint(0, max(0, width // block_size - 1))
            by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            
            bh = min(block_size, height - by)
            bw = min(block_size, width - bx)
            if bh > 0 and bw > 0:
                 output[by:by+bh, bx:bx+bw] = prev_frame[by:by+bh, bx:bx+bw]
    
    # Smear blocks from previous frame
    if random.random() < smear_prob:
        num_smear_blocks = random.randint(1, max(1, (width * height) // (block_size**2 * 4) )) # Approx 1/4 of blocks
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
            
            smear_block_src = prev_frame[by:by+bh, bx:bx+bw] # Source from prev_frame
            
            target_y = np.clip(by + vy, 0, height - bh)
            target_x = np.clip(bx + vx, 0, width - bw)
            
            # Ensure smear_block and target slice have same dimensions
            if smear_block_src.shape[0] == bh and smear_block_src.shape[1] == bw:
                 output[target_y:target_y+bh, target_x:target_x+bw] = smear_block_src
    return output

def apply_ascii_sim(frame, block_size_range=(8, 16), char_set=" .:-=+*#%@", invert=False):
    height, width = frame.shape[:2]
    bs_min = max(4, int(block_size_range[0]))
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    if bs_min >= bs_max: block_size = bs_min
    else: block_size = random.randint(bs_min, bs_max)
    block_size = max(1, block_size)

    output = frame.copy() # Draw on a copy
    font_scale = max(0.1, block_size / 15.0) # Scale font with block size
    thickness = 1
    chars = list(char_set)
    if invert: chars = chars[::-1]
    num_chars = len(chars)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for y_coord in range(0, height - block_size + 1, block_size):
        for x_coord in range(0, width - block_size + 1, block_size):
            block = gray[y_coord:y_coord+block_size, x_coord:x_coord+block_size]
            avg_brightness = np.mean(block)
            char_index = int(np.clip((avg_brightness / 255.0) * num_chars, 0, num_chars - 1))
            char_to_draw = chars[char_index]
            
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255) # Black or white based on brightness
            
            text_size_cv, _ = cv2.getTextSize(char_to_draw, FONT_FACE_TO_USE, font_scale, thickness)
            pos = (x_coord + (block_size - text_size_cv[0]) // 2, 
                   y_coord + (block_size + text_size_cv[1]) // 2)
            try:
                cv2.putText(output, char_to_draw, pos, FONT_FACE_TO_USE, font_scale, text_color, thickness, cv2.LINE_AA)
            except Exception as e:
                # print(f"Error in cv2.putText for ASCII sim: {e}")
                pass # Ignore if specific char fails
    return output

def apply_sensor_overload(frame, intensity): # Intensity here is the global intensity_func(time)
    output = frame.copy()
    # Chromatic Aberration
    max_shift = int(10 + 40 * intensity) # Scale shift with intensity
    temp_aberration = output.copy() # Work on a copy for aberration
    for i in range(3): # B, G, R
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        channel = output[:,:,i]
        shifted = np.roll(np.roll(channel, shift_y, axis=0), shift_x, axis=1)
        # Blend original with shifted for a softer aberration
        temp_aberration[:,:,i] = cv2.addWeighted(channel, 0.3, shifted, 0.7,0)
    output = temp_aberration

    # Bloom / Extreme Contrast
    if random.random() < 0.5 * intensity:
        alpha_contrast = 2.0 + 3.0 * intensity # Higher contrast
        beta_brightness = random.randint(-50, 50) * intensity # Brightness shift
        output_contrasted = cv2.convertScaleAbs(output, alpha=alpha_contrast, beta=beta_brightness)
        
        # Bloom effect on very bright areas
        bright_mask_gray = cv2.cvtColor(output_contrasted, cv2.COLOR_BGR2GRAY)
        _, bright_mask_thresh = cv2.threshold(bright_mask_gray, 230, 255, cv2.THRESH_BINARY)
        
        # Kernel size for GaussianBlur must be odd. (0,0) lets it be derived from sigma.
        sigma_val = max(1, int(10 + 20 * intensity)) # Sigma must be positive
        bloom_effect = cv2.GaussianBlur(output_contrasted, (0,0), sigmaX=sigma_val, sigmaY=sigma_val)
        
        # Apply bloom only to bright areas or blend it
        # output = cv2.addWeighted(output_contrasted, 1.0, bloom_effect, 0.3 * intensity, 0) # Original way
        # More controlled bloom:
        bloom_masked = cv2.bitwise_and(bloom_effect, bloom_effect, mask=bright_mask_thresh)
        output = cv2.addWeighted(output_contrasted, 1.0, bloom_masked, 0.5 * intensity, 0)


    # Dead pixels / Static
    if random.random() < 0.2 * intensity:
        overlay_static = np.zeros_like(output, dtype=np.uint8)
        num_dots = random.randint(1, int(5 + 20 * intensity)) # More dots with intensity
        for _ in range(num_dots):
            dot_color_val = random.randint(10,40)
            cv2.circle(overlay_static, 
                       (random.randint(0,WIDTH-1), random.randint(0,HEIGHT-1)), 
                       random.randint(1,3), # Smaller dots
                       (dot_color_val,dot_color_val,dot_color_val), 
                       -1)
        if random.random() < 0.1 * intensity: # Chance for a glitch line
            cv2.line(overlay_static, 
                     (random.randint(0,WIDTH-1), random.randint(0,HEIGHT-1)), 
                     (random.randint(0,WIDTH-1), random.randint(0,HEIGHT-1)), 
                     (random.randint(5,20),random.randint(5,20),random.randint(5,20)), # Darker line
                     random.randint(1,2))
        output = cv2.addWeighted(output, 0.9, overlay_static, 0.3 + 0.2*intensity ,0) # Blend static overlay
    return output

def apply_data_bleed_sim(frame, intensity):
    output = frame.copy()
    num_bleeds = random.randint(5, int(10 + 20 * intensity)) # Control number of bleeds

    for _ in range(num_bleeds):
        # Define source block
        bh_src = random.randint(int(HEIGHT*0.05), int(HEIGHT*0.3)) # Smaller source blocks
        bw_src = random.randint(int(WIDTH*0.05), int(WIDTH*0.3))
        if HEIGHT - bh_src <=0 or WIDTH - bw_src <=0 : continue
        y_src = random.randint(0, HEIGHT - bh_src -1)
        x_src = random.randint(0, WIDTH - bw_src -1)
        source_block = frame[y_src:y_src+bh_src, x_src:x_src+bw_src]
        if source_block.size == 0: continue
        
        avg_color = np.mean(source_block, axis=(0,1)).astype(np.uint8) # Average color of source
        
        # Define target area for bleed
        bleed_strength = 0.1 + 0.5 * intensity * random.random() # Control bleed opacity
        
        # Bleed can be larger than source, and offset
        bleed_radius_x_factor = random.uniform(0.5, 2.5)
        bleed_radius_y_factor = random.uniform(0.5, 2.5)
        bleed_h = int(bh_src * bleed_radius_y_factor)
        bleed_w = int(bw_src * bleed_radius_x_factor)

        # Target position, can be offset from source
        offset_x = random.randint(-bw_src // 2, bw_src // 2)
        offset_y = random.randint(-bh_src // 2, bh_src // 2)
        
        tx = np.clip(x_src + offset_x, 0, WIDTH - bleed_w)
        ty = np.clip(y_src + offset_y, 0, HEIGHT - bleed_h)
        if bleed_w <=0 or bleed_h <=0: continue

        target_area = output[ty:ty+bleed_h, tx:tx+bleed_w]
        if target_area.size == 0: continue
        
        # Create a solid color overlay matching the target area size
        color_overlay = np.full_like(target_area, avg_color)
        
        output[ty:ty+bleed_h, tx:tx+bleed_w] = cv2.addWeighted(target_area, 1.0 - bleed_strength, color_overlay, bleed_strength, 0)
    return output

def apply_flickering_glitch_layer(frame, intensity, effect_func, prev_frame=None, **kwargs_from_main_loop):
    if random.random() > (0.3 + 0.6 * intensity): # Chance to skip flicker
        return frame

    glitch_layer_input = frame.copy() # Apply effect on a copy

    # --- Parameter Preparation for Inner Effect ---
    # This section attempts to prepare parameters for the inner 'effect_func'.
    # It's complex because the naming convention in global_params (e.g., 'perlin_alpha')
    # differs from how the main loop derives keys (e.g., 'alpha' from 'alpha_range').
    # A full fix requires aligning these conventions.
    # For now, this passes kwargs_from_main_loop, and the inner effect might use defaults
    # if specific scaled parameters aren't found due to name mismatches.
    
    params_to_pass_inner = {}
    sig_inner = inspect.signature(effect_func)

    if 'prev_frame' in sig_inner.parameters:
        params_to_pass_inner['prev_frame'] = prev_frame if prev_frame is not None else frame
    if 'intensity' in sig_inner.parameters: # Pass intensity if the inner function accepts it
         params_to_pass_inner['intensity'] = intensity

    # Attempt to pass through relevant scaled ranges from kwargs_from_main_loop (current_visual_params)
    # This is heuristic due to naming convention differences.
    for p_name_sig_inner in sig_inner.parameters:
        if p_name_sig_inner.endswith("_range"):
            # Try to find a matching key in kwargs_from_main_loop
            # Example: p_name_sig_inner = "alpha_range"
            # We need to find if "perlin_alpha", "feedback_alpha" etc. is in kwargs_from_main_loop
            # and corresponds to this effect_func and this p_name_sig_inner.
            # This is non-trivial without a clear mapping.
            # The original code in the main loop used `key_base = p_name_sig_inner[:-6]`.
            # We can try that here with kwargs_from_main_loop.
            key_base_candidate = p_name_sig_inner[:-6]
            if key_base_candidate in kwargs_from_main_loop and isinstance(kwargs_from_main_loop[key_base_candidate], tuple):
                 # This assumes generic keys like "alpha" exist in kwargs_from_main_loop
                 # and are meant for this p_name_sig_inner.
                 params_to_pass_inner[p_name_sig_inner] = kwargs_from_main_loop[key_base_candidate]
            # else: if not found, inner_effect_func will use its default for this range.

    glitched_part = None
    try:
        # Call the inner effect function
        if params_to_pass_inner: # If we prepared any specific params
             glitched_part = effect_func(glitch_layer_input, **params_to_pass_inner)
        else: # Fallback: call with minimal, relying on defaults or what's directly in kwargs
             glitched_part = effect_func(glitch_layer_input, prev_frame=params_to_pass_inner.get('prev_frame'), intensity=params_to_pass_inner.get('intensity'))

    except Exception as e: # Fallback if specific ranges are missing or other errors
        # print(f"TypeError/Error in flickering_glitch_layer calling {effect_func.__name__}: {e}. Using defaults/simpler call.")
        try:
            # Try a simpler call, relying more on defaults within effect_func
            minimal_params = {}
            if 'prev_frame' in sig_inner.parameters: minimal_params['prev_frame'] = prev_frame if prev_frame is not None else frame
            if 'intensity' in sig_inner.parameters: minimal_params['intensity'] = intensity
            glitched_part = effect_func(glitch_layer_input, **minimal_params)
        except Exception as e_fallback:
            # print(f"Fallback call also failed for {effect_func.__name__}: {e_fallback}")
            glitched_part = glitch_layer_input # No glitch if effect fails badly

    if glitched_part is None: glitched_part = glitch_layer_input


    # Create a random mask (e.g., horizontal stripes, blocks, or random noise)
    mask_type = random.choice(["stripes", "blocks", "noise"])
    flicker_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    if mask_type == "stripes":
        num_stripes = random.randint(5, 30)
        stripe_h = HEIGHT // num_stripes
        for i in range(num_stripes):
            if random.random() < 0.5: # Chance for a stripe to be part of the mask
                flicker_mask[i*stripe_h : (i+1)*stripe_h, :] = 255
    elif mask_type == "blocks":
        num_blocks_w = random.randint(5, 20)
        num_blocks_h = random.randint(5, 20)
        block_w = WIDTH // num_blocks_w
        block_h = HEIGHT // num_blocks_h
        for r_idx in range(num_blocks_h):
            for c_idx in range(num_blocks_w):
                if random.random() < 0.3: # Chance for a block
                    flicker_mask[r_idx*block_h:(r_idx+1)*block_h, c_idx*block_w:(c_idx+1)*block_w] = 255
    else: # noise mask
        flicker_mask = (np.random.randint(0, 2, size=(HEIGHT, WIDTH), dtype=np.uint8) * 255)


    mask_inv = cv2.bitwise_not(flicker_mask)
    
    fg = cv2.bitwise_and(glitched_part, glitched_part, mask=flicker_mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    return cv2.add(fg, bg)


def apply_figurative_noise_sim(frame, intensity): # Wrapper for perlin with specific settings
    # Softer, larger scale perlin noise
    return apply_perlin_noise(frame, 
                              alpha_range=(0.1 * intensity, 0.3 * intensity), 
                              scale_range=(80, 200), # Larger scale for "figurative"
                              oct_range=(3,5)) # Fewer octaves for smoother noise

def apply_crt_ghost_sim(frame, prev_frame): # Simple ghosting
    if prev_frame is not None:
        return cv2.addWeighted(frame, 0.85, prev_frame, 0.15, 0) # Current frame stronger
    return frame

def apply_slit_scan_sim(frame): # Basic horizontal slit-scan simulation
    rows, cols, _ = frame.shape
    output = frame.copy()
    slit_pos_y = rows // 2 # Example: scan line at center
    
    # This is a simplified version. True slit-scan accumulates slices over time.
    # This version creates a spatial distortion based on a "virtual" scan.
    for r_idx in range(rows):
         # Shift amount varies with distance from "slit" and some noise
         dist_from_slit = abs(r_idx - slit_pos_y)
         # Max shift decreases further from slit, scaled by a sine wave for organic feel
         max_smear = (cols / 10) * (1 - dist_from_slit / (rows/2)) * abs(math.sin(r_idx * 0.05 + random.uniform(0,math.pi)))
         max_smear = max(0, max_smear) # Ensure non-negative
         smear_amount = int(random.uniform(-max_smear, max_smear) * 0.5) # Smaller random smear
         
         output[r_idx, :] = np.roll(output[r_idx, :], smear_amount, axis=0) # Roll pixels in the row
    return output

def apply_vector_field_sim(frame): # Wrapper for warp with specific settings
    # Subtle, flowing warp
    return apply_warp(frame, 
                      amplitude_range=(2, 8 + 5*random.random()), # Smaller amplitude
                      freq_range=(0.08, 0.3 + 0.1*random.random())) # Higher frequency for finer details


# --- Audio Effects ---
def apply_distortion(data, intensity_range=(1.5, 5.0)):
    intensity = random.uniform(*intensity_range)
    # Simple clipping distortion
    return np.clip(data * intensity, -0.98, 0.98) # Clip to avoid extreme values

def apply_bitcrush(data, bit_depth_range=(4, 12)):
    bd_min = max(2, int(bit_depth_range[0]))
    bd_max = max(bd_min + 1, int(bit_depth_range[1]))
    if bd_min >= bd_max: bits = bd_min
    else: bits = random.randint(bd_min, bd_max)
    
    if bits >= 16: return data # No bitcrushing if depth is high
    
    # Quantize signal to fewer levels
    steps = 2**(bits -1) # Number of positive steps
    return np.round(data * steps) / steps

def apply_spectral_glitch(data, intensity=0.5): # Intensity 0-1
    n_samples = len(data)
    if n_samples < 1024: return data # Need enough samples for FFT

    data_float = data.astype(np.float32)
    spectrum = fft(data_float)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    n_freqs = len(spectrum)

    # Number of frequency bins to glitch
    num_glitches = int(n_freqs * 0.01 * intensity * random.uniform(0.5, 1.5)) # Fewer, more impactful glitches
    
    # Glitch some frequency magnitudes or phases
    # Operate on positive frequencies (first half of spectrum)
    positive_freq_bins = n_freqs // 2
    if positive_freq_bins <=1 : return data # Not enough freq bins

    indices_to_glitch = random.sample(range(1, positive_freq_bins), min(num_glitches, positive_freq_bins -1 ))

    for idx in indices_to_glitch:
        if random.random() < 0.6: # Attenuate magnitude
             attenuation_factor = random.uniform(0.0, 0.3) # Strong attenuation
             magnitude[idx] *= attenuation_factor
             if idx < n_freqs: magnitude[n_freqs - idx] *= attenuation_factor # Symmetric part
        else: # Swap phases of adjacent bins for smearing
             if idx + 1 < positive_freq_bins:
                  phase[idx], phase[idx+1] = phase[idx+1], phase[idx]
                  # Update symmetric part carefully
                  if n_freqs - idx < n_freqs: phase[n_freqs - idx] = -phase[idx] 
                  if n_freqs - (idx+1) < n_freqs : phase[n_freqs - (idx+1)] = -phase[idx+1]
    
    new_spectrum = magnitude * np.exp(1j * phase)
    glitched_data = np.real(ifft(new_spectrum))
    
    # Normalize to prevent volume spikes/dips
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
    if n_ir == 0 : return data
    
    # Truncate IR if it's too long relative to data, or convolve will be mostly tail
    if n_ir >= n_data: 
        impulse_response = impulse_response[:max(1,n_data//2)]
        n_ir = len(impulse_response)
    if n_ir == 0 : return data # After truncation, IR might be empty
        
    reverbed_data = np.convolve(data, impulse_response, mode='same') # 'same' keeps output length same as input
    
    # Normalize to prevent volume changes
    max_abs_orig = np.max(np.abs(data))
    max_abs_rev = np.max(np.abs(reverbed_data))
    if max_abs_rev > 1e-6 and max_abs_orig > 1e-6:
        reverbed_data *= (max_abs_orig / max_abs_rev)
    elif max_abs_rev > 1e-6: # If original was silent
        reverbed_data /= max_abs_rev
        
    return reverbed_data

IMPULSE_RESPONSES = {
    "damp": np.exp(-np.linspace(0, 10, int(SAMPLE_RATE * 0.2))) * np.random.randn(int(SAMPLE_RATE * 0.2)),
    "metal_hit": np.sin(2*np.pi*1500*np.linspace(0,0.1,int(SAMPLE_RATE*0.1))) * np.exp(-np.linspace(0, 20, int(SAMPLE_RATE * 0.1))),
    "noise_burst": np.random.randn(int(SAMPLE_RATE * 0.1)) * np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * 0.1))) 
}
# Normalize impulse responses
for k_ir in IMPULSE_RESPONSES:
    max_abs_ir = np.max(np.abs(IMPULSE_RESPONSES[k_ir]))
    if max_abs_ir > 1e-6: IMPULSE_RESPONSES[k_ir] /= max_abs_ir


def apply_broken_speaker_sim(data, intensity): # Intensity 0-1
    output = data.copy()
    
    # Simulate frequency response issues (e.g., parts of spectrum cut or scooped)
    if random.random() < 0.7 * intensity:
        spectrum = fft(output)
        n = len(spectrum)
        positive_freq_bins = n // 2
        if positive_freq_bins > 10: # Ensure enough bins
            if random.random() < 0.5: # Low-pass or high-pass like effect
                cut_freq_bin = int((random.uniform(500, 3000) / (SAMPLE_RATE / 2)) * positive_freq_bins)
                if random.random() < 0.5: # Low-pass
                    spectrum[cut_freq_bin:positive_freq_bins] = 0
                    spectrum[n-positive_freq_bins+1 : n-cut_freq_bin+1] = 0
                else: # High-pass
                    spectrum[:cut_freq_bin] = 0
                    spectrum[n-cut_freq_bin+1:] = 0 # Correct symmetric part for high-pass
            else: # Mid-scoop
                scoop_center_hz = random.uniform(1000, 6000)
                scoop_width_hz = scoop_center_hz * random.uniform(0.1, 0.5)
                scoop_start_hz = max(1, scoop_center_hz - scoop_width_hz / 2)
                scoop_end_hz = scoop_center_hz + scoop_width_hz / 2

                scoop_start_bin = int((scoop_start_hz / (SAMPLE_RATE/2)) * positive_freq_bins)
                scoop_end_bin = int((scoop_end_hz / (SAMPLE_RATE/2)) * positive_freq_bins)
                scoop_end_bin = min(scoop_end_bin, positive_freq_bins) # Ensure within bounds

                if scoop_start_bin < scoop_end_bin :
                    spectrum[scoop_start_bin:scoop_end_bin] = 0
                    spectrum[n-scoop_end_bin+1 : n-scoop_start_bin+1] = 0 # Symmetric part
            output = np.real(ifft(spectrum))

    # Heavy distortion
    output = apply_distortion(output, intensity_range=(5.0 + 10 * intensity, 15.0 + 20 * intensity))
    
    # Crackles
    if random.random() < 0.5 * intensity:
        num_crackles = random.randint(10, int(50 + 100 * intensity)) # More crackles with intensity
        for _ in range(num_crackles):
            if len(output) == 0: break
            pos = random.randint(0, len(output)-1)
            # Ensure crackle_len is valid and doesn't exceed bounds
            max_crackle_len = min(50, len(output)-pos)
            if max_crackle_len <=0 : continue
            crackle_len = random.randint(1, max_crackle_len)
            
            output[pos:pos+crackle_len] += (np.random.rand(crackle_len) - 0.5) * 0.5 * intensity # Add noise
            
    return np.clip(output, -1.0, 1.0) # Final clip

# --- Tone, Noise, Melody, Rhythm Generation ---
def generate_tone(freq, duration_samples, vol, fm_chance=0.5, harmonic_chance=0.4):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
    wave = np.zeros(duration_samples, dtype=np.float32)

    if random.random() < fm_chance: # Frequency Modulation
        mod_freq = freq * random.uniform(0.1, 5.0)
        mod_depth_factor = random.uniform(1.0, 8.0) # Modulation index relative to carrier amplitude (implicitly via vol)
        # Actual modulation depth in Hz: mod_depth_factor * mod_freq (approx, depends on definition)
        # For phase modulation: sin(2*pi*fc*t + I*sin(2*pi*fm*t)), I is modulation index
        # Here, it's more like I = mod_depth_factor * vol (if vol is amplitude)
        # Let's use a common FM synthesis approach:
        mod_index = mod_depth_factor * 5.0 # A more direct modulation index
        if abs(freq * t[-1]) < 1e9 and abs(mod_freq * t[-1]) < 1e9: # Avoid huge numbers for np.sin arguments
            wave = np.sin(2 * np.pi * freq * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
        else: # Fallback if arguments too large (less likely with typical audio durations)
            wave = np.sin(2 * np.pi * freq * t)
    elif random.random() < harmonic_chance: # Additive Synthesis (simple)
        harmonic_count = random.randint(1, 4)
        wave = np.sin(2 * np.pi * freq * t) # Fundamental
        for h_idx in range(2, harmonic_count + 2): # Harmonics 2nd, 3rd, ...
            harmonic_vol_factor = random.uniform(0.1, 0.5) / h_idx # Higher harmonics quieter
            phase_shift = random.uniform(0, np.pi)
            wave += harmonic_vol_factor * np.sin(2 * np.pi * freq * h_idx * t + phase_shift)
        max_abs_wave = np.max(np.abs(wave))
        if max_abs_wave > 1e-6: wave /= max_abs_wave # Normalize additive part before applying main volume
    else: # Simple Sine Wave
        wave = np.sin(2 * np.pi * freq * t)
    
    wave *= vol # Apply overall volume

    # ADSR Envelope (simplified)
    attack_len = min(int(SAMPLE_RATE * 0.005), duration_samples // 3)
    decay_len = min(int(SAMPLE_RATE * random.uniform(0.05, 0.3)), max(0, duration_samples - attack_len))
    sustain_level = random.uniform(0.1, 0.7)
    # Release is tricky if sound ends abruptly; for now, decay to sustain or end.
    # release_len = min(int(SAMPLE_RATE * random.uniform(0.05, 0.4)), max(0, duration_samples - attack_len - decay_len))
    
    envelope = np.ones(duration_samples)
    if attack_len > 0: 
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
    
    sustain_start_idx = attack_len + decay_len
    if decay_len > 0 and sustain_start_idx <= duration_samples:
        envelope[attack_len:sustain_start_idx] = np.linspace(1, sustain_level, decay_len)
        if sustain_start_idx < duration_samples:
            envelope[sustain_start_idx:] = sustain_level # Hold sustain level
    elif attack_len < duration_samples : # No decay, just attack to sustain
         envelope[attack_len:] = sustain_level


    # Apply a simple fade out at the very end to avoid clicks if sustain is high
    fade_out_len = min(int(SAMPLE_RATE * 0.01), duration_samples // 4)
    if fade_out_len > 0 and duration_samples > fade_out_len:
        current_level_at_fade_start = envelope[duration_samples - fade_out_len -1] if duration_samples - fade_out_len -1 >=0 else sustain_level
        envelope[duration_samples - fade_out_len:] = np.linspace(current_level_at_fade_start, 0, fade_out_len)

    return wave * envelope


def generate_noise(noise_type, duration_samples, vol, features=[]):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    
    # NOTE: Many noise types from THEMES are not implemented here yet.
    # This function needs to be fully fleshed out.
    # Adding stubs for a few more common ones.

    out_noise = np.zeros(duration_samples, dtype=np.float32)

    if noise_type == "white":
        out_noise = (np.random.rand(duration_samples) - 0.5) * 2.0
    elif noise_type == "pink": # Approximation using Voss-McCartney algorithm (simple version)
        # For a better pink noise, use a proper filter or more sophisticated generation.
        # This is a very rough approximation.
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786] # Example filter coeffs
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        white = (np.random.rand(duration_samples + len(b)) - 0.5) * 2.0 # Add padding for filter
        try:
            from scipy.signal import lfilter
            pink_filtered = lfilter(b, a, white)
            out_noise = pink_filtered[len(b):] # Remove filter transient
        except ImportError:
            out_noise = (np.random.rand(duration_samples) - 0.5) * 1.8 # Fallback to white-ish
    elif noise_type == "brown": # Brownian or Red noise (random walk)
        white = (np.random.rand(duration_samples) - 0.5) * 0.2 # Smaller steps
        out_noise = np.cumsum(white)
        max_abs_brown = np.max(np.abs(out_noise))
        if max_abs_brown > 1e-6: out_noise /= max_abs_brown # Normalize
    elif noise_type == "glitch": # Short, sharp clicks or noise bursts
        num_glitches = random.randint(5, int(duration_samples / (SAMPLE_RATE * 0.01))) # Up to 100 glitches/sec
        for _ in range(num_glitches):
            glitch_len = random.randint(1, max(1, int(SAMPLE_RATE * 0.005))) # 1 to 5ms glitches
            if duration_samples < glitch_len + 10: continue
            start = random.randint(0, duration_samples - glitch_len -1)
            out_noise[start:start+glitch_len] = (np.random.rand(glitch_len)-0.5) * random.uniform(0.5, 2.0) # Vary glitch amplitude
    elif noise_type == "data_stream_sim":
        # Rapidly changing filtered noise and clicks
        base_noise = generate_noise("pink", duration_samples, 0.5, []) # Quieter base
        clicks = generate_noise("glitch", duration_samples, 0.5, [])
        # Modulate click density/presence with a slow LFO
        mod_freq_lfo = random.uniform(0.5, 5.0) # Slow modulation
        mod_lfo = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq_lfo * np.linspace(0, duration_samples*time_step, duration_samples))
        out_noise = base_noise + clicks * mod_lfo
    elif noise_type == "white_bursts":
        # --- CORRECTED SECTION FOR white_bursts ---
        min_bursts_val = 1 # Allow at least 1 burst if duration is very short
        # Calculate max possible bursts based on a burst every 0.05s (20 bursts/sec)
        max_possible_bursts = int(duration_samples / (SAMPLE_RATE * 0.05)) 
        
        if max_possible_bursts <= 0:
            num_bursts = 0
        elif max_possible_bursts < min_bursts_val: # If max possible is less than desired min
            num_bursts = random.randint(1, max_possible_bursts) # Make as many as possible
        else: # If enough duration for desired min (e.g. 3) or more
            # Original code had random.randint(3, max_possible_bursts).
            # If we want to keep the "at least 3" idea IF POSSIBLE:
            intended_min_bursts = 3
            actual_min = min(intended_min_bursts, max_possible_bursts)
            num_bursts = random.randint(actual_min, max_possible_bursts)

        for _ in range(num_bursts):
            burst_len = random.randint(int(SAMPLE_RATE*0.002), int(SAMPLE_RATE*0.05)) # 2ms to 50ms
            if duration_samples < burst_len + 10: continue # Safety for very short overall duration_samples
            start = random.randint(0, duration_samples - burst_len -1)
            out_noise[start:start+burst_len] = (np.random.rand(burst_len)-0.5) * 2.0 # Full volume white noise
    elif noise_type == "static": # Similar to white noise, maybe with some filtering
        out_noise = (np.random.rand(duration_samples) - 0.5) * 1.9
        # Could add slight filtering here for "static" feel vs pure white
    else: # Fallback for unimplemented types
        # print(f"Warning: Noise type '{noise_type}' not fully implemented. Using white noise fallback.")
        out_noise = (np.random.rand(duration_samples) - 0.5) * 2.0 
    
    final_noise = out_noise * vol
    # Normalize final output to prevent clipping if multiple noises combined or vol is high
    max_abs_final = np.max(np.abs(final_noise))
    if max_abs_final > 1.0 : final_noise /= max_abs_final
    elif max_abs_final < 1e-6 and vol > 1e-3: # If it was meant to be audible but became silent
        pass # Keep it silent if calculations led to it

    return final_noise


def generate_biometric_rhythm(duration_samples, vol, rhythm_type="heartbeat_distorted"):
    noise_out = np.zeros(duration_samples, dtype=np.float32)
    if duration_samples <=0: return noise_out
    
    if rhythm_type == "heartbeat_distorted" or "heartbeat" in rhythm_type:
        bpm = random.uniform(30, 120) # Beats per minute
        # Add some variation to BPM over time for realism
        bpm_variation_factor = random.uniform(0.7, 1.3) 
        bpm *= bpm_variation_factor
        bpm = np.clip(bpm, 20, 180) # Keep BPM in a reasonable range

        bps = bpm / 60.0 # Beats per second
        if bps <= 1e-3: return noise_out # Avoid division by zero or extremely slow beats
        
        beat_interval_samples_base = int(SAMPLE_RATE / bps)
        if beat_interval_samples_base <= 0: return noise_out

        # Create a single heartbeat sound (thump-thump)
        thud1_len = int(SAMPLE_RATE * random.uniform(0.05, 0.10))
        thud1_freq = random.uniform(30, 80)
        t_thud1 = np.linspace(0, thud1_len * time_step, thud1_len, endpoint=False)
        thud1_env = np.exp(-np.linspace(0, random.uniform(8,18), thud1_len)) # Faster decay
        thud1_sound = np.sin(2*np.pi*thud1_freq*t_thud1) * thud1_env
        
        thud2_len = int(SAMPLE_RATE * random.uniform(0.03, 0.08)) # Shorter second thump
        thud2_freq = random.uniform(40, 90)
        t_thud2 = np.linspace(0, thud2_len * time_step, thud2_len, endpoint=False)
        thud2_env = np.exp(-np.linspace(0, random.uniform(10,20), thud2_len))
        thud2_sound = np.sin(2*np.pi*thud2_freq*t_thud2) * thud2_env * 0.7 # Second thump quieter

        single_beat_sound = np.zeros(thud1_len + thud2_len + int(SAMPLE_RATE*0.05)) # Add small gap
        single_beat_sound[:thud1_len] = thud1_sound
        single_beat_sound[thud1_len + int(SAMPLE_RATE*0.02) : thud1_len + int(SAMPLE_RATE*0.02) + thud2_len] = thud2_sound
        
        single_beat_sound = apply_distortion(single_beat_sound, (1.5, 4.0)) # Distort the beat

        current_sample = 0
        while current_sample < duration_samples:
            # Add slight timing variation to beat interval
            beat_interval_samples = int(beat_interval_samples_base * random.uniform(0.9, 1.1))
            if beat_interval_samples <=0 : beat_interval_samples = beat_interval_samples_base

            actual_beat_len = min(len(single_beat_sound), duration_samples - current_sample)
            if actual_beat_len <=0 : break
            
            noise_out[current_sample : current_sample + actual_beat_len] += single_beat_sound[:actual_beat_len]
            current_sample += beat_interval_samples
    elif "breathing" in rhythm_type:
        # Simple filtered noise for breathing
        breath_cycle_sec = random.uniform(3, 7) # Duration of one inhale-exhale cycle
        breath_cycle_samples = int(breath_cycle_sec * SAMPLE_RATE)
        
        current_sample = 0
        while current_sample < duration_samples:
            actual_cycle_len = min(breath_cycle_samples, duration_samples - current_sample)
            if actual_cycle_len <=0 : break
            
            # Generate noise for one cycle
            raw_breath_noise = generate_noise("pink", actual_cycle_len, 1.0, []) # Pink noise base
            
            # Envelope for inhale/exhale
            t_cycle = np.linspace(0, 1, actual_cycle_len)
            inhale_env = np.sin(np.pi * t_cycle[:actual_cycle_len//2])**2 # Smoother inhale
            exhale_env = np.cos(np.pi * t_cycle[actual_cycle_len//2:])**2 # Smoother exhale
            breath_env = np.concatenate((inhale_env, exhale_env))
            if len(breath_env) < actual_cycle_len: # Pad if needed due to //2
                breath_env = np.pad(breath_env, (0, actual_cycle_len - len(breath_env)), 'edge')
            
            noise_out[current_sample : current_sample + actual_cycle_len] += raw_breath_noise * breath_env
            current_sample += actual_cycle_len
            
    final_rhythm = noise_out * vol
    max_abs_rhythm = np.max(np.abs(final_rhythm))
    if max_abs_rhythm > 1e-6 : final_rhythm /= max_abs_rhythm # Normalize before returning
    return final_rhythm


def generate_melody_fragment(duration_samples, vol, theme_data, intensity):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    
    scale_name = theme_data.get("melody_scale", "minor_pentatonic")
    scale_intervals = SCALES.get(scale_name, SCALES["minor_pentatonic"])
    base_freq = random.uniform(60, 660) # Base frequency for the fragment
    
    # Number of notes and their duration
    notes_in_fragment = random.randint(max(1, int(2 - intensity)), int(6 + 10 * intensity)) # Fewer notes if low intensity
    notes_in_fragment = max(1, notes_in_fragment) # At least one note

    avg_note_duration_samples = duration_samples // notes_in_fragment
    if avg_note_duration_samples <=0 : avg_note_duration_samples = max(1, int(SAMPLE_RATE * 0.05)) # Min duration if too short

    melody = np.zeros(duration_samples, dtype=np.float32)
    current_pos = 0

    for _ in range(notes_in_fragment):
        if current_pos >= duration_samples: break
        
        # Note properties
        scale_degree = random.choice(scale_intervals)
        octave_shift = random.choice([-1, 0, 0, 1, 1, 2]) # More chances for 0 and 1 octave
        note_freq_start = base_freq * (2**((scale_degree + octave_shift * 12) / 12.0))
        
        # Pitch bend
        pitch_bend_semitones = random.uniform(-6, 6) * intensity # Bend more with higher intensity
        note_freq_end = note_freq_start * (2**(pitch_bend_semitones / 12.0))
        
        note_freq_start = np.clip(note_freq_start, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)
        note_freq_end = np.clip(note_freq_end, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)
        
        # Vary note duration slightly
        actual_note_len = int(avg_note_duration_samples * random.uniform(0.7, 1.3))
        actual_note_len = min(actual_note_len, duration_samples - current_pos)
        actual_note_len = max(1, actual_note_len) # Ensure positive length
        if actual_note_len <= 0: break

        # Generate tone with pitch bend
        t_note = np.linspace(0, actual_note_len * time_step, actual_note_len, endpoint=False)
        current_freqs_for_note = np.linspace(note_freq_start, note_freq_end, actual_note_len)
        
        # Calculate phase for continuous frequency change (FM-like)
        instantaneous_phase = np.cumsum(2 * np.pi * current_freqs_for_note * time_step)
        note_tone_bent = np.sin(instantaneous_phase)
        
        # Envelope for this note
        attack_len_note = min(int(actual_note_len*0.1), actual_note_len) 
        decay_len_note = min(int(actual_note_len*0.4), actual_note_len - attack_len_note)
        sustain_level_note = random.uniform(0.2,0.8)
        envelope_note = np.ones(actual_note_len)
        if attack_len_note > 0: envelope_note[:attack_len_note] = np.linspace(0,1,attack_len_note)
        if decay_len_note > 0 and attack_len_note + decay_len_note <= actual_note_len :
            envelope_note[attack_len_note:attack_len_note+decay_len_note] = np.linspace(1,sustain_level_note,decay_len_note)
            if attack_len_note+decay_len_note < actual_note_len:
                envelope_note[attack_len_note+decay_len_note:] = sustain_level_note
        elif attack_len_note < actual_note_len: 
            envelope_note[attack_len_note:] = sustain_level_note
        
        # Apply fade at end of note
        fade_out_note = min(int(actual_note_len * 0.1), actual_note_len // 2)
        if fade_out_note > 0:
            start_level = envelope_note[actual_note_len - fade_out_note -1] if actual_note_len - fade_out_note -1 >=0 else sustain_level_note
            envelope_note[actual_note_len - fade_out_note:] = np.linspace(start_level, 0, fade_out_note)


        melody[current_pos : current_pos + actual_note_len] += note_tone_bent * envelope_note
        current_pos += actual_note_len

    # Counter Melody (simpler, more sparse)
    if random.random() < theme_data.get("melody_counter_melody_chance", 0.3) * intensity:
        counter_melody = np.zeros(duration_samples, dtype=np.float32)
        current_pos_counter = int(random.uniform(0, avg_note_duration_samples * 0.5)) # Start slightly offset

        for _ in range(max(1,notes_in_fragment // 2)): # Fewer notes for counter melody
            if current_pos_counter >= duration_samples: break
            
            dissonant_intervals = [-1, 1, -5, 5, -7, 7] # More dissonant choices
            scale_degree_main = random.choice(scale_intervals) # Base on main scale
            counter_degree_offset = random.choice(dissonant_intervals)
            counter_degree = (scale_degree_main + counter_degree_offset) % 12 # Ensure within octave
            
            octave_shift_cnt = random.choice([-1,0,1])
            note_freq_cnt = base_freq * 0.7 * (2**((counter_degree + octave_shift_cnt*12)/12.0)) # Slightly different base
            note_freq_cnt = np.clip(note_freq_cnt,LOW_FREQ_CUTOFF,HIGH_FREQ_CUTOFF)
            
            actual_note_len_cnt = int(avg_note_duration_samples * random.uniform(1.0, 2.0)) # Can be longer
            actual_note_len_cnt = min(actual_note_len_cnt, duration_samples - current_pos_counter)
            actual_note_len_cnt = max(1, actual_note_len_cnt)
            if actual_note_len_cnt <=0: break
            
            # Simpler tone for counter melody
            note_tone_cnt = generate_tone(note_freq_cnt, actual_note_len_cnt, 0.7, fm_chance=0.2, harmonic_chance=0.1) 
            counter_melody[current_pos_counter : current_pos_counter + actual_note_len_cnt] += note_tone_cnt
            current_pos_counter += actual_note_len_cnt + int(avg_note_duration_samples * random.uniform(0.5,1.5)) # Longer gaps

        melody = cv2.addWeighted(melody, 1.0, counter_melody, 0.6, 0) # Blend counter melody


    # Apply effects to the whole fragment
    distortion_amount = theme_data.get("melody_distortion", 10.0) * (0.5 + intensity * 0.5) # Scale distortion
    melody = apply_distortion(melody, intensity_range=(max(1.1, distortion_amount*0.7), max(1.5, distortion_amount*1.5)))
    
    if random.random() < 0.6*intensity: 
        melody = apply_bitcrush(melody, bit_depth_range=(max(2,int(8-intensity*2)), max(3,int(10-intensity*2)))) # More crushing with intensity
    
    if random.random() < 0.3*intensity: 
        melody = apply_spectral_glitch(melody, intensity=intensity*0.8)
    
    # Granular/Shredding effect (simplified)
    if random.random() < 0.2*intensity and len(melody) > int(SAMPLE_RATE * 0.05): # Only if melody is somewhat long
        grain_len_base = int(SAMPLE_RATE * random.uniform(0.005, 0.02))
        num_grains_to_rearrange = max(0, int(len(melody) / (grain_len_base * 2)) // (2 + int(3*(1-intensity)))) # More grains if low intensity
        
        shredded_melody = melody.copy() # Work on a copy
        for _ in range(num_grains_to_rearrange):
            grain_len = max(1, int(grain_len_base * random.uniform(0.5,1.5)))
            if len(melody) < grain_len : break
            
            start_idx_src = random.randint(0, len(melody) - grain_len)
            grain = melody[start_idx_src : start_idx_src + grain_len].copy() # Copy the grain
            
            # Place grain somewhere else, possibly overlapping
            start_idx_dst = random.randint(0, len(shredded_melody) - grain_len)
            # Blend or replace
            if random.random() < 0.5:
                shredded_melody[start_idx_dst : start_idx_dst + grain_len] += grain * random.uniform(0.5,1.2)
            else:
                shredded_melody[start_idx_dst : start_idx_dst + grain_len] = grain * random.uniform(0.5,1.2)
        melody = shredded_melody
        # Normalize shredded melody
        max_abs_shred = np.max(np.abs(melody))
        if max_abs_shred > 1e-6: melody /= max_abs_shred


    final_melody = melody * vol
    max_abs_final_melody = np.max(np.abs(final_melody))
    if max_abs_final_melody > 1.0: final_melody /= max_abs_final_melody # Ensure no clipping from vol
    
    return final_melody


# --- Frame Generation ---
def generate_frames_enhanced(theme_data, intensity_func, global_params):
    DURATION = random.randint(MIN_DURATION, MAX_DURATION)
    FPS = random.randint(MIN_FPS, MAX_FPS)
    FRAME_COUNT = DURATION * FPS
    
    # Make a mutable copy of visual params for this generation run
    current_visual_params = copy.deepcopy(global_params["visual"])
    instability_chance = global_params["instability_chance"] * 1.5 # Slightly higher for visuals
    
    print(f"Generating {DURATION}s video ({FPS} FPS), Theme: {global_params['theme_name']}...")
    # print(f"Visual Instability Chance: {instability_chance:.4f}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Use "avc1" for H.264 if preferred and available
    video_temp_path = f"video_temp_{int(time.time())}_{random.randint(100,999)}.mp4"
    video = cv2.VideoWriter(video_temp_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    prev_frame = None
    frame_hold_counter = 0
    held_frame = None
    effect_history = [] # To avoid too much repetition of the same effect
    
    current_theme_mutated = copy.deepcopy(theme_data) # For dynamic mutations within this generation
    mutation_timer = 0
    mutation_interval = FPS * random.randint(5,15) # How often to try mutating theme colors

    # ARG elements state
    arg_qr_img = None
    arg_qr_display_frames = 0
    arg_text_code = None
    arg_text_display_frames = 0
    
    # Persistent texts setup
    persistent_texts = []
    num_persistent = random.randint(8, 20)
    for _ in range(num_persistent):
        word = str(random.choice(current_theme_mutated["words"]))
        symbol = str(random.choice(current_theme_mutated["symbols"]))
        
        if random.random() < current_theme_mutated.get("zalgo_words_chance", 0.3):
            word = zalgo_text(word)
        
        # Chance to encode word
        if random.random() < 0.1: word = base64.b64encode(word.encode('utf-8', 'ignore')).decode()
        elif random.random() < 0.05: word = binascii.hexlify(word.encode('utf-8', 'ignore')).decode()
        
        text_content = word + symbol
        # Ensure color tuple is (B,G,R), not (B,G,R,A) for OpenCV putText
        chosen_palette = random.choice(current_theme_mutated["colors"])
        text_color_bgr = tuple(c for c in random.choice(chosen_palette)[:3])


        persistent_texts.append({ 
            "text": text_content, 
            "pos": (random.randint(-WIDTH//4, WIDTH), random.randint(-HEIGHT//4, HEIGHT)), # Allow starting off-screen
            "font_size": random.uniform(0.4, 5.0), 
            "color": text_color_bgr,
            "lifetime": random.randint(int(FPS * 0.5), int(FPS * 15)), # Longer lifetimes possible
            "frame_count": 0,
            "move_speed": (random.uniform(-8, 8), random.uniform(-6, 6)), # Slower average movement
            "alpha": random.uniform(0.3, 1.0) # For potential blending (not directly used by cv2.putText)
        })

    for i in range(FRAME_COUNT):
        current_time_sec = i / FPS
        current_intensity = intensity_func(current_time_sec)
        current_intensity = np.clip(current_intensity, 0.01, 4.0) # Ensure intensity is within bounds
        
        mutation_timer += 1
        if mutation_timer >= mutation_interval: # Dynamic Theme Mutation (e.g., colors)
            mutation_timer = 0
            if random.random() < 0.3 * current_intensity: # Higher chance with higher intensity
                try:
                    palette_idx = random.randrange(len(current_theme_mutated["colors"]))
                    color_idx = random.randrange(len(current_theme_mutated["colors"][palette_idx]))
                    # Mutate color slightly
                    original_color = current_theme_mutated["colors"][palette_idx][color_idx]
                    mutated_color = tuple(np.clip(c + random.randint(-40,40), 0, 255) for c in original_color[:3]) # Mutate BGR
                    if len(original_color) == 4: # Preserve alpha if it exists
                        mutated_color = (*mutated_color, original_color[3])
                    current_theme_mutated["colors"][palette_idx][color_idx] = mutated_color
                except IndexError: pass # Should not happen if THEMES is well-defined
                # print(f"Theme Mutation: Color palette {palette_idx} changed at {current_time_sec:.1f}s")

        # Parameter Instability for visual effects
        if random.random() < instability_chance * current_intensity: # More instability with high intensity
            param_key_to_mutate = random.choice(list(current_visual_params.keys()))
            
            if param_key_to_mutate == "effect_probabilities":
                effect_func_name = random.choice(list(current_visual_params["effect_probabilities"].keys()))
                old_prob = current_visual_params["effect_probabilities"][effect_func_name]
                current_visual_params["effect_probabilities"][effect_func_name] = np.clip(old_prob + random.uniform(-0.2, 0.2), 0.01, 0.99)
            elif isinstance(current_visual_params[param_key_to_mutate], tuple) and len(current_visual_params[param_key_to_mutate]) == 2:
                # Mutate min/max range parameters
                c_min, c_max = current_visual_params[param_key_to_mutate]
                range_width = abs(c_max - c_min) + 1e-6 # Avoid division by zero
                
                # Shift midpoint and scale width
                midpoint_shift_factor = random.uniform(-0.15, 0.15) * range_width
                scale_factor = random.uniform(0.75, 1.25)
                
                new_mid = (c_min + c_max) / 2 + midpoint_shift_factor
                new_half_width = (range_width / 2) * scale_factor
                
                n_min = new_mid - new_half_width
                n_max = new_mid + new_half_width
                
                # Ensure min < max and apply type-specific constraints
                if "alpha" in param_key_to_mutate or "intensity" in param_key_to_mutate : # Float ranges usually 0-1
                    n_min = max(0.01, n_min)
                    n_max = min(1.0, max(n_min + 0.05, n_max)) # Ensure max is at least min + delta
                elif any(k in param_key_to_mutate for k in ["shift","block","thresh","size", "oct", "num", "thick"]): # Integer ranges
                    n_min = int(max(1, n_min)) # Most integer params are >= 1
                    n_max = int(max(n_min + 1, n_max)) # Ensure max is at least min + 1
                else: # Generic tuple, try to preserve type if possible or just ensure min < max
                    if n_min >= n_max: n_max = n_min + 0.1 # Simple fix if types are unknown
                
                current_visual_params[param_key_to_mutate] = (n_min, n_max)

        # Frame Hold
        if frame_hold_counter > 0:
            if held_frame is not None:
                video.write(held_frame)
                frame_hold_counter -=1
                continue # Skip rest of frame gen for held frames
            else: # Should not happen if counter > 0
                frame_hold_counter = 0 
        
        # Chance to start a new frame hold
        if random.random() < (0.02 + 0.03 * current_intensity) and frame_hold_counter == 0:
            frame_hold_counter = random.randint(1, int(FPS * (0.3 + current_intensity * 0.7))) # Hold longer with intensity
            held_frame = None # Will be set to current frame at end of this iteration

        # Base Frame Generation
        base_roll = random.random()
        if base_roll < 0.2: # Solid color
             bg_color_tuple = random.choice(random.choice(current_theme_mutated["colors"]))
             frame = np.full((HEIGHT, WIDTH, 3), bg_color_tuple[:3], dtype=np.uint8)
        elif base_roll < 0.4: # Gradient
             c1_tpl = random.choice(random.choice(current_theme_mutated["colors"]))[:3]
             c2_tpl = random.choice(random.choice(current_theme_mutated["colors"]))[:3]
             frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
             for k_row in range(HEIGHT):
                 interp_factor = k_row / HEIGHT
                 frame[k_row, :] = [int(c1_tpl[ch]*(1-interp_factor) + c2_tpl[ch]*interp_factor) for ch in range(3)]
             if random.random() < 0.5: # Chance to rotate gradient
                 frame = cv2.rotate(frame, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]))
        else: # Dark noisy base
            noise_max_val = int(30 + 60*current_intensity) # More intense noise if intensity is high
            noise_max_val = min(255, noise_max_val) # Cap at 255
            frame = np.random.randint(0, max(1, noise_max_val), (HEIGHT,WIDTH,3), dtype=np.uint8)

        # Glitch Catalyst Event (burst of effects)
        is_catalyst_event = False
        # Use a dedicated catalyst_threshold from params, or a default
        catalyst_thresh_val = current_visual_params.get("catalyst_threshold", 2.8) 
        if current_intensity > catalyst_thresh_val and random.random() < (0.15 + 0.15 * current_intensity): # Higher chance if very intense
            is_catalyst_event = True
            num_effects_catalyst = random.randint(4, 7) # More effects during catalyst
            # print(f"GLITCH CATALYST at {current_time_sec:.1f}s! Intensity: {current_intensity:.2f}")

        # Apply Visual Effects
        available_effect_names = current_theme_mutated.get("visual_effects", [])
        # Filter for callable functions defined in globals()
        effect_candidates = [globals()[name] for name in available_effect_names if name in globals() and callable(globals()[name])]
        
        num_effects_base = random.randint(1,2) if not is_catalyst_event else num_effects_catalyst
        num_effects_intensity_driven = int(current_intensity * 1.5) if not is_catalyst_event else 0 # More effects with intensity
        num_effects_to_apply = min(len(effect_candidates), num_effects_base + num_effects_intensity_driven)
        
        applied_effects_count = 0
        random.shuffle(effect_candidates) # Apply in random order
        applied_this_frame_names = []

        for effect_func in effect_candidates:
            if applied_effects_count >= num_effects_to_apply: break
            
            effect_name = effect_func.__name__
            
            # Avoid over-using the same effect recently
            if effect_history.count(effect_name) > 2 and random.random() < 0.7: continue 
            
            # Probability of applying this effect
            base_prob = current_visual_params.get("effect_probabilities", {}).get(effect_name, 0.5) # Default prob if not in map
            prob_modifier_intensity = 0.5 + current_intensity * 0.5 # Scale prob with intensity
            final_prob_to_apply = np.clip(base_prob * prob_modifier_intensity, 0.05, 0.95)
            
            if random.random() < final_prob_to_apply or is_catalyst_event: # Higher chance during catalyst
                try:
                    sig = inspect.signature(effect_func)
                    params_to_pass_to_effect = {}
                    
                    if 'prev_frame' in sig.parameters: params_to_pass_to_effect['prev_frame'] = prev_frame
                    if 'intensity' in sig.parameters: params_to_pass_to_effect['intensity'] = current_intensity
                    
                    # --- Parameter range scaling ---
                    # This section attempts to scale ranges based on global_params and current_intensity.
                    # NOTE: This logic might not correctly match all parameter names if global_params keys
                    # (e.g., "perlin_alpha") don't align with derived keys from signatures (e.g., "alpha").
                    # A robust solution requires aligning these naming conventions.
                    for p_name_in_sig in sig.parameters:
                        if p_name_in_sig.endswith("_range"):
                            # Try to find a corresponding base key in current_visual_params
                            # e.g., for "alpha_range", look for "alpha" or "perlin_alpha", etc.
                            # This is the tricky part due to naming inconsistencies.
                            # Simplistic approach: use p_name_in_sig[:-6] as key_base
                            key_base = p_name_in_sig[:-6] 
                            
                            # Check if a direct match for key_base or a more specific key exists
                            # (e.g. "perlin_alpha" if key_base is "alpha" and effect is perlin_noise)
                            # For now, we assume key_base is the intended key in current_visual_params
                            if key_base in current_visual_params and isinstance(current_visual_params[key_base], tuple):
                                original_min, original_max = current_visual_params[key_base]
                                factor = 0.7 + 0.3 * current_intensity # Scale factor based on intensity
                                
                                scaled_min_val = original_min * factor
                                scaled_max_val = original_max * factor
                                if scaled_min_val > scaled_max_val: # Ensure min <= max
                                    scaled_min_val, scaled_max_val = scaled_max_val, scaled_min_val

                                final_range_tuple = ()
                                if any(k in p_name_in_sig for k in ["alpha", "intensity_val"]): # Float ranges
                                    final_range_tuple = (max(0.01, scaled_min_val), min(1.0, max(scaled_min_val + 0.05, scaled_max_val)))
                                elif p_name_in_sig == "oct_range": # Octave range might not scale like others
                                     final_range_tuple = (max(1, int(original_min)), max(int(original_min)+1, int(original_max))) # Use original for octaves
                                else: # Assume int ranges for others
                                    final_range_tuple = (max(1, int(scaled_min_val)), max(int(scaled_min_val) + 1, int(scaled_max_val)))
                                
                                if final_range_tuple:
                                     params_to_pass_to_effect[p_name_in_sig] = final_range_tuple
                            # If key_base not found or not a tuple, effect will use its default for this range.

                    if effect_func == apply_flickering_glitch_layer:
                        # Select a different effect to apply within the flicker
                        inner_effect_options = [f for f in effect_candidates if f != apply_flickering_glitch_layer]
                        if inner_effect_options:
                            inner_effect_func_selected = random.choice(inner_effect_options)
                            # Pass current_visual_params as kwargs to flicker, it will try to prepare them.
                            frame = effect_func(frame, intensity=current_intensity, 
                                                effect_func=inner_effect_func_selected, 
                                                prev_frame=prev_frame, 
                                                **current_visual_params) 
                        else: # No other effects to flicker with
                            pass 
                    else: # Standard effect call
                         frame = effect_func(frame, **params_to_pass_to_effect)
                    
                    applied_effects_count += 1
                    applied_this_frame_names.append(effect_name)
                except Exception as e:
                    print(f"Error applying visual effect {effect_name} at frame {i}: {e}")
                    # import traceback
                    # traceback.print_exc() # For more detailed debugging if needed
        
        effect_history.extend(applied_this_frame_names)
        effect_history = effect_history[-20:] # Keep history of last 20 applied effects

        # ARG Visuals (QR Codes, Text Codes, Protocol Messages)
        # QR Code Display
        if arg_qr_display_frames > 0:
            if arg_qr_img is not None and arg_qr_img.size > 0 :
                qr_x = random.randint(0, max(0,WIDTH - ARG_QR_SIZE))
                qr_y = random.randint(0, max(0,HEIGHT - ARG_QR_SIZE))
                temp_qr_to_display = arg_qr_img.copy()
                
                # Glitch the QR code itself slightly based on intensity
                if random.random() < 0.3*current_intensity:
                    qr_pixel_min = max(2, int(3 - current_intensity)) # More pixelated if low intensity
                    qr_pixel_max = max(qr_pixel_min + 1, int(10 - current_intensity*2))
                    temp_qr_to_display = apply_pixelation(temp_qr_to_display, block_size_range=(qr_pixel_min, qr_pixel_max))
                
                if qr_y + ARG_QR_SIZE <= HEIGHT and qr_x + ARG_QR_SIZE <= WIDTH:
                     frame_roi = frame[qr_y:qr_y+ARG_QR_SIZE, qr_x:qr_x+ARG_QR_SIZE]
                     # Ensure temp_qr has same channels as frame_roi (both should be BGR 3-channel)
                     if temp_qr_to_display.shape[2] != frame_roi.shape[2]: # Should not happen if generate_qr_code_image is correct
                         temp_qr_to_display = cv2.cvtColor(temp_qr_to_display, cv2.COLOR_BGR2GRAY)
                         temp_qr_to_display = cv2.cvtColor(temp_qr_to_display, cv2.COLOR_GRAY2BGR)
                     
                     if frame_roi.shape == temp_qr_to_display.shape:
                          # Blend QR code, make it fade in/out with display_frames
                          blend_alpha = 0.9 * np.clip(arg_qr_display_frames / (FPS*0.5), 0.1, 1.0) # Fade out effect
                          frame[qr_y:qr_y+ARG_QR_SIZE, qr_x:qr_x+ARG_QR_SIZE] = cv2.addWeighted(frame_roi, 1.0 - blend_alpha, temp_qr_to_display, blend_alpha,0)
            arg_qr_display_frames -=1
        elif random.random() < (0.003 + 0.007 * current_intensity) * global_params.get("arg_qr_prob_modifier", 1.0): # Lower base probability
            qr_data_content = "".join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(8,15)))
            qr_data = current_theme_mutated.get("arg_elements",{}).get("qr_data_prefix","QR_") + qr_data_content
            arg_qr_img = generate_qr_code_image(qr_data, ARG_QR_SIZE)
            if arg_qr_img.size > 0 : arg_qr_display_frames = int(FPS * random.uniform(0.7, 2.5)) # Display longer
            # print(f"ARG: Displaying QR Code: {qr_data} for {arg_qr_display_frames} frames")

        # ARG Text Code Display
        if arg_text_display_frames > 0:
            if arg_text_code:
                txt_scale_arg = ARG_TEXT_FONT_SCALE_BASE * (0.8 + current_intensity*0.4) # Scale with intensity
                text_size_arg_cv2, _ = cv2.getTextSize(arg_text_code, FONT_FACE_TO_USE, txt_scale_arg, 2)
                
                # Ensure text stays on screen
                text_x_arg = random.randint(10, max(11, WIDTH - text_size_arg_cv2[0] - 20)) 
                text_y_arg = random.randint(text_size_arg_cv2[1] + 20, max(text_size_arg_cv2[1]+21, HEIGHT - 20))
                
                # Text color with alpha-like blending (by drawing on overlay)
                text_alpha_blend = np.clip(arg_text_display_frames / (FPS*0.3), 0.2, 1.0) # Fade out
                text_color_arg = (200,220,255) # Base color (light blue/white)
                
                # Create overlay for text to blend with alpha
                overlay_text_arg = frame.copy()
                cv2.putText(overlay_text_arg, arg_text_code, (text_x_arg, text_y_arg), FONT_FACE_TO_USE, 
                            txt_scale_arg, text_color_arg, random.randint(1,2), cv2.LINE_AA)
                frame = cv2.addWeighted(frame, 1.0 - text_alpha_blend*0.7, overlay_text_arg, text_alpha_blend*0.7, 0) # Blend factor for text

            arg_text_display_frames -=1
        elif random.random() < (0.005 + 0.015 * current_intensity) * global_params.get("arg_text_prob_modifier", 1.0): # Lower base probability
            arg_text_code = generate_arg_text_code(current_theme_mutated, length=random.randint(20,50))
            arg_text_display_frames = int(FPS * random.uniform(0.5, 2.0)) # Display longer
            # print(f"ARG: Displaying Text Code: {arg_text_code} for {arg_text_display_frames} frames")

        # ARG Fake Errors / Protocol Messages
        if random.random() < (0.01 + 0.03 * current_intensity) * global_params.get("arg_protocol_prob_modifier", 1.0): # Lower base probability
            arg_elements_proto = current_theme_mutated.get("arg_elements", {})
            error_list_proto = arg_elements_proto.get("fake_errors", [])
            key_list_proto = arg_elements_proto.get("cipher_keys", [])
            protocol_text_content = ""
            
            if error_list_proto and (random.random() < 0.6 or not key_list_proto): # Prefer errors if available
                protocol_text_content = random.choice(error_list_proto)
            elif key_list_proto:
                protocol_text_content = "KEY_FRAGMENT: " + random.choice(key_list_proto)
            
            if protocol_text_content:
                font_scale_proto = ARG_ERROR_CODE_FONT_SCALE * (0.9 + current_intensity*0.6) # Larger for impact
                text_size_proto, _ = cv2.getTextSize(protocol_text_content, FONT_FACE_TO_USE, font_scale_proto, 3)
                
                # Position more centrally for protocol messages
                text_x_proto = max(10, (WIDTH - text_size_proto[0])//2 + random.randint(-50,50))
                text_y_proto = max(text_size_proto[1]+10, (HEIGHT + text_size_proto[1])//2 + random.randint(-HEIGHT//5, HEIGHT//5))
                
                # Flashy color, depends on background brightness
                mean_bg_brightness = np.mean(frame[text_y_proto-text_size_proto[1]:text_y_proto+10, text_x_proto:text_x_proto+text_size_proto[0]]) if text_y_proto-text_size_proto[1] >=0 else np.mean(frame)

                flash_text_color_proto = (255,random.randint(0,80),random.randint(0,80)) if mean_bg_brightness > 100 else (230,random.randint(200,255),random.randint(200,255))
                
                # Draw on overlay and blend for a "flash" effect
                overlay_proto = frame.copy()
                cv2.putText(overlay_proto, protocol_text_content, (text_x_proto, text_y_proto), FONT_FACE_TO_USE, 
                            font_scale_proto, flash_text_color_proto, random.randint(2,4), cv2.LINE_AA)
                frame = cv2.addWeighted(frame, 0.4, overlay_proto, 0.6,0) # Stronger blend for protocol messages
                # print(f"ARG: Flashing Protocol Text: {protocol_text_content}")
        
        # Subliminal Flash (quick, single frame visual jolt)
        flash_probability_subliminal = 0.10 + 0.15 * current_intensity # Higher chance with intensity
        if random.random() < flash_probability_subliminal:
            flash_type_sub = random.choice(["invert", "text_short", "symbol_large", "color_wash", "noise_burst_viz", "contrast_spike", "ascii_flash_brief"])
            overlay_sub = frame.copy() # Work on a copy
            try:
                if flash_type_sub == "invert": 
                    overlay_sub = 255 - frame
                elif flash_type_sub == "text_short":
                    sub_text = random.choice(current_theme_mutated["words"])
                    if random.random() < 0.5: sub_text = zalgo_text(sub_text[:5]) # Short, zalgoed
                    cv2.putText(overlay_sub, sub_text, (random.randint(50, WIDTH-200), random.randint(50, HEIGHT-50)), FONT_FACE_TO_USE, random.uniform(1.5,3.5), random.choice(random.choice(current_theme_mutated["colors"]))[:3], random.randint(2,4), cv2.LINE_AA)
                elif flash_type_sub == "symbol_large":
                    sub_sym = random.choice(current_theme_mutated["symbols"])
                    cv2.putText(overlay_sub, sub_sym, (random.randint(WIDTH//4, WIDTH//2), random.randint(HEIGHT//4, HEIGHT//2)), FONT_FACE_TO_USE, random.uniform(5.0,10.0), random.choice(random.choice(current_theme_mutated["colors"]))[:3], random.randint(3,6), cv2.LINE_AA)
                elif flash_type_sub == "color_wash":
                    wash_color = random.choice(random.choice(current_theme_mutated["colors"]))[:3]
                    color_layer = np.full_like(frame, wash_color, dtype=np.uint8)
                    overlay_sub = cv2.addWeighted(frame, 0.2, color_layer, 0.8,0)
                elif flash_type_sub == "noise_burst_viz":
                    overlay_sub = apply_perlin_noise(overlay_sub, alpha_range=(0.7,0.95), scale_range=(5,20), oct_range=(5,7)) # Intense, small scale noise
                elif flash_type_sub == "contrast_spike":
                    overlay_sub = apply_extreme_contrast(overlay_sub, alpha_range=(3.0,8.0), beta_range=(-50,50))
                elif flash_type_sub == "ascii_flash_brief":
                     overlay_sub = apply_ascii_sim(overlay_sub, block_size_range=(max(2,int(8-current_intensity*1.5)),max(3,int(15-current_intensity*3))), invert=random.choice([True,False]))
                
                # Blend the subliminal overlay strongly for one frame
                frame = cv2.addWeighted(frame, random.uniform(0.0,0.2), overlay_sub, random.uniform(0.8,1.0),0) 
            except Exception as e_sub: 
                # print(f"Subliminal flash error: {e_sub}")
                pass 

        # Persistent Text Overlay
        # Create a transparent overlay for text if alpha blending is desired,
        # otherwise draw directly. For simplicity, drawing directly here.
        # True alpha blending per text item is more complex with cv2.putText.
        frame_with_text = frame.copy() # Draw text on this copy
        active_persistent_texts_new = []
        for text_info in persistent_texts:
            px, py = text_info["pos"]
            mx, my = text_info["move_speed"]
            
            # Update position
            text_info["pos"] = (int(px+mx), int(py+my))
            
            # Bounce off screen edges (approximate, based on text start, not full width/height)
            text_w_approx = len(text_info["text"]) * int(text_info["font_size"] * 10) # Very rough width
            if text_info["pos"][0] < -text_w_approx or text_info["pos"][0] > WIDTH: text_info["move_speed"] = (-mx, my)
            if text_info["pos"][1] < -30 or text_info["pos"][1] > HEIGHT + 30: text_info["move_speed"] = (mx, -my) # 30 for font height approx
            
            text_info["frame_count"] +=1
            if text_info["frame_count"] < text_info["lifetime"]:
                try:
                    # Scale font size slightly with intensity for dynamism
                    display_font_size_eff = max(0.1, text_info["font_size"] * (0.8 + current_intensity*0.25))
                    thickness_eff = max(1, int(display_font_size_eff // 2)) # Thickness scales with font size
                    
                    # Use text_info["color"] which should be (B,G,R)
                    cv2.putText(frame_with_text, text_info["text"], text_info["pos"], FONT_FACE_TO_USE, 
                                display_font_size_eff, text_info["color"], thickness_eff, cv2.LINE_AA)
                    active_persistent_texts_new.append(text_info)
                except Exception as e_pt: 
                    # print(f"Error drawing persistent text: {e_pt}")
                    active_persistent_texts_new.append(text_info) # Keep it even if drawing failed once
            elif random.random() < 0.75: # High chance to respawn
                # Respawn with new properties
                new_word = str(random.choice(current_theme_mutated["words"]))
                new_symbol = str(random.choice(current_theme_mutated["symbols"]))
                if random.random() < current_theme_mutated.get("zalgo_words_chance", 0.3): new_word = zalgo_text(new_word)
                text_info["text"] = new_word + new_symbol
                text_info["pos"] = (random.randint(-WIDTH//4, WIDTH), random.randint(-HEIGHT//4, HEIGHT))
                text_info["font_size"] = random.uniform(0.4, 5.0)
                text_info["color"] = tuple(c for c in random.choice(random.choice(current_theme_mutated["colors"]))[:3])
                text_info["lifetime"] = random.randint(int(FPS * 0.5), int(FPS * 15))
                text_info["frame_count"] = 0
                text_info["move_speed"] = (random.uniform(-8, 8), random.uniform(-6, 6))
                active_persistent_texts_new.append(text_info)
        persistent_texts = active_persistent_texts_new
        frame = frame_with_text # Assign frame with text drawn on it

        # End of frame processing
        prev_frame = frame.copy() # For effects that use previous frame
        if frame_hold_counter > 0 and held_frame is None: # If this is the first frame of a hold sequence
            held_frame = frame.copy()
            
        video.write(frame)
        
        if (i + 1) % (FPS * 5) == 0 and frame_hold_counter == 0: # Log progress less frequently
            print(f"Video Frame {i+1}/{FRAME_COUNT} (Intensity: {current_intensity:.2f})...")
            
    video.release()
    return DURATION, FPS, video_temp_path

# --- Audio Generation ---
def generate_audio_enhanced(video_duration, video_fps, theme_data, intensity_func, global_params):
    DURATION = video_duration
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    if NUM_SAMPLES <= 0: return None # No audio if duration is zero
    
    NUM_CHANNELS = 2 # Stereo
    samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32)
    
    current_audio_params = copy.deepcopy(global_params["audio"]) # Mutable copy for this run
    audio_instability_chance = global_params["instability_chance"] * 1.2 # Slightly higher for audio
    
    print(f"Generating {DURATION:.1f}s audio, Theme: {global_params['theme_name']}...")
    
    min_freq, max_freq = theme_data["audio_freq_range"]
    noise_types_for_theme = theme_data["audio_noise_types"]
    features_for_theme = theme_data["audio_features"]
    
    current_theme_mutated_audio = copy.deepcopy(theme_data) # For audio-specific mutations if any

    last_event_end_sample = 0
    melody_track = np.zeros(NUM_SAMPLES, dtype=np.float32) # Separate track for melody
    biometric_track_mono = np.zeros(NUM_SAMPLES, dtype=np.float32) # Mono biometric track

    # Pre-generate biometric track if feature is present
    if "biometric_rhythm" in features_for_theme or any("heartbeat" in nt or "breathing" in nt for nt in noise_types_for_theme):
        # Choose a biometric type from available noise types or a default
        biometric_noise_options = [nt for nt in noise_types_for_theme if "heart" in nt or "breath" in nt or "biometric" in nt]
        chosen_biometric_type = random.choice(biometric_noise_options) if biometric_noise_options else "heartbeat_distorted"
        
        # Generate biometric track with varying intensity based on global intensity profile
        # For simplicity, using an average intensity for the whole track, or could modulate it.
        avg_intensity_for_bio = np.mean([intensity_func(t) for t in np.linspace(0, DURATION, 10)])
        biometric_vol = 0.15 + 0.25 * avg_intensity_for_bio # Scale volume with average intensity
        biometric_track_mono = generate_biometric_rhythm(NUM_SAMPLES, biometric_vol, rhythm_type=chosen_biometric_type)

    # Main audio event loop
    while last_event_end_sample < NUM_SAMPLES:
        current_sample_time = last_event_end_sample / SAMPLE_RATE
        try:
            current_intensity = intensity_func(current_sample_time)
        except ValueError: # Handle case where time might be outside interpolation range (shouldn't happen if fill_value used)
            current_intensity = 1.0 
        current_intensity = np.clip(current_intensity, 0.01, 4.0)

        # Audio parameter instability
        if random.random() < audio_instability_chance * current_intensity:
            param_key_audio = random.choice(list(current_audio_params.keys()))
            if isinstance(current_audio_params[param_key_audio], tuple) and len(current_audio_params[param_key_audio]) == 2:
                c_min,c_max=current_audio_params[param_key_audio]
                range_w=abs(c_max-c_min)+1e-6
                mid_s=random.uniform(-range_w*0.15,range_w*0.15)
                scale_f=random.uniform(0.8,1.2)
                n_mid=(c_min+c_max)/2+mid_s
                n_w=range_w*scale_f
                n_min=n_mid-n_w/2
                n_max=n_mid+n_w/2
                # Apply type-specific constraints for audio params
                if "depth" in param_key_audio: # e.g., bitcrush_depth
                    n_min=max(2,int(n_min))
                    n_max=max(n_min+1, int(n_max))
                elif "intensity" in param_key_audio: # e.g. distortion_intensity
                     n_min = max(1.0, n_min)
                     n_max = max(n_min + 0.1, n_max)
                current_audio_params[param_key_audio]=(n_min,n_max)

        # Event duration and volume
        event_max_duration_sec = max(0.01, (0.5 + 1.0/(current_intensity+0.5))) # Shorter events with high intensity
        event_min_duration_sec = 0.002
        event_duration_sec = random.uniform(event_min_duration_sec, event_max_duration_sec)
        
        event_duration_samples = min(max(1,int(event_duration_sec*SAMPLE_RATE)), NUM_SAMPLES - last_event_end_sample)
        if event_duration_samples <= 0: break # No more space for events

        event_start_sample = last_event_end_sample
        event_end_sample = event_start_sample + event_duration_samples
        
        event_vol_base = random.uniform(0.01,0.5)
        event_vol_scaled = np.clip(event_vol_base * (0.1 + current_intensity * 1.2), 0.001, 1.0)

        # Choose event type: silence, melody, noise, tone
        type_roll = random.random()
        silence_thresh = max(0.005, 0.1 * (1.5 - current_intensity)) # More silence if low intensity
        melody_thresh = silence_thresh + (0.05 + 0.15 * current_intensity) # Higher chance of melody with intensity
        noise_thresh = melody_thresh + (0.5 + 0.20 * current_intensity) # Higher chance of noise

        segment_mono = np.zeros(event_duration_samples,dtype=np.float32)
        event_type="silence"
        is_melody_event = False

        if type_roll < silence_thresh:
            pass # Stays silent
        elif type_roll < melody_thresh:
            event_type="melody"
            is_melody_event = True
            melody_vol_mod = 0.2 + 0.4*current_intensity # Melody slightly louder with intensity
            segment_mono = generate_melody_fragment(event_duration_samples, event_vol_scaled * melody_vol_mod, current_theme_mutated_audio, current_intensity)
        elif type_roll < noise_thresh:
            event_type="noise"
            if not noise_types_for_theme: noise_choice = "white" # Fallback if theme has no noise types
            else: noise_choice = random.choice(noise_types_for_theme)
            
            # Prioritize harsher noises if intensity is high
            if current_intensity > 2.0 and random.random() < 0.7:
                 harsh_noise_options=[n for n in noise_types_for_theme if any(k in n for k in["screech","clip","feedback","artifact","data_stream","glitch"])]
                 if harsh_noise_options: noise_choice=random.choice(harsh_noise_options)
            segment_mono = generate_noise(noise_choice, event_duration_samples, event_vol_scaled, features_for_theme)
        else:
            event_type="tone"
            fm_chance_eff=np.clip(0.2+current_intensity*0.5,0.1,0.95)
            harmonic_chance_eff=np.clip(0.1+current_intensity*0.4,0.1,0.9)
            freq_val_event=random.uniform(min_freq, max_freq * (0.3 + current_intensity * 0.7)) # Wider freq range with intensity
            segment_mono = generate_tone(np.clip(freq_val_event,LOW_FREQ_CUTOFF,HIGH_FREQ_CUTOFF), 
                                   event_duration_samples,event_vol_scaled,fm_chance_eff,harmonic_chance_eff)

        # Ensure segment_mono has correct length if generation failed or was short
        if len(segment_mono) != event_duration_samples and event_duration_samples > 0:
            padded_segment_mono=np.zeros(event_duration_samples,dtype=np.float32)
            len_to_copy_mono=min(len(segment_mono),event_duration_samples)
            if len_to_copy_mono > 0: padded_segment_mono[:len_to_copy_mono]=segment_mono[:len_to_copy_mono]
            segment_mono=padded_segment_mono
        
        # Apply audio effects to the generated segment (if not silence)
        if event_type != "silence" and len(segment_mono) > 0:
            effect_application_chance_mod = 0.5 + current_intensity * 0.5 # Higher chance to apply effects with intensity
            
            if "distortion" in features_for_theme and random.random() < 0.5 * effect_application_chance_mod:
                d_min,d_max = current_audio_params.get("distortion_intensity", (1.0, 25.0))
                dist_range_eff = (d_min*(0.5+current_intensity*1.0), d_max*(0.8+current_intensity*1.2))
                segment_mono = apply_distortion(segment_mono,intensity_range=np.clip(dist_range_eff,1.0,50.0)) # Cap max distortion
            
            if "bitcrush" in features_for_theme and random.random() < 0.4 * effect_application_chance_mod:
                 b_min_bits,b_max_bits = current_audio_params.get("bitcrush_depth", (2,8))
                 # Crush more (lower bit depth) with higher intensity
                 target_bits_eff = int(max(2, b_min_bits + (b_max_bits - b_min_bits) * (1.0 - current_intensity*0.2) ))
                 target_bits_eff = max(2, min(target_bits_eff, 16)) # Clamp target bits
                 segment_mono = apply_bitcrush(segment_mono,bit_depth_range=(2, max(2,target_bits_eff)))
            
            if "spectral_glitch" in features_for_theme and random.random() < 0.3 * effect_application_chance_mod:
                segment_mono = apply_spectral_glitch(segment_mono,intensity=current_intensity*1.1) # More intense spectral glitch
            
            if "broken_speaker_sim" in features_for_theme and random.random() < 0.15 * effect_application_chance_mod: # Less frequent
                segment_mono = apply_broken_speaker_sim(segment_mono,current_intensity)
            
            reverb_feature_name = next((f for f in features_for_theme if "convolution_reverb" in f), None)
            if reverb_feature_name and random.random() < 0.1 * effect_application_chance_mod: # Less frequent
                 try:
                      impulse_name_key = reverb_feature_name.split("_")[-1] # e.g., "damp" from "convolution_reverb_damp"
                      if impulse_name_key in IMPULSE_RESPONSES:
                          segment_mono = apply_convolution_reverb(segment_mono,IMPULSE_RESPONSES[impulse_name_key])
                 except Exception as e_rev:
                     # print(f"Reverb error: {e_rev}")
                     pass
            
            # Panning
            pan_extremity_eff = np.clip(0.3 + current_intensity*0.7, 0.1, 1.0) # Pan wider with intensity
            pan_val = random.uniform(-pan_extremity_eff, pan_extremity_eff) # -1 (left) to 1 (right)
            gain_l_pan = np.sqrt(0.5 * (1 - pan_val))
            gain_r_pan = np.sqrt(0.5 * (1 + pan_val))
            
            # Add to main stereo mix or melody track
            end_idx_safe = min(event_end_sample, NUM_SAMPLES) # Ensure not writing past buffer
            length_to_write = end_idx_safe - event_start_sample
            if length_to_write <=0: continue

            if is_melody_event:
                melody_track[event_start_sample:end_idx_safe] += segment_mono[:length_to_write]
            else:
                samples[event_start_sample:end_idx_safe,0] += segment_mono[:length_to_write] * gain_l_pan
                samples[event_start_sample:end_idx_safe,1] += segment_mono[:length_to_write] * gain_r_pan
        
        # Determine overlap for next event: more overlap with higher intensity for denser soundscape
        overlap_factor_eff = np.clip(0.1 + current_intensity*0.6, 0.05, 0.9) # From 5% to 90% overlap
        advance_samples_count = int(event_duration_samples * (1.0 - overlap_factor_eff))
        last_event_end_sample += max(1, advance_samples_count) # Ensure progress

    # Mix in biometric track (panned centrally or slightly stereo)
    bio_pan = random.uniform(-0.2, 0.2) # Slight stereo spread for biometric
    bio_gain_l = np.sqrt(0.5 * (1 - bio_pan))
    bio_gain_r = np.sqrt(0.5 * (1 + bio_pan))
    # Scale biometric track volume by overall intensity at end (or average)
    intensity_at_end_val = global_params.get("intensity_at_end",1.0)
    biometric_mix_level = 0.15 + 0.20 * intensity_at_end_val 
    samples[:,0] += biometric_track_mono * biometric_mix_level * bio_gain_l
    samples[:,1] += biometric_track_mono * biometric_mix_level * bio_gain_r
    
    # Normalize and mix in melody track
    max_melody_abs_val = np.max(np.abs(melody_track))
    if max_melody_abs_val > 1e-6: melody_track /= max_melody_abs_val # Normalize melody before mixing
    
    melody_mix_level_final = 0.20 + 0.25 * intensity_at_end_val # Melody volume based on end intensity
    samples[:,0] += melody_track * melody_mix_level_final
    samples[:,1] += melody_track * melody_mix_level_final
    
    # Final normalization of the entire mix
    max_abs_amplitude_final = np.max(np.abs(samples))
    if max_abs_amplitude_final > 1e-6:
        samples /= max_abs_amplitude_final
    elif NUM_SAMPLES > 0 : # If it's silent but was supposed to have sound
        print("Warning: Final audio mix is silent or near silent.")
        
    # Convert to 16-bit PCM
    samples_int16=(np.clip(samples,-1.0,1.0)*32767).astype(np.int16)
    
    audio_temp_path=f"audio_temp_{int(time.time())}_{random.randint(100,999)}.wav"
    try:
        with wave.open(audio_temp_path,'wb') as wf:
            wf.setnchannels(NUM_CHANNELS)
            wf.setsampwidth(2) # 2 bytes for 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples_int16.tobytes())
        return audio_temp_path
    except Exception as e:
        print(f"Error writing WAV file: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    start_time_main = time.time()
    
    initial_theme_name = random.choice(list(THEMES.keys()))
    # Make a deepcopy of the chosen theme to avoid modifying the original THEMES dict
    current_theme_data_master = copy.deepcopy(THEMES[initial_theme_name])
    
    print(f"--- Starting Shardmind Protocol Generation v3.1 ---")
    print(f"Initial Theme: {initial_theme_name}")
    
    # Global parameters, including visual effect ranges.
    # NOTE: The keys for ranges (e.g., "perlin_alpha") might not perfectly align with
    # how `generate_frames_enhanced` tries to parse them from function signatures (e.g., "alpha" from "alpha_range").
    # This means dynamic scaling/instability of these ranges might not work as intended for all effects
    # without a more robust parameter name mapping system.
    global_params = { 
        "theme_name": initial_theme_name,
        "visual": { 
            "catalyst_threshold": 2.8, # Intensity level to trigger "catalyst" events
            "effect_probabilities": { # Base probability for each effect
                "apply_perlin_noise":0.7, "apply_block_shift":0.6, "apply_color_channel_shift":0.5, 
                "apply_warp":0.5, "apply_pixelation":0.4, "apply_scanlines":0.3, 
                "apply_solarize":0.2, "apply_extreme_contrast":0.5, "apply_feedback":0.6, 
                "apply_datamosh_sim":0.3, "apply_ascii_sim":0.2, "apply_sensor_overload":0.25,
                "apply_data_bleed_sim":0.2, "apply_flickering_glitch_layer":0.25, 
                "apply_crt_ghost_sim":0.15, "apply_slit_scan_sim":0.1, 
                "apply_vector_field_sim":0.1, "apply_figurative_noise_sim":0.1
            },
            # Default ranges for visual effect parameters (these keys should ideally match derived keys)
            # Example: for apply_perlin_noise(alpha_range=...), key might be "alpha" or "perlin_alpha"
            "perlin_alpha": (0.1, 0.9), "perlin_scale": (2.0, 80.0), "perlin_oct": (3,9), # Added octaves
            "block_shift_max_shift": (10, WIDTH // 3), "block_shift_block_size": (5, HEIGHT // 2), # More specific keys
            "color_channel_shift_max_shift": (5, 60),
            "warp_amplitude": (5, 80), "warp_freq": (0.002, 0.1),
            "pixelation_block_size": (4, 96), # Specific key for pixelation's block_size_range
            "extreme_contrast_alpha": (1.0, 6.0), "extreme_contrast_beta": (-100, 100),
            "feedback_alpha": (0.02, 0.6),
            "solarize_threshold": (60, 200),
            "scanlines_intensity": (0.4, 1.0), "scanlines_thickness": (1,4),
            "datamosh_sim_block_size": (32,128), 
            "ascii_sim_block_size": (8,16)
            # Add other effect parameters here if they need global control / instability
        },
        "audio": { # Parameters for audio effects
            "distortion_intensity": (1.0, 25.0), 
            "bitcrush_depth": (2, 8) # Bit depth range (e.g. 2 bits to 8 bits)
        },
        "instability_chance": 0.012, # Base chance for parameters to mutate per frame/event
        # ARG probabilities can be adjusted here if needed
        "arg_qr_prob_modifier": 1.0, 
        "arg_text_prob_modifier": 1.0, 
        "arg_protocol_prob_modifier": 1.0,
        "intensity_at_end": 1.0 # Placeholder, will be updated by intensity profile
    }

    # Generate intensity profile once, for both video and audio
    # Estimate max duration slightly longer to ensure profile covers full range
    estimated_max_duration_for_profile_sec = MAX_DURATION + 10 
    estimated_max_samples_for_profile = int(estimated_max_duration_for_profile_sec * SAMPLE_RATE)
    intensity_function = generate_intensity_profile(estimated_max_samples_for_profile, SAMPLE_RATE)
    
    # Get intensity near the end for audio mixing balance
    try:
        global_params["intensity_at_end"] = np.clip(intensity_function(MAX_DURATION - 2), 0.1, 3.0) # Check 2s before max
    except ValueError: # If MAX_DURATION-2 is out of bounds for interp1d (shouldn't be with fill_value)
        global_params["intensity_at_end"] = 1.0


    # Generate video frames
    actual_duration_video, actual_fps_video, video_temp_file_path = generate_frames_enhanced(
        current_theme_data_master, intensity_function, global_params
    )
    if not video_temp_file_path: 
        print("Video generation failed!")
        exit(1)
    
    # Generate audio
    audio_temp_file_path = generate_audio_enhanced(
        actual_duration_video, actual_fps_video, current_theme_data_master, intensity_function, global_params
    )
    if not audio_temp_file_path: 
        print("Audio generation failed!")
        if video_temp_file_path and os.path.exists(video_temp_file_path): os.remove(video_temp_file_path)
        exit(1)

    # Combine video and audio using ffmpeg
    if video_temp_file_path and audio_temp_file_path:
        print("Combining video and audio with ffmpeg...")
        if os.path.exists(OUTPUT_FILE):
            try: 
                os.remove(OUTPUT_FILE)
            except OSError as e: 
                print(f"Could not remove existing output file '{OUTPUT_FILE}': {e}")
        
        # Ensure ffmpeg can be found (add to PATH or provide full path)
        ffmpeg_command = [ 
            'ffmpeg', '-y', 
            '-i', video_temp_file_path, 
            '-i', audio_temp_file_path, 
            '-c:v', 'libx264',     # Video codec
            '-preset', 'medium',   # Encoding speed/quality trade-off
            '-crf', '23',          # Constant Rate Factor (quality, lower is better, 18-28 good range)
            '-pix_fmt', 'yuv420p', # Pixel format for compatibility
            '-c:a', 'aac',         # Audio codec
            '-b:a', '192k',        # Audio bitrate
            '-shortest',           # Finish encoding when the shortest input stream ends
            OUTPUT_FILE 
        ]
        try:
            # Using capture_output=True can consume a lot of memory for verbose ffmpeg output.
            # For long videos, consider redirecting to files or using None.
            result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            print(f"Successfully created video: '{OUTPUT_FILE}'")
            # if result.stdout: print("FFMPEG STDOUT:", result.stdout) # Optional: print ffmpeg output
            # if result.stderr: print("FFMPEG STDERR:", result.stderr) # stderr often has progress/info
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg command failed with exit code {e.returncode}.")
            print(f"FFMPEG STDOUT: {e.stdout}")
            print(f"FFMPEG STDERR: {e.stderr}")
        except FileNotFoundError:
            print("ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
        except Exception as e_ffmpeg:
            print(f"An unexpected error occurred during ffmpeg execution: {e_ffmpeg}")
            
    # Clean up temporary files
    for temp_file_to_clean in [video_temp_file_path, audio_temp_file_path]:
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            try: 
                os.remove(temp_file_to_clean)
                # print(f"Removed temp file: {temp_file_to_clean}")
            except OSError as e_clean: 
                print(f"Error removing temporary file {temp_file_to_clean}: {e_clean}")
                
    print(f"--- Total Time: {time.time() - start_time_main:.2f}s ---")










