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

# --- Configuration (Performance Tweaks Applied) ---
WIDTH, HEIGHT = 1280, 720 # HD Resolution
MIN_DURATION, MAX_DURATION = 20, 45 # Reduced max duration
MIN_FPS, MAX_FPS = 15, 25 # Reduced max FPS
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
    text = str(text_input) 
    zalgo_chars_combined = [chr(i) for i in range(0x0300, 0x036F + 1)] 
    all_zalgo = zalgo_chars_combined
    if not all_zalgo: return text 
    output = ""
    for char_original in text:
        output += char_original
        num_zalgo = random.randint(1, 7) # Slightly reduced Zalgo intensity
        for _ in range(num_zalgo):
            output += random.choice(all_zalgo)
    return output

# --- THEMES ---
THEMES = {
    "Shardmind Protocol": {
        "colors": [
            [(0,0,0), (255,255,255), (0,100,255), (255,0,100)], 
            [(10,10,30), (200,220,255), (50,200,50), (100,100,100)], 
        ],
        "words": ["TRANSMIT", "RECEIVE", "PACKET", "CORRUPT", "SIGNAL", "PROTOCOL", "ENTITY", "SHARD", "MIND", "QUERY", "RESPONSE", "DECODE", "ENCRYPT", "MANIFEST"],
        "symbols": ["::", "->", "||", "&&", "0x", "#>", "%$", "§", "∆", "∑", "Ω", "サイバー"], 
        "zalgo_words_chance": 0.4, # Slightly reduced
        "visual_effects": ["apply_perlin_noise", "apply_block_shift", "apply_color_channel_shift", "apply_warp", "apply_pixelation", "apply_scanlines", "apply_extreme_contrast", "apply_feedback", "apply_datamosh_sim", "apply_ascii_sim", "apply_sensor_overload", "apply_flickering_glitch_layer"],
        "audio_freq_range": (20, 20000),
        "audio_noise_types": ["digital_artifact", "static", "modem_sim", "feedback_screech", "glitch", "white_bursts", "data_stream_sim"],
        "audio_features": ["stutter", "bitcrush", "extreme_panning", "distortion", "spectral_glitch", "dtmf_bursts", "morse_fragments"],
        "melody_scale": "chromatic",
        "melody_distortion": 15.0, # Reduced
        "melody_counter_melody_chance": 0.5,
        "arg_elements": {
            "fake_errors": ["ERR_SYNC_FAIL", "PACKET_CRC_MISMATCH", "MEM_ACCESS_VIOLATION_0XFFFE", "UNKNOWN_OPCODE", "ENTITY_STREAM_INTERRUPTED"],
            "cipher_keys": ["ALPHA", "OMEGA", "NULL", "VOID", "777", "13", "3301"], 
            "qr_data_prefix": "SHARD_LINK_",
            "text_code_types": ["base64", "hex", "binary_short"]
        }
    },
    "Flesh Interface": {
        "colors": [
            [(30,0,0), (150,20,20), (200,180,170), (80,50,40)], 
            [(0,0,0), (255,10,10), (100,5,5), (50,50,50)], 
        ],
        "words": ["MERGE", "INTEGRATE", "BIOFEEDBACK", "SYNAPSE", "NEURAL", "PULSE", "MUTATE", "CONSUME", "REJECT", "HOST", "PARASITE", "SYMBIOSIS"],
        "symbols": ["⚕️", "🧬", "🧠", "👁️", "〰️", "∬", "𝔅ℑ𝔒", "𝔐𝔈ℭℌ"], 
        "zalgo_words_chance": 0.6,
        "visual_effects": ["apply_perlin_noise", "apply_warp", "apply_feedback", "apply_data_bleed_sim", "apply_solarize", "apply_crt_ghost_sim", "apply_figurative_noise_sim", "apply_pixelation"],
        "audio_freq_range": (18, 8000),
        "audio_noise_types": ["heartbeat", "breathing", "squelch", "wet_clicks", "bone_cracking_sim", "granular_flesh", "choking_sim", "biometric_rhythm"], 
        "audio_features": ["wet_sounds", "slow_lfo", "dissonance", "low_rumble", "distortion", "extreme_pitch_bend", "broken_speaker_sim", "convolution_reverb_damp"],
        "melody_scale": "phrygian",
        "melody_distortion": 10.0, # Reduced
        "melody_counter_melody_chance": 0.3,
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
    "minor_pentatonic": [0, 3, 5, 7, 10], "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": list(range(12)), "major": [0, 2, 4, 5, 7, 9, 11],
    "locrian": [0, 1, 3, 5, 6, 8, 10], "augmented": [0, 4, 8],
    "whole_tone": [0, 2, 4, 6, 8, 10],
}

# --- Font Configuration ---
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_FACE_TO_USE = DEFAULT_FONT 

# --- Pygame Mixer Init ---
try:
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=2048) # Reduced buffer
except pygame.error as e:
    print(f"Pygame mixer init failed (non-critical for file output): {e}")

# --- Intensity Profile Generation ---
def generate_intensity_profile(duration_samples, sample_rate, scale=30.0, octaves=5, persistence=0.5, lacunarity=2.0): # Reduced octaves
    duration_seconds = duration_samples / sample_rate
    num_points = int(duration_seconds * 15) # Reduced points for interpolation
    if num_points < 2: num_points = 2 
    profile = np.zeros(num_points)
    base = random.randint(0, 1000)
    for i in range(num_points):
        n1 = noise.pnoise1(i / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base)
        n2 = noise.pnoise1(i / (scale*0.3) + 1000, octaves=2, persistence=0.7, lacunarity=1.8, base=base+1) # Reduced octaves
        profile[i] = (n1 * 0.7) + (n2 * 0.3)
    min_val, max_val = np.min(profile), np.max(profile)
    if max_val - min_val > 1e-6: 
        normalized_profile = (profile - min_val) / (max_val - min_val)
        intensity_min = 0.05 
        intensity_max = 3.5  
        scaled_profile = intensity_min + normalized_profile * (intensity_max - intensity_min)
    else:
        scaled_profile = np.full(num_points, (0.05 + 3.5) / 2) 
    profile_times = np.linspace(0, duration_seconds, num_points)
    intensity_func = interp1d(profile_times, scaled_profile, kind='linear', bounds_error=False, # Linear interpolation is faster
                              fill_value=(scaled_profile[0], scaled_profile[-1]))
    return intensity_func

# --- ARG Content Generation Functions ---
def generate_qr_code_image(data, size=ARG_QR_SIZE):
    try:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=max(1, size // 25), border=1) # Smaller box size
        qr.add_data(data)
        qr.make(fit=True)
        img_pil = qr.make_image(fill_color="white", back_color="black").convert('RGB')
        img_pil = img_pil.resize((size, size), Image.NEAREST) 
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # print(f"Error generating QR code for '{data}': {e}") # Silenced for speed
        return np.zeros((size, size, 3), dtype=np.uint8) 

def generate_arg_text_code(theme_data, length=12): # Reduced default length
    code_type_options = theme_data.get("arg_elements", {}).get("text_code_types", ["hex"])
    if not code_type_options: code_type_options = ["hex"] 
    code_type = random.choice(code_type_options)
    actual_length = random.randint(max(1, length // 2), length) 
    try:
        if code_type == "base64":
            raw_data = os.urandom(actual_length * 3 // 4 + 1) 
            return base64.b64encode(raw_data).decode('utf-8')[:actual_length]
        elif code_type == "hex":
            raw_data = os.urandom(actual_length // 2 + 1) 
            return binascii.hexlify(raw_data).decode('utf-8')[:actual_length]
        elif code_type == "binary_short": # This can be slow if length is large
            actual_length = min(actual_length, 24) # Cap binary string length
            raw_data = os.urandom(actual_length // 8 + 1) 
            return "".join(format(byte, '08b') for byte in raw_data)[:actual_length]
    except Exception as e:
        # print(f"Error generating ARG text code type {code_type}: {e}") # Silenced
        pass
    return "CODE_ERR"


# --- Visual Glitch Functions (some parameters tweaked for performance) ---
def apply_perlin_noise(frame, alpha_range=(0.1, 0.7), scale_range=(5.0, 60.0), oct_range=(2, 5)): # Reduced ranges/octaves
    alpha = random.uniform(*alpha_range)
    scale = random.uniform(*scale_range)
    octaves = random.randint(*oct_range)
    persistence = random.uniform(0.3, 0.6)
    lacunarity = random.uniform(1.9, 3.0)
    seed = random.randint(0, 1000) # Smaller seed range
    height, width = frame.shape[:2]
    x_coords_norm = np.arange(width) / scale
    y_coords_norm = np.arange(height) / scale
    gray_noise = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        y_coord = y_coords_norm[i]
        for j in range(width):
            x_coord = x_coords_norm[j]
            gray_noise[i, j] = noise.pnoise2(y_coord, x_coord, octaves=octaves,
                                             persistence=persistence, lacunarity=lacunarity, base=seed)
    min_gn, max_gn = np.min(gray_noise), np.max(gray_noise)
    if max_gn - min_gn > 1e-6: normalized_noise = (gray_noise - min_gn) / (max_gn - min_gn)
    else: normalized_noise = np.zeros_like(gray_noise)
    colored_noise = (normalized_noise * 255).astype(np.uint8)
    colored_noise = cv2.cvtColor(colored_noise, cv2.COLOR_GRAY2BGR)
    if random.random() < 0.4: colored_noise = 255 - colored_noise
    return cv2.addWeighted(frame, 1 - alpha, colored_noise, alpha, 0)

def apply_block_shift(frame, max_shift_range=(5, WIDTH // 4), block_size_range=(10, HEIGHT // 3), num_blocks_range=(10, 50)): # Reduced counts/ranges
    max_shift = random.randint(max(1,int(max_shift_range[0])), max(2,int(max_shift_range[1])))
    bs_min = max(1, int(block_size_range[0]))
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    num_blocks = random.randint(*num_blocks_range)
    height, width = frame.shape[:2]
    temp_frame = frame.copy()
    for _ in range(num_blocks):
        bh = random.randint(bs_min, bs_max); bw = random.randint(bs_min, bs_max) 
        if height - bh <= 0 or width - bw <= 0: continue 
        y = random.randint(0, height - bh -1); x = random.randint(0, width - bw -1)  
        shift_x = random.randint(-max_shift, max_shift); shift_y = random.randint(-max_shift, max_shift)
        target_y_start = np.clip(y + shift_y, 0, height - bh); target_x_start = np.clip(x + shift_x, 0, width - bw)
        try:
            block = frame[y:y+bh, x:x+bw]
            temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block
        except ValueError: pass 
    return temp_frame

# ... (other visual effects might need similar parameter tweaks for performance if they are bottlenecks) ...

def apply_color_channel_shift(frame, max_shift_range=(3, 40)): # Reduced shift
    max_shift = random.randint(max(1,int(max_shift_range[0])), max(2,int(max_shift_range[1])))
    temp_frame = frame.copy()
    for i in range(3): 
        shift_x = random.randint(-max_shift, max_shift); shift_y = random.randint(-max_shift, max_shift)
        channel = frame[:,:,i]; shifted_channel = np.roll(np.roll(channel, shift_y, axis=0), shift_x, axis=1)
        temp_frame[:,:,i] = shifted_channel
    return temp_frame

def apply_warp(frame, amplitude_range=(3, 50), freq_range=(0.003, 0.08)): # Reduced ranges
    rows, cols = frame.shape[:2]
    amplitude_x = random.uniform(*amplitude_range); frequency_x = random.uniform(*freq_range)
    amplitude_y = random.uniform(*amplitude_range) * random.uniform(0.4, 1.6) 
    frequency_y = random.uniform(*freq_range) * random.uniform(0.4, 1.6)   
    phase_x = random.uniform(0, 2 * math.pi); phase_y = random.uniform(0, 2 * math.pi)
    x_mesh, y_mesh = np.meshgrid(np.arange(cols), np.arange(rows))
    offset_x = (amplitude_x * np.sin(2 * math.pi * y_mesh * frequency_x + phase_x)).astype(np.float32)
    offset_y = (amplitude_y * np.cos(2 * math.pi * x_mesh * frequency_y + phase_y)).astype(np.float32) 
    map_x = np.clip(x_mesh.astype(np.float32) + offset_x, 0, cols - 1)
    map_y = np.clip(y_mesh.astype(np.float32) + offset_y, 0, rows - 1)
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_feedback(frame, prev_frame, alpha_range=(0.02, 0.5)): # Max alpha reduced
    if prev_frame is None: return frame
    alpha = random.uniform(*alpha_range)
    modified_prev = prev_frame.copy()
    if random.random() < 0.15: # Reduced chance of transform
        rows, cols = modified_prev.shape[:2]; angle = random.uniform(-3, 3); scale = random.uniform(0.97, 1.03)
        tx = random.uniform(-7, 7); ty = random.uniform(-7, 7); center = (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center, angle, scale); M[0, 2] += tx; M[1, 2] += ty
        modified_prev = cv2.warpAffine(modified_prev, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.08: # Reduced chance of color shift
         shift_bgr = (random.randint(-10,10),random.randint(-10,10),random.randint(-10,10))
         color_shift_matrix = np.full_like(modified_prev, shift_bgr, dtype=np.int16) 
         modified_prev = np.clip(modified_prev.astype(np.int16) + color_shift_matrix, 0, 255).astype(np.uint8)
    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)

def apply_pixelation(frame, block_size_range=(5, 70)): # Range adjusted
    height, width = frame.shape[:2]; bs_min = max(2, int(block_size_range[0])) 
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    block_size = random.randint(bs_min, bs_max) if bs_min < bs_max else bs_min
    block_size = max(1, block_size)
    pixel_w, pixel_h = max(1, width // block_size), max(1, height // block_size)
    temp = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

def apply_scanlines(frame, intensity_range=(0.3, 0.8), thickness_range=(1, 3), color_variation=30): # Tweaked
    intensity = random.uniform(*intensity_range); thickness = random.randint(*thickness_range)
    scanline_layer = np.zeros_like(frame); height, width = frame.shape[:2]
    base_color_val = random.randint(0, 60) 
    for y_coord in range(0, height, thickness * 2): 
        line_color_val = base_color_val + random.randint(-color_variation, color_variation)
        line_color = tuple(np.clip([line_color_val]*3, 0, 255)) 
        cv2.line(scanline_layer, (0, y_coord), (width, y_coord), line_color, thickness)
    return cv2.addWeighted(frame, 1.0, scanline_layer, intensity, 0)

def apply_solarize(frame, threshold_range=(70, 190)): # Tweaked
    threshold = random.randint(*threshold_range); solarized_frame = frame.copy()
    mask = frame > threshold; solarized_frame[mask] = 255 - solarized_frame[mask]
    return solarized_frame

def apply_extreme_contrast(frame, alpha_range=(1.0, 5.0), beta_range=(-80, 80)): # Tweaked
    alpha = random.uniform(*alpha_range); beta = random.randint(*beta_range)   
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def apply_datamosh_sim(frame, prev_frame, hold_prob=0.25, smear_prob=0.4, block_size_range=(40, 100)): # Tweaked
    if prev_frame is None: return frame
    height, width = frame.shape[:2]; output = frame.copy()
    bs_min = max(16, int(block_size_range[0])); bs_max = max(bs_min + 1, int(block_size_range[1]))
    block_size = random.randint(bs_min, bs_max) if bs_min < bs_max else bs_min
    block_size = max(1, block_size) 
    if random.random() < hold_prob:
        num_hold_blocks = random.randint(1, max(1, (width * height) // (block_size**2 * 4) )) # Reduced density
        for _ in range(num_hold_blocks):
            if width // block_size <=0 or height // block_size <=0: continue
            bx_idx = random.randint(0, max(0, width // block_size - 1)); by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            bh = min(block_size, height - by); bw = min(block_size, width - bx)
            if bh > 0 and bw > 0: output[by:by+bh, bx:bx+bw] = prev_frame[by:by+bh, bx:bx+bw]
    if random.random() < smear_prob:
        num_smear_blocks = random.randint(1, max(1, (width * height) // (block_size**2 * 5) )) # Reduced density
        for _ in range(num_smear_blocks):
            if width // block_size <=0 or height // block_size <=0: continue
            bx_idx = random.randint(0, max(0, width // block_size - 1)); by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            bh = min(block_size, height - by); bw = min(block_size, width - bx)
            if bh <= 0 or bw <= 0: continue
            vx = random.randint(-block_size // 3, block_size // 3); vy = random.randint(-block_size // 3, block_size // 3) # Reduced smear vector
            smear_block_src = prev_frame[by:by+bh, bx:bx+bw] 
            target_y = np.clip(by + vy, 0, height - bh); target_x = np.clip(bx + vx, 0, width - bw)
            if smear_block_src.shape[0] == bh and smear_block_src.shape[1] == bw:
                 output[target_y:target_y+bh, target_x:target_x+bw] = smear_block_src
    return output

def apply_ascii_sim(frame, block_size_range=(10, 20), char_set=" .:-=#%@", invert=False): # Larger blocks, simpler charset
    height, width = frame.shape[:2]; bs_min = max(4, int(block_size_range[0]))
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    block_size = random.randint(bs_min, bs_max) if bs_min < bs_max else bs_min
    block_size = max(1, block_size)
    output = frame.copy(); font_scale = max(0.1, block_size / 18.0) # Adjusted scale
    thickness = 1; chars = list(char_set); 
    if invert: chars = chars[::-1]
    num_chars = len(chars); gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for y_coord in range(0, height - block_size + 1, block_size):
        for x_coord in range(0, width - block_size + 1, block_size):
            block = gray[y_coord:y_coord+block_size, x_coord:x_coord+block_size]
            avg_brightness = np.mean(block)
            char_index = int(np.clip((avg_brightness / 255.0) * num_chars, 0, num_chars - 1))
            char_to_draw = chars[char_index]
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255) 
            text_size_cv, _ = cv2.getTextSize(char_to_draw, FONT_FACE_TO_USE, font_scale, thickness)
            pos = (x_coord + (block_size - text_size_cv[0]) // 2, y_coord + (block_size + text_size_cv[1]) // 2)
            try: cv2.putText(output, char_to_draw, pos, FONT_FACE_TO_USE, font_scale, text_color, thickness, cv2.LINE_AA)
            except: pass 
    return output

def apply_sensor_overload(frame, intensity): 
    output = frame.copy(); max_shift = int(8 + 30 * intensity) # Reduced shift
    temp_aberration = output.copy() 
    for i in range(3): 
        shift_x = random.randint(-max_shift, max_shift); shift_y = random.randint(-max_shift, max_shift)
        channel = output[:,:,i]; shifted = np.roll(np.roll(channel, shift_y, axis=0), shift_x, axis=1)
        temp_aberration[:,:,i] = cv2.addWeighted(channel, 0.4, shifted, 0.6,0) # Adjusted blend
    output = temp_aberration
    if random.random() < 0.4 * intensity: # Reduced chance
        alpha_contrast = 1.5 + 2.5 * intensity; beta_brightness = random.randint(-40, 40) * intensity 
        output_contrasted = cv2.convertScaleAbs(output, alpha=alpha_contrast, beta=beta_brightness)
        bright_mask_gray = cv2.cvtColor(output_contrasted, cv2.COLOR_BGR2GRAY)
        _, bright_mask_thresh = cv2.threshold(bright_mask_gray, 235, 255, cv2.THRESH_BINARY) # Higher threshold
        sigma_val = max(1, int(8 + 15 * intensity)); bloom_effect = cv2.GaussianBlur(output_contrasted, (0,0), sigmaX=sigma_val, sigmaY=sigma_val)
        bloom_masked = cv2.bitwise_and(bloom_effect, bloom_effect, mask=bright_mask_thresh)
        output = cv2.addWeighted(output_contrasted, 1.0, bloom_masked, 0.4 * intensity, 0)
    if random.random() < 0.15 * intensity: # Reduced chance
        overlay_static = np.zeros_like(output, dtype=np.uint8); num_dots = random.randint(1, int(3 + 15 * intensity)) 
        for _ in range(num_dots):
            dot_color_val = random.randint(10,30)
            cv2.circle(overlay_static, (random.randint(0,WIDTH-1), random.randint(0,HEIGHT-1)), random.randint(1,2), (dot_color_val,)*3, -1)
        if random.random() < 0.08 * intensity: 
            cv2.line(overlay_static, (random.randint(0,WIDTH-1), random.randint(0,HEIGHT-1)), (random.randint(0,WIDTH-1), random.randint(0,HEIGHT-1)), (random.randint(5,15),)*3, 1)
        output = cv2.addWeighted(output, 0.9, overlay_static, 0.25 + 0.2*intensity ,0) 
    return output

def apply_data_bleed_sim(frame, intensity):
    output = frame.copy(); num_bleeds = random.randint(3, int(8 + 15 * intensity)) # Reduced bleeds
    for _ in range(num_bleeds):
        bh_src = random.randint(int(HEIGHT*0.08), int(HEIGHT*0.25)); bw_src = random.randint(int(WIDTH*0.08), int(WIDTH*0.25))
        if HEIGHT - bh_src <=0 or WIDTH - bw_src <=0 : continue
        y_src = random.randint(0, HEIGHT - bh_src -1); x_src = random.randint(0, WIDTH - bw_src -1)
        source_block = frame[y_src:y_src+bh_src, x_src:x_src+bw_src]
        if source_block.size == 0: continue
        avg_color = np.mean(source_block, axis=(0,1)).astype(np.uint8) 
        bleed_strength = 0.1 + 0.4 * intensity * random.random() 
        bleed_radius_x_factor = random.uniform(0.6, 2.0); bleed_radius_y_factor = random.uniform(0.6, 2.0)
        bleed_h = int(bh_src * bleed_radius_y_factor); bleed_w = int(bw_src * bleed_radius_x_factor)
        offset_x = random.randint(-bw_src // 3, bw_src // 3); offset_y = random.randint(-bh_src // 3, bh_src // 3)
        tx = np.clip(x_src + offset_x, 0, WIDTH - bleed_w); ty = np.clip(y_src + offset_y, 0, HEIGHT - bleed_h)
        if bleed_w <=0 or bleed_h <=0: continue
        target_area = output[ty:ty+bleed_h, tx:tx+bleed_w]
        if target_area.size == 0: continue
        color_overlay = np.full_like(target_area, avg_color)
        output[ty:ty+bleed_h, tx:tx+bleed_w] = cv2.addWeighted(target_area, 1.0 - bleed_strength, color_overlay, bleed_strength, 0)
    return output

# ... (Other visual effects like flickering, figurative_noise, crt_ghost, slit_scan, vector_field are relatively less intensive or wrappers)

def apply_flickering_glitch_layer(frame, intensity, effect_func, prev_frame=None, **kwargs_from_main_loop):
    if random.random() > (0.25 + 0.5 * intensity): return frame # Slightly reduced chance
    glitch_layer_input = frame.copy(); params_to_pass_inner = {}; sig_inner = inspect.signature(effect_func)
    if 'prev_frame' in sig_inner.parameters: params_to_pass_inner['prev_frame'] = prev_frame if prev_frame is not None else frame
    if 'intensity' in sig_inner.parameters: params_to_pass_inner['intensity'] = intensity
    for p_name_sig_inner in sig_inner.parameters:
        if p_name_sig_inner.endswith("_range"):
            key_base_candidate = p_name_sig_inner[:-6]
            if key_base_candidate in kwargs_from_main_loop and isinstance(kwargs_from_main_loop[key_base_candidate], tuple):
                 params_to_pass_inner[p_name_sig_inner] = kwargs_from_main_loop[key_base_candidate]
    glitched_part = None
    try:
        glitched_part = effect_func(glitch_layer_input, **params_to_pass_inner) if params_to_pass_inner else effect_func(glitch_layer_input, prev_frame=params_to_pass_inner.get('prev_frame'), intensity=params_to_pass_inner.get('intensity'))
    except: 
        try: # Fallback
            minimal_params = {k:v for k,v in [('prev_frame',prev_frame if prev_frame is not None else frame),('intensity',intensity)] if k in sig_inner.parameters}
            glitched_part = effect_func(glitch_layer_input, **minimal_params)
        except: glitched_part = glitch_layer_input 
    if glitched_part is None: glitched_part = glitch_layer_input
    mask_type = random.choice(["stripes", "blocks", "noise"]); flicker_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    if mask_type == "stripes":
        num_stripes = random.randint(3, 20); stripe_h = HEIGHT // num_stripes if num_stripes > 0 else HEIGHT
        for i in range(num_stripes):
            if random.random() < 0.5: flicker_mask[i*stripe_h : (i+1)*stripe_h, :] = 255
    elif mask_type == "blocks":
        num_blocks_w = random.randint(3, 15); num_blocks_h = random.randint(3, 15)
        block_w = WIDTH // num_blocks_w if num_blocks_w > 0 else WIDTH; block_h = HEIGHT // num_blocks_h if num_blocks_h > 0 else HEIGHT
        for r_idx in range(num_blocks_h):
            for c_idx in range(num_blocks_w):
                if random.random() < 0.25: flicker_mask[r_idx*block_h:(r_idx+1)*block_h, c_idx*block_w:(c_idx+1)*block_w] = 255
    else: flicker_mask = (np.random.randint(0, 2, size=(HEIGHT, WIDTH), dtype=np.uint8) * 255)
    mask_inv = cv2.bitwise_not(flicker_mask)
    fg = cv2.bitwise_and(glitched_part, glitched_part, mask=flicker_mask); bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return cv2.add(fg, bg)

def apply_figurative_noise_sim(frame, intensity): 
    return apply_perlin_noise(frame, alpha_range=(0.05 * intensity, 0.25 * intensity), scale_range=(70, 180), oct_range=(2,4)) 
def apply_crt_ghost_sim(frame, prev_frame): 
    if prev_frame is not None: return cv2.addWeighted(frame, 0.88, prev_frame, 0.12, 0) 
    return frame
def apply_slit_scan_sim(frame): 
    rows, cols, _ = frame.shape; output = frame.copy(); slit_pos_y = rows // 2 
    for r_idx in range(rows):
         dist_from_slit = abs(r_idx - slit_pos_y)
         max_smear = (cols / 12) * (1 - dist_from_slit / (rows/2)) * abs(math.sin(r_idx * 0.06 + random.uniform(0,math.pi)))
         smear_amount = int(random.uniform(-max(0,max_smear), max(0,max_smear)) * 0.4) 
         output[r_idx, :] = np.roll(output[r_idx, :], smear_amount, axis=0) 
    return output
def apply_vector_field_sim(frame): 
    return apply_warp(frame, amplitude_range=(2, 6 + 4*random.random()), freq_range=(0.07, 0.25 + 0.1*random.random())) 

# --- Audio Effects (some parameters tweaked for performance) ---
def apply_distortion(data, intensity_range=(1.2, 4.0)): # Reduced range
    intensity = random.uniform(*intensity_range)
    return np.clip(data * intensity, -0.98, 0.98) 

def apply_bitcrush(data, bit_depth_range=(5, 13)): # Slightly higher min bit depth
    bd_min = max(2, int(bit_depth_range[0])); bd_max = max(bd_min + 1, int(bit_depth_range[1]))
    bits = random.randint(bd_min, bd_max) if bd_min < bd_max else bd_min
    if bits >= 16: return data 
    steps = 2**(bits -1); return np.round(data * steps) / steps

def apply_spectral_glitch(data, intensity=0.5): 
    n_samples = len(data)
    if n_samples < 512: return data # Reduced min samples
    data_float = data.astype(np.float32); spectrum = fft(data_float)
    magnitude = np.abs(spectrum); phase = np.angle(spectrum); n_freqs = len(spectrum)
    num_glitches = int(n_freqs * 0.005 * intensity * random.uniform(0.5, 1.5)) # Fewer glitches
    positive_freq_bins = n_freqs // 2
    if positive_freq_bins <=1 : return data 
    indices_to_glitch = random.sample(range(1, positive_freq_bins), min(num_glitches, positive_freq_bins -1 ))
    for idx in indices_to_glitch:
        if random.random() < 0.6: 
             attenuation_factor = random.uniform(0.0, 0.2) 
             magnitude[idx] *= attenuation_factor
             if idx < n_freqs: magnitude[n_freqs - idx] *= attenuation_factor 
        else: 
             if idx + 1 < positive_freq_bins:
                  phase[idx], phase[idx+1] = phase[idx+1], phase[idx]
                  if n_freqs - idx < n_freqs: phase[n_freqs - idx] = -phase[idx] 
                  if n_freqs - (idx+1) < n_freqs : phase[n_freqs - (idx+1)] = -phase[idx+1]
    new_spectrum = magnitude * np.exp(1j * phase); glitched_data = np.real(ifft(new_spectrum))
    max_abs_orig = np.max(np.abs(data_float)); max_abs_glitched = np.max(np.abs(glitched_data))
    if max_abs_glitched > 1e-6 and max_abs_orig > 1e-6: glitched_data *= (max_abs_orig / max_abs_glitched)
    elif max_abs_glitched > 1e-6: glitched_data /= max_abs_glitched
    return glitched_data.astype(data.dtype)

def apply_convolution_reverb(data, impulse_response): # This can be slow
    n_data = len(data); n_ir = len(impulse_response)
    if n_ir == 0 or n_data == 0: return data
    # Further shorten IR for performance if it's still too long
    max_ir_len = int(SAMPLE_RATE * 0.1) # Max 100ms IR for faster convolution
    if n_ir > max_ir_len: impulse_response = impulse_response[:max_ir_len]; n_ir = len(impulse_response)
    if n_ir == 0: return data
    if n_ir >= n_data: impulse_response = impulse_response[:max(1,n_data//3)]; n_ir = len(impulse_response)
    if n_ir == 0 : return data 
    reverbed_data = np.convolve(data, impulse_response, mode='same') 
    max_abs_orig = np.max(np.abs(data)); max_abs_rev = np.max(np.abs(reverbed_data))
    if max_abs_rev > 1e-6 and max_abs_orig > 1e-6: reverbed_data *= (max_abs_orig / max_abs_rev)
    elif max_abs_rev > 1e-6: reverbed_data /= max_abs_rev
    return reverbed_data

IMPULSE_RESPONSES = { # Shorter IRs for performance
    "damp": np.exp(-np.linspace(0, 8, int(SAMPLE_RATE * 0.1))) * np.random.randn(int(SAMPLE_RATE * 0.1)),
    "metal_hit": np.sin(2*np.pi*1500*np.linspace(0,0.05,int(SAMPLE_RATE*0.05))) * np.exp(-np.linspace(0, 15, int(SAMPLE_RATE * 0.05))),
    "noise_burst": np.random.randn(int(SAMPLE_RATE * 0.05)) * np.exp(-np.linspace(0, 4, int(SAMPLE_RATE * 0.05))) 
}
for k_ir in IMPULSE_RESPONSES: # Normalize
    max_abs_ir = np.max(np.abs(IMPULSE_RESPONSES[k_ir])); 
    if max_abs_ir > 1e-6: IMPULSE_RESPONSES[k_ir] /= max_abs_ir

def apply_broken_speaker_sim(data, intensity): 
    output = data.copy().astype(np.float32) # Ensure float for FFT
    if random.random() < 0.6 * intensity: # Reduced chance
        spectrum = fft(output); n = len(spectrum); positive_freq_bins = n // 2
        if positive_freq_bins > 10: 
            if random.random() < 0.5: 
                cut_freq_bin = int((random.uniform(700, 2500) / (SAMPLE_RATE / 2)) * positive_freq_bins)
                if random.random() < 0.5: spectrum[cut_freq_bin:positive_freq_bins] = 0; spectrum[n-positive_freq_bins+1 : n-cut_freq_bin+1] = 0
                else: spectrum[:cut_freq_bin] = 0; spectrum[n-cut_freq_bin+1:] = 0 
            else: 
                scoop_center_hz = random.uniform(1200, 5000); scoop_width_hz = scoop_center_hz * random.uniform(0.1, 0.4)
                scoop_start_hz = max(1, scoop_center_hz - scoop_width_hz / 2); scoop_end_hz = scoop_center_hz + scoop_width_hz / 2
                scoop_start_bin = int((scoop_start_hz / (SAMPLE_RATE/2)) * positive_freq_bins)
                scoop_end_bin = int(min(scoop_end_hz, SAMPLE_RATE/2-1) / (SAMPLE_RATE/2) * positive_freq_bins) # Ensure end_hz is valid
                if scoop_start_bin < scoop_end_bin :
                    spectrum[scoop_start_bin:scoop_end_bin] = 0; spectrum[n-scoop_end_bin+1 : n-scoop_start_bin+1] = 0 
            output = np.real(ifft(spectrum))
    output = apply_distortion(output, intensity_range=(4.0 + 8 * intensity, 12.0 + 15 * intensity)) # Tweaked range
    if random.random() < 0.4 * intensity: # Reduced chance/count for crackles
        num_crackles = random.randint(5, int(25 + 50 * intensity)) 
        for _ in range(num_crackles):
            if len(output) == 0: break
            pos = random.randint(0, len(output)-1); max_crackle_len = min(30, len(output)-pos) # Shorter crackles
            if max_crackle_len <=0 : continue
            crackle_len = random.randint(1, max_crackle_len)
            crackle_noise_1d = (np.random.rand(crackle_len) - 0.5) * 0.4 * intensity # Reduced amplitude
            # Ensure output slice and noise are 1D for addition
            output_slice = output[pos:pos+crackle_len]
            if output_slice.ndim == 1:
                 output[pos:pos+crackle_len] += crackle_noise_1d
            elif output_slice.ndim == 2 and output_slice.shape[1] == 1: # Column vector
                 output[pos:pos+crackle_len] += crackle_noise_1d[:, np.newaxis]
    return np.clip(output, -1.0, 1.0) 

# --- Tone, Noise, Melody, Rhythm Generation (some parameters tweaked for performance) ---
def generate_tone(freq, duration_samples, vol, fm_chance=0.4, harmonic_chance=0.3): # Reduced chances
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
    wave = np.zeros(duration_samples, dtype=np.float32)
    if random.random() < fm_chance: 
        mod_freq = freq * random.uniform(0.2, 4.0); mod_index = random.uniform(1.0, 6.0) * 5.0 
        if abs(freq * t[-1]) < 1e8 and abs(mod_freq * t[-1]) < 1e8: # Reduced threshold
            wave = np.sin(2 * np.pi * freq * t + mod_index * np.sin(2 * np.pi * mod_freq * t))
        else: wave = np.sin(2 * np.pi * freq * t)
    elif random.random() < harmonic_chance: 
        harmonic_count = random.randint(1, 3); wave = np.sin(2 * np.pi * freq * t) 
        for h_idx in range(2, harmonic_count + 2): 
            wave += (random.uniform(0.1, 0.4) / h_idx) * np.sin(2 * np.pi * freq * h_idx * t + random.uniform(0, np.pi))
        max_abs_wave = np.max(np.abs(wave)); 
        if max_abs_wave > 1e-6: wave /= max_abs_wave 
    else: wave = np.sin(2 * np.pi * freq * t)
    wave *= vol 
    attack_len = min(int(SAMPLE_RATE*0.003), duration_samples//4); decay_len = min(int(SAMPLE_RATE*random.uniform(0.03,0.2)), max(0,duration_samples-attack_len))
    sustain_level = random.uniform(0.2,0.6); envelope = np.ones(duration_samples)
    if attack_len > 0: envelope[:attack_len] = np.linspace(0,1,attack_len)
    sustain_start_idx = attack_len + decay_len
    if decay_len > 0 and sustain_start_idx <= duration_samples:
        envelope[attack_len:sustain_start_idx] = np.linspace(1,sustain_level,decay_len)
        if sustain_start_idx < duration_samples: envelope[sustain_start_idx:] = sustain_level 
    elif attack_len < duration_samples : envelope[attack_len:] = sustain_level
    fade_out_len = min(int(SAMPLE_RATE*0.008), duration_samples//5)
    if fade_out_len > 0 and duration_samples > fade_out_len:
        start_level = envelope[duration_samples-fade_out_len-1] if duration_samples-fade_out_len-1 >=0 else sustain_level
        envelope[duration_samples-fade_out_len:] = np.linspace(start_level,0,fade_out_len)
    return wave * envelope

def generate_noise(noise_type, duration_samples, vol, features=[]):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    out_noise = np.zeros(duration_samples, dtype=np.float32)
    if noise_type == "white": out_noise = (np.random.rand(duration_samples) - 0.5) * 2.0
    elif noise_type == "pink": 
        # Simplified pink noise for speed, less accurate
        white = (np.random.rand(duration_samples) - 0.5) * 2.0
        out_noise = np.cumsum(white) # Brownian motion approximation
        out_noise -= np.mean(out_noise) # Remove DC offset
        max_abs = np.max(np.abs(out_noise))
        if max_abs > 1e-6: out_noise /= max_abs
    elif noise_type == "brown": 
        white = (np.random.rand(duration_samples) - 0.5) * 0.2; out_noise = np.cumsum(white)
        max_abs = np.max(np.abs(out_noise)); 
        if max_abs > 1e-6: out_noise /= max_abs 
    elif noise_type == "glitch": 
        num_glitches = random.randint(3, int(duration_samples/(SAMPLE_RATE*0.02))) # Reduced glitches
        for _ in range(num_glitches):
            glitch_len = random.randint(1,max(1,int(SAMPLE_RATE*0.003))) 
            if duration_samples < glitch_len+5: continue
            start = random.randint(0,duration_samples-glitch_len-1)
            out_noise[start:start+glitch_len] = (np.random.rand(glitch_len)-0.5)*random.uniform(0.4,1.8) 
    elif noise_type == "data_stream_sim":
        base_noise = generate_noise("pink",duration_samples,0.4,[]); clicks = generate_noise("glitch",duration_samples,0.4,[])
        mod = 0.5 + 0.5*np.sin(2*np.pi*random.uniform(0.8,4.0)*np.linspace(0,duration_samples*time_step,duration_samples))
        out_noise = base_noise + clicks * mod
    elif noise_type == "white_bursts":
        num_bursts = random.randint(1, max(1,int(duration_samples/(SAMPLE_RATE*0.08)))) # Fewer bursts
        for _ in range(num_bursts):
            burst_len = random.randint(int(SAMPLE_RATE*0.001),int(SAMPLE_RATE*0.03)) 
            if duration_samples < burst_len+5: continue
            start = random.randint(0,duration_samples-burst_len-1)
            out_noise[start:start+burst_len] = (np.random.rand(burst_len)-0.5)*1.8 
    elif noise_type == "static": out_noise = (np.random.rand(duration_samples) - 0.5) * 1.8
    else: out_noise = (np.random.rand(duration_samples) - 0.5) * 2.0 
    final_noise = out_noise * vol
    max_abs = np.max(np.abs(final_noise)); 
    if max_abs > 1.0: final_noise /= max_abs
    return final_noise

def generate_biometric_rhythm(duration_samples, vol, rhythm_type="heartbeat_distorted"):
    noise_out = np.zeros(duration_samples, dtype=np.float32)
    if duration_samples <=0: return noise_out
    if "heartbeat" in rhythm_type:
        bpm = random.uniform(35, 100) * random.uniform(0.8, 1.2); bps = bpm / 60.0
        if bps <= 1e-3: return noise_out 
        beat_interval_samples_base = int(SAMPLE_RATE / bps)
        if beat_interval_samples_base <= 0: return noise_out
        thud1_len = int(SAMPLE_RATE*random.uniform(0.04,0.08)); thud1_freq = random.uniform(35,70)
        t1 = np.linspace(0,thud1_len*time_step,thud1_len,endpoint=False); env1 = np.exp(-np.linspace(0,random.uniform(7,15),thud1_len))
        s1 = np.sin(2*np.pi*thud1_freq*t1)*env1
        thud2_len = int(SAMPLE_RATE*random.uniform(0.02,0.06)); thud2_freq = random.uniform(45,80)
        t2 = np.linspace(0,thud2_len*time_step,thud2_len,endpoint=False); env2 = np.exp(-np.linspace(0,random.uniform(9,18),thud2_len))
        s2 = np.sin(2*np.pi*thud2_freq*t2)*env2*0.6
        single_beat = np.zeros(thud1_len+thud2_len+int(SAMPLE_RATE*0.015)); single_beat[:thud1_len]=s1
        single_beat[thud1_len+int(SAMPLE_RATE*0.01):thud1_len+int(SAMPLE_RATE*0.01)+thud2_len]=s2
        single_beat = apply_distortion(single_beat,(1.2,3.0)) 
        current_sample = 0
        while current_sample < duration_samples:
            interval = int(beat_interval_samples_base*random.uniform(0.95,1.05))
            if interval <=0: interval = beat_interval_samples_base
            actual_len = min(len(single_beat),duration_samples-current_sample)
            if actual_len <=0 : break
            noise_out[current_sample:current_sample+actual_len]+=single_beat[:actual_len]
            current_sample += interval
    elif "breathing" in rhythm_type:
        cycle_sec = random.uniform(3.5,6); cycle_samples = int(cycle_sec*SAMPLE_RATE)
        current_sample = 0
        while current_sample < duration_samples:
            actual_len = min(cycle_samples,duration_samples-current_sample)
            if actual_len <=0 : break
            raw_noise = generate_noise("pink",actual_len,1.0,[]) 
            t_cyc = np.linspace(0,1,actual_len)
            env_in = np.sin(np.pi*t_cyc[:actual_len//2])**1.5; env_out = np.cos(np.pi*t_cyc[actual_len//2:])**1.5
            breath_env = np.concatenate((env_in,env_out)); 
            if len(breath_env) < actual_len: breath_env = np.pad(breath_env,(0,actual_len-len(breath_env)),'edge')
            noise_out[current_sample:current_sample+actual_len]+=raw_noise*breath_env
            current_sample += actual_len
    final_rhythm = noise_out * vol; max_abs = np.max(np.abs(final_rhythm))
    if max_abs > 1e-6 : final_rhythm /= max_abs 
    return final_rhythm

def generate_melody_fragment(duration_samples, vol, theme_data, intensity):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    scale_name = theme_data.get("melody_scale", "minor_pentatonic"); scale_intervals = SCALES.get(scale_name, SCALES["minor_pentatonic"])
    base_freq = random.uniform(70, 550) 
    notes_in_fragment = random.randint(max(1,int(1+intensity*0.5)), int(4+8*intensity)) # Adjusted count
    notes_in_fragment = max(1, notes_in_fragment) 
    avg_note_duration = duration_samples // notes_in_fragment
    if avg_note_duration <=0 : avg_note_duration = max(1,int(SAMPLE_RATE*0.04)) 
    melody = np.zeros(duration_samples,dtype=np.float32); current_pos = 0
    for _ in range(notes_in_fragment):
        if current_pos >= duration_samples: break
        scale_deg = random.choice(scale_intervals); oct_shift = random.choice([-1,0,0,1])
        f_start = base_freq*(2**((scale_deg+oct_shift*12)/12.0))
        bend = random.uniform(-4,4)*intensity; f_end = f_start*(2**(bend/12.0))
        f_start = np.clip(f_start,LOW_FREQ_CUTOFF,HIGH_FREQ_CUTOFF); f_end = np.clip(f_end,LOW_FREQ_CUTOFF,HIGH_FREQ_CUTOFF)
        note_len = int(avg_note_duration*random.uniform(0.6,1.2)); note_len = min(note_len,duration_samples-current_pos); note_len=max(1,note_len)
        if note_len <= 0: break
        t_note = np.linspace(0,note_len*time_step,note_len,endpoint=False); freqs = np.linspace(f_start,f_end,note_len)
        phase = np.cumsum(2*np.pi*freqs*time_step); note_tone = np.sin(phase)
        att=min(int(note_len*0.08),note_len); dec=min(int(note_len*0.3),note_len-att); sus=random.uniform(0.1,0.7)
        env=np.ones(note_len); 
        if att>0: env[:att]=np.linspace(0,1,att)
        if dec>0 and att+dec<=note_len: env[att:att+dec]=np.linspace(1,sus,dec); 
        if att+dec<note_len: env[att+dec:]=sus
        elif att<note_len: env[att:]=sus
        fade=min(int(note_len*0.08),note_len//3); 
        if fade>0: start_lvl=env[note_len-fade-1] if note_len-fade-1>=0 else sus; env[note_len-fade:]=np.linspace(start_lvl,0,fade)
        melody[current_pos:current_pos+note_len]+=note_tone*env; current_pos+=note_len
    if random.random() < theme_data.get("melody_counter_melody_chance",0.3)*intensity*0.7: # Reduced chance for counter
        # Simplified counter melody for performance
        pass 
    dist_amt = theme_data.get("melody_distortion",10.0)*(0.4+intensity*0.4)
    melody = apply_distortion(melody,intensity_range=(max(1.1,dist_amt*0.6),max(1.5,dist_amt*1.3)))
    if random.random()<0.5*intensity: melody = apply_bitcrush(melody,bit_depth_range=(max(2,int(7-intensity*1.5)),max(3,int(9-intensity*1.5)))) 
    if random.random()<0.25*intensity: melody = apply_spectral_glitch(melody,intensity=intensity*0.7)
    final_melody = melody*vol; max_abs = np.max(np.abs(final_melody)); 
    if max_abs > 1.0: final_melody /= max_abs 
    return final_melody.flatten() # Ensure 1D output

# ... (generate_frames_enhanced - minor tweaks for performance might be needed in text counts, ARG frequencies)
def generate_frames_enhanced(theme_data, intensity_func, global_params):
    DURATION = random.randint(MIN_DURATION, MAX_DURATION)
    FPS = random.randint(MIN_FPS, MAX_FPS)
    FRAME_COUNT = DURATION * FPS
    current_visual_params = copy.deepcopy(global_params["visual"])
    instability_chance = global_params["instability_chance"] * 1.2 # Reduced base instability
    print(f"Generating {DURATION}s video ({FPS} FPS), Theme: {global_params['theme_name']}...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v"); video_temp_path = f"video_temp_{int(time.time())}.mp4" # Simpler temp name
    video = cv2.VideoWriter(video_temp_path, fourcc, FPS, (WIDTH, HEIGHT))
    prev_frame = None; frame_hold_counter = 0; held_frame = None; effect_history = []
    current_theme_mutated = copy.deepcopy(theme_data)
    mutation_timer = 0; mutation_interval = FPS * random.randint(7,18) # Wider mutation interval
    arg_qr_img = None; arg_qr_display_frames = 0; arg_text_code = None; arg_text_display_frames = 0
    persistent_texts = []
    num_persistent = random.randint(5, 15) # Reduced persistent texts
    for _ in range(num_persistent):
        word = str(random.choice(current_theme_mutated["words"])); symbol = str(random.choice(current_theme_mutated["symbols"]))
        if random.random() < current_theme_mutated.get("zalgo_words_chance",0.3): word = zalgo_text(word)
        if random.random() < 0.08: word = base64.b64encode(word.encode('utf-8','ignore')).decode()
        elif random.random() < 0.04: word = binascii.hexlify(word.encode('utf-8','ignore')).decode()
        text_content = word + symbol; chosen_palette = random.choice(current_theme_mutated["colors"])
        text_color_bgr = tuple(c for c in random.choice(chosen_palette)[:3])
        persistent_texts.append({"text":text_content, "pos":(random.randint(-WIDTH//5,WIDTH),random.randint(-HEIGHT//5,HEIGHT)), 
            "font_size":random.uniform(0.3,4.0),"color":text_color_bgr, "lifetime":random.randint(int(FPS*0.4),int(FPS*12)), 
            "frame_count":0, "move_speed":(random.uniform(-7,7),random.uniform(-5,5)), "alpha":random.uniform(0.4,1.0)})

    for i in range(FRAME_COUNT):
        current_time_sec = i / FPS; current_intensity = np.clip(intensity_func(current_time_sec),0.01,4.0)
        mutation_timer += 1
        if mutation_timer >= mutation_interval: 
            mutation_timer = 0
            if random.random() < 0.25 * current_intensity: # Reduced mutation chance
                try: # Simplified color mutation
                    palette_idx=random.randrange(len(current_theme_mutated["colors"])); color_idx=random.randrange(len(current_theme_mutated["colors"][palette_idx]))
                    original_color=current_theme_mutated["colors"][palette_idx][color_idx]
                    mut_color=tuple(np.clip(c+random.randint(-30,30),0,255) for c in original_color[:3])
                    current_theme_mutated["colors"][palette_idx][color_idx] = (*mut_color, original_color[3]) if len(original_color)==4 else mut_color
                except: pass 
        if random.random() < instability_chance * current_intensity * 0.8: # Reduced instability effect
            # Simplified instability logic for brevity and potential speed up
            param_key_mutate = random.choice(list(current_visual_params.keys()))
            if param_key_mutate == "effect_probabilities":
                effect_name_mutate = random.choice(list(current_visual_params["effect_probabilities"].keys()))
                current_visual_params["effect_probabilities"][effect_name_mutate] = np.clip(current_visual_params["effect_probabilities"][effect_name_mutate] + random.uniform(-0.15,0.15), 0.01, 0.95)
            elif isinstance(current_visual_params[param_key_mutate], tuple) and len(current_visual_params[param_key_mutate])==2:
                c_min,c_max=current_visual_params[param_key_mutate]; range_w=abs(c_max-c_min)+1e-6
                n_mid=(c_min+c_max)/2+random.uniform(-0.1,0.1)*range_w; n_hw=range_w/2*random.uniform(0.8,1.2)
                n_min,n_max = n_mid-n_hw, n_mid+n_hw
                if "alpha" in param_key_mutate: n_min,n_max = max(0.01,n_min),min(1.0,max(n_min+0.05,n_max))
                elif any(k in param_key_mutate for k in ["shift","block","size","oct","num","thick"]): n_min,n_max=int(max(1,n_min)),int(max(n_min+1,n_max))
                else: n_max = max(n_min+0.01, n_max) if n_min>=n_max else n_max
                current_visual_params[param_key_mutate] = (n_min,n_max)

        if frame_hold_counter > 0:
            if held_frame is not None: video.write(held_frame); frame_hold_counter -=1; continue
            else: frame_hold_counter = 0 
        if random.random() < (0.015 + 0.025 * current_intensity) and frame_hold_counter == 0: # Reduced hold chance
            frame_hold_counter = random.randint(1, int(FPS * (0.25 + current_intensity * 0.6))); held_frame = None 
        
        base_roll = random.random() # Base frame generation
        if base_roll < 0.15: frame = np.full((HEIGHT,WIDTH,3),random.choice(random.choice(current_theme_mutated["colors"]))[:3],dtype=np.uint8)
        elif base_roll < 0.3: 
            c1,c2=random.choice(random.choice(current_theme_mutated["colors"]))[:3],random.choice(random.choice(current_theme_mutated["colors"]))[:3]
            frame=np.array([[int(c1[ch]*(1-r/HEIGHT)+c2[ch]*(r/HEIGHT)) for ch in range(3)] for r in range(HEIGHT)],dtype=np.uint8)[:,np.newaxis,:]
            frame=np.repeat(frame,WIDTH,axis=1)
            if random.random()<0.4: frame=cv2.rotate(frame,random.choice([0,1,2])) # cv2.ROTATE_ constants
        else: frame=np.random.randint(0,max(1,int(25+50*current_intensity)),(HEIGHT,WIDTH,3),dtype=np.uint8)

        is_catalyst = current_intensity > current_visual_params.get("catalyst_threshold",2.8) and random.random()<(0.1+0.12*current_intensity)
        
        available_fx = [globals()[name] for name in current_theme_mutated.get("visual_effects",[]) if name in globals() and callable(globals()[name])]
        num_fx_apply = min(len(available_fx), (random.randint(1,2) if not is_catalyst else random.randint(3,5)) + int(current_intensity*1.2))
        applied_count=0; random.shuffle(available_fx); applied_names_this_frame=[]

        for fx_func in available_fx: # Apply visual effects
            if applied_count >= num_fx_apply: break
            fx_name = fx_func.__name__
            if effect_history.count(fx_name)>1 and random.random()<0.65 and not is_catalyst: continue # Reduced repetition avoidance strictness
            prob = current_visual_params.get("effect_probabilities",{}).get(fx_name,0.45) * (0.6+current_intensity*0.4)
            if random.random() < np.clip(prob,0.03,0.92) or is_catalyst:
                try:
                    sig=inspect.signature(fx_func); params_pass={}; 
                    if 'prev_frame' in sig.parameters: params_pass['prev_frame']=prev_frame
                    if 'intensity' in sig.parameters: params_pass['intensity']=current_intensity
                    for p_sig in sig.parameters: # Simplified param range passing
                        if p_sig.endswith("_range") and p_sig[:-6] in current_visual_params:
                            params_pass[p_sig] = current_visual_params[p_sig[:-6]]
                    if fx_func == apply_flickering_glitch_layer:
                        inner_fx_opts = [f for f in available_fx if f!=apply_flickering_glitch_layer]
                        if inner_fx_opts: frame=fx_func(frame,intensity,random.choice(inner_fx_opts),prev_frame,**current_visual_params)
                    else: frame = fx_func(frame,**params_pass)
                    applied_count+=1; applied_names_this_frame.append(fx_name)
                except: pass # Silently skip effect on error for speed
        effect_history.extend(applied_names_this_frame); effect_history=effect_history[-15:] # Shorter history

        # ARG Visuals (reduced frequency)
        if arg_qr_display_frames > 0:
            if arg_qr_img is not None and arg_qr_img.size > 0 :
                qr_x,qr_y=random.randint(0,max(0,WIDTH-ARG_QR_SIZE)),random.randint(0,max(0,HEIGHT-ARG_QR_SIZE))
                temp_qr = arg_qr_img.copy()
                if random.random()<0.25*current_intensity: temp_qr=apply_pixelation(temp_qr,block_size_range=(max(2,int(4-current_intensity)),max(3,int(12-current_intensity*1.5))))
                if qr_y+ARG_QR_SIZE<=HEIGHT and qr_x+ARG_QR_SIZE<=WIDTH:
                    roi=frame[qr_y:qr_y+ARG_QR_SIZE,qr_x:qr_x+ARG_QR_SIZE]
                    if roi.shape==temp_qr.shape: frame[qr_y:qr_y+ARG_QR_SIZE,qr_x:qr_x+ARG_QR_SIZE]=cv2.addWeighted(roi,1.0-0.8*np.clip(arg_qr_display_frames/(FPS*0.4),0.1,1.0),temp_qr,0.8*np.clip(arg_qr_display_frames/(FPS*0.4),0.1,1.0),0)
            arg_qr_display_frames-=1
        elif random.random() < (0.002 + 0.005 * current_intensity) * global_params.get("arg_qr_prob_modifier",0.8): # Reduced QR chance
            qr_data=current_theme_mutated.get("arg_elements",{}).get("qr_data_prefix","QR_")+"".join(random.choices(string.ascii_uppercase+string.digits,k=random.randint(6,12)))
            arg_qr_img=generate_qr_code_image(qr_data,ARG_QR_SIZE); 
            if arg_qr_img.size>0: arg_qr_display_frames=int(FPS*random.uniform(0.6,2.0))
        
        if arg_text_display_frames > 0: # ARG Text
            if arg_text_code:
                scl=ARG_TEXT_FONT_SCALE_BASE*(0.7+current_intensity*0.35); (tw,_),_=cv2.getTextSize(arg_text_code,FONT_FACE_TO_USE,scl,2)
                tx,ty=random.randint(5,max(6,WIDTH-tw-15)),random.randint(25,max(26,HEIGHT-15))
                alpha_blend=np.clip(arg_text_display_frames/(FPS*0.25),0.2,1.0)
                overlay=frame.copy(); cv2.putText(overlay,arg_text_code,(tx,ty),FONT_FACE_TO_USE,scl,(200,220,255),random.randint(1,2),cv2.LINE_AA)
                frame=cv2.addWeighted(frame,1.0-alpha_blend*0.6,overlay,alpha_blend*0.6,0)
            arg_text_display_frames-=1
        elif random.random() < (0.004 + 0.01 * current_intensity) * global_params.get("arg_text_prob_modifier",0.8): # Reduced text chance
            arg_text_code=generate_arg_text_code(current_theme_mutated,length=random.randint(15,40))
            arg_text_display_frames=int(FPS*random.uniform(0.4,1.8))

        if random.random() < (0.008 + 0.025 * current_intensity) * global_params.get("arg_protocol_prob_modifier",0.8): # Reduced protocol chance
            el=current_theme_mutated.get("arg_elements",{}); errs=el.get("fake_errors",[]); keys=el.get("cipher_keys",[]); proto_txt=""
            if errs and (random.random()<0.65 or not keys): proto_txt=random.choice(errs)
            elif keys: proto_txt="KEY_FRAG: "+random.choice(keys)
            if proto_txt:
                scl=ARG_ERROR_CODE_FONT_SCALE*(0.8+current_intensity*0.5);(tw,_),_=cv2.getTextSize(proto_txt,FONT_FACE_TO_USE,scl,3)
                tx,ty=max(5,(WIDTH-tw)//2+random.randint(-40,40)),max(20,(HEIGHT+_)//2+random.randint(-HEIGHT//6,HEIGHT//6))
                bg_mean=np.mean(frame[max(0,ty-_):min(HEIGHT,ty+10),max(0,tx):min(WIDTH,tx+tw)]) if ty-_>=0 else np.mean(frame)
                color=(255,random.randint(0,70),random.randint(0,70)) if bg_mean>100 else (230,random.randint(190,255),random.randint(190,255))
                overlay=frame.copy();cv2.putText(overlay,proto_txt,(tx,ty),FONT_FACE_TO_USE,scl,color,random.randint(2,3),cv2.LINE_AA)
                frame=cv2.addWeighted(frame,0.5,overlay,0.5,0) # Softer blend

        if random.random() < (0.08 + 0.12 * current_intensity): # Subliminal flash (reduced base chance)
            # Simplified subliminal flash for speed
            flash_type = random.choice(["invert", "color_wash", "contrast_spike"])
            overlay_sub = frame.copy()
            try:
                if flash_type == "invert": overlay_sub = 255 - frame
                elif flash_type == "color_wash":
                    wash_color = random.choice(random.choice(current_theme_mutated["colors"]))[:3]
                    overlay_sub = cv2.addWeighted(frame,0.3,np.full_like(frame,wash_color,dtype=np.uint8),0.7,0)
                elif flash_type == "contrast_spike":
                    overlay_sub = apply_extreme_contrast(overlay_sub,alpha_range=(2.5,6.0),beta_range=(-40,40))
                frame=cv2.addWeighted(frame,random.uniform(0.1,0.3),overlay_sub,random.uniform(0.7,0.9),0)
            except: pass

        # Persistent text (simplified update)
        frame_txt_overlay = frame.copy(); active_texts_new=[]
        for txt_info in persistent_texts:
            px,py=txt_info["pos"]; mx,my=txt_info["move_speed"]; txt_info["pos"]=(int(px+mx),int(py+my))
            # Simplified bounce
            if not (-WIDTH*0.2 < txt_info["pos"][0] < WIDTH*1.2): txt_info["move_speed"]=(-mx,my)
            if not (-HEIGHT*0.2 < txt_info["pos"][1] < HEIGHT*1.2): txt_info["move_speed"]=(mx,-my)
            txt_info["frame_count"]+=1
            if txt_info["frame_count"]<txt_info["lifetime"]:
                try:
                    scl=max(0.1,txt_info["font_size"]*(0.75+current_intensity*0.2)); thick=max(1,int(scl//2.5))
                    cv2.putText(frame_txt_overlay,txt_info["text"],txt_info["pos"],FONT_FACE_TO_USE,scl,txt_info["color"],thick,cv2.LINE_AA)
                    active_texts_new.append(txt_info)
                except: active_texts_new.append(txt_info)
            elif random.random()<0.65: # Reduced respawn complexity
                txt_info["text"]=zalgo_text(random.choice(current_theme_mutated["words"])) if random.random()<0.5 else random.choice(current_theme_mutated["symbols"])
                txt_info["pos"]=(random.randint(0,WIDTH),random.randint(0,HEIGHT)); txt_info["frame_count"]=0
                active_texts_new.append(txt_info)
        persistent_texts=active_texts_new; frame=frame_txt_overlay
        
        prev_frame = frame.copy()
        if frame_hold_counter > 0 and held_frame is None: held_frame = frame.copy()
        video.write(frame)
        if (i+1)%(FPS*10)==0 and frame_hold_counter==0: print(f"VFrame {i+1}/{FRAME_COUNT} (Int:{current_intensity:.2f})")
    video.release()
    return DURATION, FPS, video_temp_path

# --- Audio Generation (with broadcasting fix and performance tweaks) ---
def generate_audio_enhanced(video_duration, video_fps, theme_data, intensity_func, global_params):
    DURATION = video_duration; NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    if NUM_SAMPLES <= 0: return None 
    NUM_CHANNELS = 2; samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32)
    current_audio_params = copy.deepcopy(global_params["audio"]) 
    audio_instability_chance = global_params["instability_chance"] * 1.0 # Reduced
    print(f"Generating {DURATION:.1f}s audio, Theme: {global_params['theme_name']}...")
    min_freq, max_freq = theme_data["audio_freq_range"]
    noise_types = theme_data["audio_noise_types"]; features = theme_data["audio_features"]
    current_theme_mutated_audio = copy.deepcopy(theme_data) 
    last_event_end_sample = 0
    melody_track = np.zeros(NUM_SAMPLES, dtype=np.float32) 
    biometric_track_mono = np.zeros(NUM_SAMPLES, dtype=np.float32) 

    if any("heartbeat" in nt or "breathing" in nt or "biometric" in nt for nt in noise_types):
        bio_opts = [nt for nt in noise_types if "heart" in nt or "breath" in nt or "biometric" in nt]
        bio_type = random.choice(bio_opts) if bio_opts else "heartbeat_distorted"
        avg_intens = np.mean([intensity_func(t) for t in np.linspace(0,DURATION,5)]) # Fewer samples for mean
        bio_vol = 0.1 + 0.2 * avg_intens 
        biometric_track_mono = generate_biometric_rhythm(NUM_SAMPLES,bio_vol,rhythm_type=bio_type)

    while last_event_end_sample < NUM_SAMPLES:
        current_sample_time = last_event_end_sample/SAMPLE_RATE
        current_intensity = np.clip(intensity_func(current_sample_time),0.01,4.0)
        # Simplified audio param instability for speed
        if random.random() < audio_instability_chance * current_intensity * 0.5: pass # Reduced effect

        event_max_dur = max(0.005, (0.4 + 0.8/(current_intensity+0.4))) # Shorter max events
        event_dur_s = random.uniform(0.001, event_max_dur)
        event_dur_spl = min(max(1,int(event_dur_s*SAMPLE_RATE)), NUM_SAMPLES-last_event_end_sample)
        if event_dur_spl <= 0: break 
        event_start_spl, event_end_spl = last_event_end_sample, last_event_end_sample+event_dur_spl
        event_vol = np.clip(random.uniform(0.005,0.4)*(0.1+current_intensity*1.1),0.001,0.9)

        type_roll = random.random(); silence_th = max(0.003,0.08*(1.4-current_intensity))
        melody_th = silence_th+(0.04+0.12*current_intensity); noise_th = melody_th+(0.45+0.18*current_intensity)
        segment_mono = np.zeros(event_dur_spl,dtype=np.float32); event_type="silence"; is_melody=False

        if type_roll < silence_th: pass
        elif type_roll < melody_th:
            event_type="melody";is_melody=True; vol_mod=0.15+0.35*current_intensity
            segment_mono = generate_melody_fragment(event_dur_spl,event_vol*vol_mod,current_theme_mutated_audio,current_intensity)
        elif type_roll < noise_th:
            event_type="noise"; choice=random.choice(noise_types) if noise_types else "white"
            if current_intensity>1.8 and random.random()<0.6: # Prefer harsh noises less strictly
                 harsh=[n for n in noise_types if any(k in n for k in["screech","clip","artifact","glitch"])]
                 if harsh: choice=random.choice(harsh)
            segment_mono = generate_noise(choice,event_dur_spl,event_vol,features)
        else:
            event_type="tone"; fm_ch=np.clip(0.15+current_intensity*0.4,0.1,0.9); harm_ch=np.clip(0.08+current_intensity*0.3,0.1,0.85)
            freq=random.uniform(min_freq,max_freq*(0.25+current_intensity*0.65))
            segment_mono = generate_tone(np.clip(freq,LOW_FREQ_CUTOFF,HIGH_FREQ_CUTOFF),event_dur_spl,event_vol,fm_ch,harm_ch)
        
        # Ensure segment_mono is 1D before further processing and addition
        segment_mono = segment_mono.ravel() 

        if len(segment_mono) != event_dur_spl and event_dur_spl > 0: # Pad if necessary
            padded_seg = np.zeros(event_dur_spl,dtype=np.float32)
            len_copy = min(len(segment_mono),event_dur_spl)
            if len_copy > 0: padded_seg[:len_copy]=segment_mono[:len_copy]
            segment_mono = padded_seg
        
        if event_type!="silence" and len(segment_mono)>0:
            fx_chance = 0.4 + current_intensity*0.45 # Overall chance to apply any effect
            if "distortion" in features and random.random()<0.45*fx_chance:
                d_min,d_max=current_audio_params.get("distortion_intensity",(1.0,20.0)) # Reduced max
                segment_mono=apply_distortion(segment_mono,intensity_range=np.clip((d_min*(0.4+current_intensity*0.8),d_max*(0.7+current_intensity*1.0)),1.0,35.0))
            if "bitcrush" in features and random.random()<0.35*fx_chance:
                 b_min,b_max=current_audio_params.get("bitcrush_depth",(3,10)) # Higher min depth
                 target_bits=int(max(2,b_min+(b_max-b_min)*(1.0-current_intensity*0.15)))
                 segment_mono=apply_bitcrush(segment_mono,bit_depth_range=(max(2,target_bits-1),max(3,target_bits+1)))
            # Spectral glitch and reverb are expensive, reduce their application chance
            if "spectral_glitch" in features and random.random()<0.15*fx_chance: 
                segment_mono=apply_spectral_glitch(segment_mono,intensity=current_intensity*0.9)
            if "broken_speaker_sim" in features and random.random()<0.1*fx_chance: 
                segment_mono=apply_broken_speaker_sim(segment_mono,current_intensity)
            rev_feat = next((f for f in features if "convolution_reverb" in f),None)
            if rev_feat and random.random()<0.05*fx_chance: # Significantly reduced reverb chance
                 try:
                      imp_name=rev_feat.split("_")[-1]
                      if imp_name in IMPULSE_RESPONSES: segment_mono=apply_convolution_reverb(segment_mono,IMPULSE_RESPONSES[imp_name])
                 except: pass
            
            pan = random.uniform(-(0.2+current_intensity*0.6),(0.2+current_intensity*0.6)); pan=np.clip(pan,-1,1)
            g_l,g_r=np.sqrt(0.5*(1-pan)),np.sqrt(0.5*(1+pan))
            end_idx = min(event_end_spl,NUM_SAMPLES); length = end_idx-event_start_spl
            if length<=0: continue

            # --- BUG FIX: Ensure segment_mono is 1D before adding ---
            segment_to_add = segment_mono[:length].ravel() # Flatten segment

            if is_melody:
                melody_track[event_start_spl:end_idx] += segment_to_add
            else:
                samples[event_start_spl:end_idx,0] += segment_to_add * g_l
                samples[event_start_spl:end_idx,1] += segment_to_add * g_r
        
        overlap = np.clip(0.05+current_intensity*0.5,0.02,0.85) # Adjusted overlap
        advance = int(event_dur_spl*(1.0-overlap))
        last_event_end_sample += max(1,advance)

    bio_pan=random.uniform(-0.15,0.15); bio_gl,bio_gr=np.sqrt(0.5*(1-bio_pan)),np.sqrt(0.5*(1+bio_pan))
    intens_end = global_params.get("intensity_at_end",1.0); bio_mix = 0.1 + 0.18*intens_end 
    samples[:,0]+=biometric_track_mono*bio_mix*bio_gl; samples[:,1]+=biometric_track_mono*bio_mix*bio_gr
    
    max_mel=np.max(np.abs(melody_track)); 
    if max_mel>1e-6: melody_track/=max_mel
    mel_mix = 0.18 + 0.22*intens_end 
    samples[:,0]+=melody_track*mel_mix; samples[:,1]+=melody_track*mel_mix # Add melody to both channels
    
    max_abs=np.max(np.abs(samples)); 
    if max_abs>1e-6: samples/=max_abs
    elif NUM_SAMPLES > 0: print("Warning: Final audio mix is silent.")
        
    samples_int16=(np.clip(samples,-1.0,1.0)*32767).astype(np.int16)
    audio_temp_path=f"audio_temp_{int(time.time())}.wav" # Simpler temp name
    try:
        with wave.open(audio_temp_path,'wb') as wf:
            wf.setnchannels(NUM_CHANNELS); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples_int16.tobytes())
        return audio_temp_path
    except Exception as e: print(f"Error writing WAV: {e}"); return None

# --- Main Execution ---
if __name__ == "__main__":
    start_time_main = time.time()
    initial_theme_name = random.choice(list(THEMES.keys()))
    current_theme_data_master = copy.deepcopy(THEMES[initial_theme_name])
    print(f"--- Shardmind Protocol Generation v3.3 (Perf.Tweaks) ---") # Version bump
    print(f"Initial Theme: {initial_theme_name}")
    global_params = { "theme_name":initial_theme_name, "visual":{ "catalyst_threshold":2.9, # Slightly higher
        "effect_probabilities":{name:prob*0.85 for name,prob in [ # Globally reduce effect probs slightly
            ("apply_perlin_noise",0.65),("apply_block_shift",0.55),("apply_color_channel_shift",0.45), 
            ("apply_warp":0.45),("apply_pixelation":0.35),("apply_scanlines":0.25), 
            ("apply_solarize":0.18),("apply_extreme_contrast":0.45),("apply_feedback":0.55), 
            ("apply_datamosh_sim":0.25),("apply_ascii_sim":0.1), # Reduced ASCII sim prob significantly
            ("apply_sensor_overload":0.2),("apply_data_bleed_sim":0.18),("apply_flickering_glitch_layer":0.22), 
            ("apply_crt_ghost_sim":0.12),("apply_slit_scan_sim":0.08),("apply_vector_field_sim":0.08),
            ("apply_figurative_noise_sim":0.08)]},
        "perlin_alpha":(0.1,0.7),"perlin_scale":(5.0,60.0),"perlin_oct":(2,5), 
        "block_shift_max_shift":(5,WIDTH//4),"block_shift_block_size":(10,HEIGHT//3), 
        "color_channel_shift_max_shift":(3,40),"warp_amplitude":(3,50),"warp_freq":(0.003,0.08),
        "pixelation_block_size":(5,70),"extreme_contrast_alpha":(1.0,5.0),"extreme_contrast_beta":(-80,80),
        "feedback_alpha":(0.02,0.5),"solarize_threshold":(70,190),"scanlines_intensity":(0.3,0.8),
        "scanlines_thickness":(1,3),"datamosh_sim_block_size":(40,100),"ascii_sim_block_size":(12,22) # Larger blocks for ASCII
        },"audio":{"distortion_intensity":(1.2,20.0),"bitcrush_depth":(3,10)}, # Adjusted audio params
        "instability_chance":0.008, "arg_qr_prob_modifier":0.7, "arg_text_prob_modifier":0.7, 
        "arg_protocol_prob_modifier":0.7, "intensity_at_end":1.0 }

    est_max_dur_profile_s = MAX_DURATION+5; est_max_spl_profile = int(est_max_dur_profile_s*SAMPLE_RATE)
    intensity_function = generate_intensity_profile(est_max_spl_profile,SAMPLE_RATE)
    try: global_params["intensity_at_end"]=np.clip(intensity_function(MAX_DURATION-1),0.1,3.0)
    except: global_params["intensity_at_end"]=1.0

    actual_duration_video,actual_fps_video,video_temp_file_path = generate_frames_enhanced(current_theme_data_master,intensity_function,global_params)
    if not video_temp_file_path: print("Video gen failed!"); exit(1)
    audio_temp_file_path = generate_audio_enhanced(actual_duration_video,actual_fps_video,current_theme_data_master,intensity_function,global_params)
    if not audio_temp_file_path: 
        print("Audio gen failed!")
        if video_temp_file_path and os.path.exists(video_temp_file_path): os.remove(video_temp_file_path)
        exit(1)

    if video_temp_file_path and audio_temp_file_path:
        print("Combining video and audio with ffmpeg (using faster preset)...")
        if os.path.exists(OUTPUT_FILE):
            try: os.remove(OUTPUT_FILE)
            except OSError as e: print(f"Could not remove existing output: {e}")
        ffmpeg_command = ['ffmpeg','-y','-i',video_temp_file_path,'-i',audio_temp_file_path,
            '-c:v','libx264','-preset','fast', # Changed to 'fast' from 'medium'
            '-crf','24', # Slightly higher CRF for faster encoding, adjust if quality drops too much
            '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-shortest',OUTPUT_FILE]
        try:
            subprocess.run(ffmpeg_command,check=True,capture_output=True,text=True,encoding='utf-8',errors='ignore')
            print(f"Output: '{OUTPUT_FILE}'")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed. Code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        except FileNotFoundError: print("ffmpeg not found. Ensure it's installed and in PATH.")
        except Exception as e: print(f"ffmpeg error: {e}")
            
    for temp_file in [video_temp_file_path, audio_temp_file_path]:
        if temp_file and os.path.exists(temp_file):
            try: os.remove(temp_file)
            except OSError as e: print(f"Error removing temp file {temp_file}: {e}")
    print(f"--- Total Time: {time.time()-start_time_main:.2f}s ---")












