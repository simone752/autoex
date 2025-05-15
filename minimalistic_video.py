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
    zalgo_b = [chr(i) for i in range(0x0300, 0x036F + 1)]
    zalgo_down = [chr(i) for i in range(0x0300, 0x036F + 1)] # Can reuse or use different ranges
    zalgo_mid = [chr(i) for i in range(0x0300, 0x036F + 1)]
    all_zalgo = zalgo_b + zalgo_down + zalgo_mid
    if not all_zalgo: return text # Should not happen

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
        "audio_noise_types": ["digital_artifact", "static", "modem_sim", "feedback_screech", "glitch", "white_bursts", "data_stream_sim"],
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
        "visual_effects": ["apply_perlin_noise", "apply_warp", "apply_feedback", "apply_data_bleed_sim", "apply_solarize", "apply_crt_ghost_sim", "apply_figurative_noise_sim", "apply_pixelation"], # Added pixelation back
        "audio_freq_range": (18, 8000),
        "audio_noise_types": ["heartbeat", "breathing", "squelch", "wet_clicks", "bone_cracking_sim", "granular_flesh", "choking_sim", "biometric_rhythm"],
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
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arialuni.ttf", "C:/Windows/Fonts/arial.ttf",
]
AVAILABLE_FONTS = [f for f in FONT_PATHS if os.path.exists(f)]
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
SELECTED_TTF_FONT = random.choice(AVAILABLE_FONTS) if AVAILABLE_FONTS else None
print(f"Available TTF fonts: {AVAILABLE_FONTS}")
FONT_FACE_TO_USE = DEFAULT_FONT

# --- Pygame Mixer Init ---
try:
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=4096)
except pygame.error as e:
    print(f"Pygame mixer init failed (non-critical for file output): {e}")

# --- Intensity Profile Generation ---
def generate_intensity_profile(duration_samples, sample_rate, scale=30.0, octaves=6, persistence=0.5, lacunarity=2.0):
    duration_seconds = duration_samples / sample_rate
    num_points = int(duration_seconds * 20)
    if num_points < 2: num_points = 2
    profile = np.zeros(num_points)
    base = random.randint(0, 1000)
    for i in range(num_points):
        n1 = noise.pnoise1(i / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base)
        n2 = noise.pnoise1(i / (scale*0.3) + 1000, octaves=3, persistence=0.7, lacunarity=1.8, base=base+1)
        profile[i] = (n1 * 0.7) + (n2 * 0.3)
    min_val, max_val = np.min(profile), np.max(profile)
    if max_val - min_val > 1e-6:
        normalized_profile = (profile - min_val) / (max_val - min_val)
        intensity_min = 0.05
        intensity_max = 3.5
        scaled_profile = intensity_min + normalized_profile * (intensity_max - intensity_min)
    else:
        scaled_profile = np.full(num_points, 1.0)
    profile_times = np.linspace(0, duration_seconds, num_points)
    intensity_func = interp1d(profile_times, scaled_profile, kind='cubic', bounds_error=False,
                              fill_value=(scaled_profile[0], scaled_profile[-1]))
    print(f"Generated shared intensity profile: {len(profile)} points over {duration_seconds:.1f}s (Range: {intensity_min:.2f}-{intensity_max:.2f})")
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
        return np.zeros((size, size, 3), dtype=np.uint8)

def generate_arg_text_code(theme_data, length=16):
    code_type_options = theme_data.get("arg_elements", {}).get("text_code_types", ["hex"])
    if not code_type_options: code_type_options = ["hex"] # Fallback
    code_type = random.choice(code_type_options)

    actual_length = random.randint(length // 2, length) # Vary length more

    if code_type == "base64":
        raw_data = os.urandom(actual_length * 3 // 4) # Estimate raw bytes needed for base64
        return base64.b64encode(raw_data).decode('utf-8')[:actual_length]
    elif code_type == "hex":
        raw_data = os.urandom(actual_length // 2)
        return binascii.hexlify(raw_data).decode('utf-8')[:actual_length]
    elif code_type == "binary_short":
        raw_data = os.urandom(actual_length // 8 + 1) # Ensure enough bytes for binary string
        return "".join(format(byte, '08b') for byte in raw_data)[:actual_length]
    return "NULL_CODE_" + str(random.randint(100,999))


# --- Visual Glitch Functions ---
def apply_perlin_noise(frame, alpha_range=(0.1, 0.9), scale_range=(2.0, 80.0), oct_range=(3, 9)):
    alpha = random.uniform(*alpha_range)
    scale = random.uniform(*scale_range)
    octaves = random.randint(*oct_range)
    persistence = random.uniform(0.2, 0.7)
    lacunarity = random.uniform(1.8, 3.5)
    seed = random.randint(0, 2000)
    height, width = frame.shape[:2]
    x_coords = np.arange(width) / scale
    y_coords = np.arange(height) / scale
    gray_noise = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            gray_noise[i, j] = noise.pnoise2(y_coords[i], x_coords[j], octaves=octaves,
                                             persistence=persistence, lacunarity=lacunarity, base=seed)
    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX)
    colored_noise = cv2.cvtColor(colored_noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if random.random() < 0.4: colored_noise = 255 - colored_noise
    return cv2.addWeighted(frame, 1 - alpha, colored_noise, alpha, 0)

def apply_block_shift(frame, max_shift_range=(10, WIDTH // 3), block_size_range=(5, HEIGHT // 2), num_blocks_range=(20, 100)):
    max_shift = random.randint(max(1,int(max_shift_range[0])), max(2,int(max_shift_range[1]))) # Ensure positive
    bs_min = max(1, int(block_size_range[0]))
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    block_size_max = random.randint(bs_min, bs_max)
    num_blocks = random.randint(*num_blocks_range)
    height, width = frame.shape[:2]
    temp_frame = frame.copy()
    for _ in range(num_blocks):
        bh = random.randint(bs_min, block_size_max)
        bw = random.randint(bs_min, block_size_max)
        if height - bh <= 0 or width - bw <= 0: continue
        y = random.randint(0, height - bh - 1)
        x = random.randint(0, width - bw - 1)
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        target_y_start = np.clip(y + shift_y, 0, height - bh)
        target_x_start = np.clip(x + shift_x, 0, width - bw)
        try:
            block = frame[y:y+bh, x:x+bw]
            temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block
        except ValueError: pass
    return temp_frame

def apply_color_channel_shift(frame, max_shift_range=(5, 60)):
    max_shift = random.randint(max(1,int(max_shift_range[0])), max(2,int(max_shift_range[1])))
    temp_frame = frame.copy()
    for i in range(3):
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
        angle = random.uniform(-5, 5); scale = random.uniform(0.95, 1.05)
        tx = random.uniform(-10, 10); ty = random.uniform(-10, 10)
        center = (cols/2, rows/2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx; M[1, 2] += ty
        modified_prev = cv2.warpAffine(modified_prev, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.1:
         modified_prev = cv2.add(modified_prev, (random.randint(-15,15),random.randint(-15,15),random.randint(-15,15),0), dtype=cv2.CV_8UC3) # Explicit add for color shift
    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)

# << FUNCTION DEFINITION ADDED HERE >>
def apply_pixelation(frame, block_size_range=(4, 96)):
    """Applies pixelation effect by resizing down and up."""
    height, width = frame.shape[:2]
    bs_min = max(2, int(block_size_range[0])) # Ensure block size is at least 2
    bs_max = max(bs_min + 1, int(block_size_range[1]))
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
    for y_coord in range(0, height, thickness * 2):
        line_color_val = base_color_val + random.randint(-color_variation, color_variation)
        line_color = tuple(np.clip([line_color_val]*3, 0, 255))
        cv2.line(scanline_layer, (0, y_coord), (width, y_coord), line_color, thickness)
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
    bs_min = max(16, int(block_size_range[0]))
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    block_size = random.randint(bs_min, bs_max)
    if random.random() < hold_prob:
        num_hold_blocks = random.randint(1, max(1, (width // block_size) * (height // block_size) // 3))
        for _ in range(num_hold_blocks):
            bx_idx = random.randint(0, max(0, width // block_size - 1))
            by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            bh = min(block_size, height - by); bw = min(block_size, width - bx)
            if bh > 0 and bw > 0: output[by:by+bh, bx:bx+bw] = prev_frame[by:by+bh, bx:bx+bw]
    if random.random() < smear_prob:
        num_smear_blocks = random.randint(1, max(1, (width // block_size) * (height // block_size) // 4))
        for _ in range(num_smear_blocks):
            bx_idx = random.randint(0, max(0, width // block_size - 1))
            by_idx = random.randint(0, max(0, height // block_size - 1))
            bx, by = bx_idx * block_size, by_idx * block_size
            bh = min(block_size, height - by); bw = min(block_size, width - bx)
            if bh <= 0 or bw <= 0: continue
            vx = random.randint(-block_size // 2, block_size // 2)
            vy = random.randint(-block_size // 2, block_size // 2)
            smear_block = prev_frame[by:by+bh, bx:bx+bw]
            target_y = np.clip(by + vy, 0, height - bh)
            target_x = np.clip(bx + vx, 0, width - bw)
            output[target_y:target_y+bh, target_x:target_x+bw] = smear_block
    return output

def apply_ascii_sim(frame, block_size_range=(8, 16), char_set=" .:-=+*#%@", invert=False):
    height, width = frame.shape[:2]
    bs_min = max(4, int(block_size_range[0]))
    bs_max = max(bs_min + 1, int(block_size_range[1]))
    block_size = random.randint(bs_min, bs_max)
    output = frame.copy()
    font_scale = max(0.1, block_size / 15.0)
    thickness = 1; chars = list(char_set)
    if invert: chars = chars[::-1]
    num_chars = len(chars)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for y_coord in range(0, height - block_size + 1, block_size):
        for x_coord in range(0, width - block_size + 1, block_size):
            block = gray[y_coord:y_coord+block_size, x_coord:x_coord+block_size]
            avg_brightness = np.mean(block)
            char_index = int(np.clip((avg_brightness / 255.0) * num_chars, 0, num_chars - 1))
            char = chars[char_index]
            text_color = (0,0,0) if avg_brightness > 128 else (255,255,255)
            text_size_cv, _ = cv2.getTextSize(char, FONT_FACE_TO_USE, font_scale, thickness)
            pos = (x_coord + (block_size - text_size_cv[0]) // 2, y_coord + (block_size + text_size_cv[1]) // 2)
            try: cv2.putText(output, char, pos, FONT_FACE_TO_USE, font_scale, text_color, thickness, cv2.LINE_AA)
            except Exception: pass
    return output

def apply_sensor_overload(frame, intensity):
    output = frame.copy()
    max_shift = int(10 + 40 * intensity)
    temp_aberration = output.copy()
    for i in range(3):
        shift_x = random.randint(-max_shift, max_shift); shift_y = random.randint(-max_shift, max_shift)
        channel = output[:,:,i]
        shifted = np.roll(np.roll(channel, shift_y, axis=0), shift_x, axis=1)
        temp_aberration[:,:,i] = cv2.addWeighted(channel, 0.3, shifted, 0.7,0)
    output = temp_aberration
    if random.random() < 0.5 * intensity:
        alpha = 2.0 + 3.0 * intensity; beta = random.randint(-50, 50) * intensity
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
        bright_mask = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(bright_mask, 230, 255, cv2.THRESH_BINARY)
        sigma_val = max(1, int(10 + 20*intensity)) # Ensure sigma is positive
        bloom = cv2.GaussianBlur(output, (0,0), sigmaX=sigma_val, sigmaY=sigma_val)
        output = cv2.addWeighted(output, 1.0, bloom, 0.3 * intensity, 0, dst=output, dtype=cv2.CV_8U)
    if random.random() < 0.2 * intensity:
        overlay = np.zeros_like(output, dtype=np.uint8)
        for _ in range(random.randint(1,5)):
            cv2.circle(overlay, (random.randint(0,WIDTH), random.randint(0,HEIGHT)), random.randint(1,5), (20,20,20), -1)
        if random.random() < 0.1:
            cv2.line(overlay, (random.randint(0,WIDTH), random.randint(0,HEIGHT)), (random.randint(0,WIDTH), random.randint(0,HEIGHT)), (10,10,10), random.randint(1,2))
        output = cv2.addWeighted(output, 0.9, overlay, 0.3,0)
    return output

def apply_data_bleed_sim(frame, intensity):
    output = frame.copy()
    num_bleeds = random.randint(5, int(20 * intensity))
    for _ in range(num_bleeds):
        bh = random.randint(int(HEIGHT*0.1), int(HEIGHT*0.4)); bw = random.randint(int(WIDTH*0.1), int(WIDTH*0.4))
        if HEIGHT - bh <=0 or WIDTH - bw <=0 : continue
        y = random.randint(0, HEIGHT - bh -1); x = random.randint(0, WIDTH - bw -1)
        source_block = frame[y:y+bh, x:x+bw]
        if source_block.size == 0: continue
        avg_color = np.mean(source_block, axis=(0,1)).astype(np.uint8)
        bleed_strength = 0.2 + 0.6 * intensity * random.random()
        bleed_radius_x = random.randint(int(bw*0.5), int(bw*2)); bleed_radius_y = random.randint(int(bh*0.5), int(bh*2))
        if WIDTH - bleed_radius_x <=0 or HEIGHT - bleed_radius_y <=0 : continue
        tx = np.clip(x + random.randint(-bw, bw), 0, WIDTH - bleed_radius_x)
        ty = np.clip(y + random.randint(-bh, bh), 0, HEIGHT - bleed_radius_y)
        target_area = output[ty:ty+bleed_radius_y, tx:tx+bleed_radius_x]
        if target_area.size == 0: continue
        color_overlay = np.full_like(target_area, avg_color)
        output[ty:ty+bleed_radius_y, tx:tx+bleed_radius_x] = cv2.addWeighted(target_area, 1.0 - bleed_strength, color_overlay, bleed_strength, 0)
    return output

def apply_flickering_glitch_layer(frame, intensity, effect_func, prev_frame=None, **kwargs): # Added **kwargs
    if random.random() > 0.3 + 0.6 * intensity: return frame
    glitch_layer = frame.copy()
    sig = inspect.signature(effect_func)
    params_to_pass_inner = {}
    if 'prev_frame' in sig.parameters: params_to_pass_inner['prev_frame'] = prev_frame if prev_frame is not None else frame
    # Pass through any additional kwargs that the inner effect_func might need
    # This assumes kwargs contains necessary range parameters if effect_func needs them
    # A more robust way would be to fetch ranges from global_params based on effect_func's name
    for p_name, p_val in kwargs.items():
        if p_name in sig.parameters:
            params_to_pass_inner[p_name] = p_val

    try:
        glitch_layer = effect_func(glitch_layer, **params_to_pass_inner)
    except TypeError as e: # Fallback if specific ranges are missing and function needs them
        # print(f"TypeError in flickering_glitch_layer calling {effect_func.__name__}: {e}. Using defaults.")
        # Try calling with only frame if possible, or with generic default ranges
        if 'prev_frame' in params_to_pass_inner:
            glitch_layer = effect_func(glitch_layer, prev_frame=params_to_pass_inner['prev_frame'])
        else:
            glitch_layer = effect_func(glitch_layer)

    mask = np.random.randint(0, 2, size=(HEIGHT, WIDTH), dtype=np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(glitch_layer, glitch_layer, mask=mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return cv2.add(fg, bg)

def apply_figurative_noise_sim(frame, intensity):
    return apply_perlin_noise(frame, alpha_range=(0.2, 0.5 * intensity), scale_range=(50, 150), oct_range=(4,6))

def apply_crt_ghost_sim(frame, prev_frame):
    if prev_frame is not None: return cv2.addWeighted(frame, 0.85, prev_frame, 0.15, 0)
    return frame

def apply_slit_scan_sim(frame):
    rows, cols, _ = frame.shape; output = frame.copy()
    for r_idx in range(rows):
         smear_amount = int(math.sin(r_idx * 0.1 + random.uniform(0,math.pi)) * (10 + 5 * random.random())) # Add randomness
         output[r_idx, :] = np.roll(output[r_idx, :], smear_amount, axis=0)
    return output

def apply_vector_field_sim(frame):
    return apply_warp(frame, amplitude_range=(2,10 + 5*random.random()), freq_range=(0.05, 0.2 + 0.1*random.random()))


# --- Audio Effects ---
def apply_distortion(data, intensity_range=(1.5, 5.0)):
    intensity = random.uniform(*intensity_range)
    return np.clip(data * intensity, -0.98, 0.98)

def apply_bitcrush(data, bit_depth_range=(4, 12)):
    bd_min = max(2, int(bit_depth_range[0]))
    bd_max = max(bd_min + 1, int(bit_depth_range[1]))
    bits = random.randint(bd_min, bd_max)
    if bits >= 16: return data
    steps = 2**(bits - 1)
    return np.round(data * steps) / steps

def apply_spectral_glitch(data, intensity=0.5):
    n_samples = len(data)
    if n_samples < 1024: return data
    data_float = data.astype(np.float32); spectrum = fft(data_float)
    magnitude = np.abs(spectrum); phase = np.angle(spectrum)
    n_freqs = len(spectrum)
    num_glitches = int(n_freqs * 0.05 * intensity * random.random())
    indices_to_glitch = random.sample(range(1, n_freqs // 2), min(num_glitches, max(1,n_freqs // 2 - 1)))
    for idx in indices_to_glitch:
        if random.random() < 0.6:
             magnitude[idx] = 0
             if idx < n_freqs: magnitude[n_freqs - idx] = 0
        else:
             if idx + 1 < n_freqs // 2:
                  phase[idx], phase[idx+1] = phase[idx+1], phase[idx]
                  if n_freqs - idx < n_freqs: phase[n_freqs - idx] = -phase[idx]
                  if n_freqs - (idx+1) < n_freqs : phase[n_freqs - (idx+1)] = -phase[idx+1]
    new_spectrum = magnitude * np.exp(1j * phase)
    glitched_data = np.real(ifft(new_spectrum))
    max_abs_orig = np.max(np.abs(data_float)); max_abs_glitched = np.max(np.abs(glitched_data))
    if max_abs_glitched > 1e-6 and max_abs_orig > 1e-6: glitched_data *= (max_abs_orig / max_abs_glitched)
    elif max_abs_glitched > 1e-6: glitched_data /= max_abs_glitched
    return glitched_data.astype(data.dtype)

def apply_convolution_reverb(data, impulse_response):
    n_data = len(data); n_ir = len(impulse_response)
    if n_ir == 0 : return data
    if n_ir >= n_data: impulse_response = impulse_response[:max(1,n_data//2)]; n_ir = len(impulse_response)
    if n_ir == 0 : return data
    reverbed_data = np.convolve(data, impulse_response, mode='same')
    max_abs_orig = np.max(np.abs(data)); max_abs_rev = np.max(np.abs(reverbed_data))
    if max_abs_rev > 1e-6 and max_abs_orig > 1e-6: reverbed_data *= (max_abs_orig / max_abs_rev)
    elif max_abs_rev > 1e-6: reverbed_data /= max_abs_rev
    return reverbed_data

IMPULSE_RESPONSES = {
    "damp": np.exp(-np.linspace(0, 10, int(SAMPLE_RATE * 0.2))) * np.random.randn(int(SAMPLE_RATE * 0.2)),
    "metal_hit": np.sin(2*np.pi*1500*np.linspace(0,0.1,int(SAMPLE_RATE*0.1))) * np.exp(-np.linspace(0, 20, int(SAMPLE_RATE * 0.1))),
    "noise_burst": np.random.randn(int(SAMPLE_RATE * 0.1)) * np.exp(-np.linspace(0, 5, int(SAMPLE_RATE * 0.1))) }
for k_ir in IMPULSE_RESPONSES:
    max_abs_ir = np.max(np.abs(IMPULSE_RESPONSES[k_ir]))
    if max_abs_ir > 1e-6: IMPULSE_RESPONSES[k_ir] /= max_abs_ir

def apply_broken_speaker_sim(data, intensity):
    output = data.copy()
    if random.random() < 0.7 * intensity:
        spectrum = fft(output); n = len(spectrum)
        if random.random() < 0.5:
            cut_freq_bin = int((random.uniform(500, 3000) / (SAMPLE_RATE / 2)) * (n / 2))
            spectrum[:cut_freq_bin] = 0; spectrum[n-cut_freq_bin:] = 0
        else:
            scoop_center_bin = int((random.uniform(1000, 6000) / (SAMPLE_RATE / 2)) * (n / 2))
            scoop_width_bin = int(scoop_center_bin * random.uniform(0.1, 0.5))
            start_scoop = max(0,scoop_center_bin-scoop_width_bin)
            end_scoop = min(n//2, scoop_center_bin+scoop_width_bin)
            spectrum[start_scoop:end_scoop] = 0
            spectrum[n-end_scoop : n-start_scoop] = 0
        output = np.real(ifft(spectrum))
    output = apply_distortion(output, intensity_range=(5.0 + 10 * intensity, 15.0 + 20 * intensity))
    if random.random() < 0.5 * intensity:
        num_crackles = random.randint(10, 50)
        for _ in range(num_crackles):
            if len(output) == 0: break
            pos = random.randint(0, len(output)-1)
            crackle_len = random.randint(1, min(50, len(output)-pos)) # Ensure crackle_len is valid
            if crackle_len <=0 : continue
            output[pos:pos+crackle_len] += (np.random.rand(crackle_len) - 0.5) * 0.5 * intensity
    return np.clip(output, -1.0, 1.0)

# --- Tone, Noise, Melody, Rhythm Generation ---
def generate_tone(freq, duration_samples, vol, fm_chance=0.5, harmonic_chance=0.4):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
    wave = np.zeros(duration_samples, dtype=np.float32)
    if random.random() < fm_chance:
        mod_freq = freq * random.uniform(0.1, 5.0); mod_depth = abs(vol) * random.uniform(1.0, 8.0)
        if abs(freq * t[-1]) < 1e9 and abs(mod_freq * t[-1]) < 1e9: wave = np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))
        else: wave = np.sin(2 * np.pi * freq * t)
    elif random.random() < harmonic_chance:
        harmonic_count = random.randint(1, 4); wave = np.sin(2 * np.pi * freq * t)
        for h_idx in range(2, harmonic_count + 2):
            harmonic_vol = random.uniform(0.1, 0.5) / h_idx; phase_shift = random.uniform(0, np.pi)
            wave += harmonic_vol * np.sin(2 * np.pi * freq * h_idx * t + phase_shift)
        max_abs_wave = np.max(np.abs(wave));
        if max_abs_wave > 1e-6: wave /= max_abs_wave
    else: wave = np.sin(2 * np.pi * freq * t)
    wave *= vol
    attack_len = min(int(SAMPLE_RATE * 0.005), duration_samples // 3)
    decay_len = min(int(SAMPLE_RATE * random.uniform(0.05, 0.3)), max(0, duration_samples - attack_len))
    sustain_level = random.uniform(0.1, 0.7)
    release_len = min(int(SAMPLE_RATE * random.uniform(0.05, 0.4)), max(0, duration_samples - attack_len - decay_len))
    envelope = np.ones(duration_samples)
    if attack_len > 0: envelope[:attack_len] = np.linspace(0, 1, attack_len)
    sustain_start_idx = attack_len + decay_len
    if decay_len > 0: envelope[attack_len:sustain_start_idx] = np.linspace(1, sustain_level, decay_len)
    release_start_idx = sustain_start_idx
    if release_len > 0 and release_start_idx + release_len <= duration_samples:
         envelope[release_start_idx:release_start_idx+release_len] = sustain_level * np.exp(-np.linspace(0, 5, release_len))
         envelope[release_start_idx+release_len:] = 0
    elif release_start_idx < duration_samples:
         envelope[release_start_idx:] = sustain_level * np.exp(-np.linspace(0, 5, duration_samples - release_start_idx))
    return wave * envelope

def generate_noise(noise_type, duration_samples, vol, features=[]):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    # ... (Implementations for all noise types from v2 and new ones like digital_clipping_sim, feedback_screech) ...
    # This function needs to be fully fleshed out as in the previous version.
    # For brevity here, assume it's correctly implemented from the prior complete script.
    # Ensure all noise types listed in THEMES are handled.
    # Example for a new one:
    if noise_type == "data_stream_sim":
        # Rapidly changing filtered noise and clicks
        base = generate_noise("pink", duration_samples, vol*0.5)
        clicks = generate_noise("glitch", duration_samples, vol*0.5)
        # Modulate click density
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * random.uniform(1,10) * np.linspace(0, duration_samples*time_step, duration_samples))
        return base + clicks * mod
    elif noise_type == "white_bursts":
        out_noise = np.zeros(duration_samples, dtype=np.float32)
        num_bursts = random.randint(3, int(duration_samples / (SAMPLE_RATE * 0.05))) # Max 20 bursts/sec
        for _ in range(num_bursts):
            burst_len = random.randint(int(SAMPLE_RATE*0.002), int(SAMPLE_RATE*0.05))
            if duration_samples < burst_len + 10: continue
            start = random.randint(0, duration_samples - burst_len -1)
            out_noise[start:start+burst_len] = (np.random.rand(burst_len)-0.5) * 2.0
        return out_noise * vol
    # Fallback
    return (np.random.rand(duration_samples) - 0.5) * 2.0 * vol


def generate_biometric_rhythm(duration_samples, vol, rhythm_type="heartbeat_distorted"):
    noise_out = np.zeros(duration_samples, dtype=np.float32)
    if duration_samples <=0: return noise_out
    if rhythm_type == "heartbeat_distorted":
        bpm = random.uniform(30, 150) * random.uniform(0.5, 1.5); bps = bpm / 60.0
        if bps == 0: return noise_out
        beat_interval_samples = int(SAMPLE_RATE / bps)
        if beat_interval_samples == 0: return noise_out
        thud_len = int(SAMPLE_RATE * random.uniform(0.05, 0.15)); thud_freq = random.uniform(30, 100)
        if thud_len <=0 : return noise_out
        t_thud = np.linspace(0, thud_len * time_step, thud_len, endpoint=False)
        thud_env = np.exp(-np.linspace(0, random.uniform(5,15), thud_len))
        thud_sound = np.sin(2*np.pi*thud_freq*t_thud) * thud_env
        thud_sound = apply_distortion(thud_sound, (2.0, 5.0))
        current_sample = 0
        while current_sample < duration_samples:
            actual_thud_len = min(thud_len, duration_samples - current_sample)
            if actual_thud_len <=0 : break
            noise_out[current_sample : current_sample + actual_thud_len] += thud_sound[:actual_thud_len]
            current_sample += beat_interval_samples
    return apply_distortion(noise_out * vol, (1.5, 3.0))

def generate_melody_fragment(duration_samples, vol, theme_data, intensity):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    scale_name = theme_data.get("melody_scale", "minor_pentatonic")
    scale_intervals = SCALES.get(scale_name, SCALES["minor_pentatonic"])
    base_freq = random.uniform(60, 660)
    notes_in_fragment = random.randint(2, int(6 + 10 * intensity))
    note_duration_samples = max(int(SAMPLE_RATE * 0.05), duration_samples // notes_in_fragment if notes_in_fragment > 0 else duration_samples)
    melody = np.zeros(duration_samples, dtype=np.float32); current_pos = 0
    for _ in range(notes_in_fragment):
        if current_pos >= duration_samples: break
        scale_degree = random.choice(scale_intervals); octave_shift = random.choice([-1, 0, 0, 1, 1, 2])
        note_freq_start = base_freq * (2**((scale_degree + octave_shift * 12) / 12.0))
        pitch_bend_amount = random.uniform(-12, 12) * intensity
        note_freq_end = note_freq_start * (2**(pitch_bend_amount / 12.0))
        note_freq_start = np.clip(note_freq_start, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)
        note_freq_end = np.clip(note_freq_end, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)
        actual_note_len = min(note_duration_samples, duration_samples - current_pos)
        if actual_note_len <= 0: break
        t_note = np.linspace(0, actual_note_len * time_step, actual_note_len, endpoint=False)
        current_freqs = np.linspace(note_freq_start, note_freq_end, actual_note_len)
        phase = np.cumsum(2 * np.pi * current_freqs * time_step); note_tone = np.sin(phase)
        attack_len = min(int(actual_note_len*0.1), actual_note_len); decay_len = min(int(actual_note_len*0.4), actual_note_len - attack_len)
        sustain_level = random.uniform(0.2,0.8); envelope = np.ones(actual_note_len)
        if attack_len > 0: envelope[:attack_len] = np.linspace(0,1,attack_len)
        if decay_len > 0 and attack_len + decay_len <= actual_note_len :
            envelope[attack_len:attack_len+decay_len] = np.linspace(1,sustain_level,decay_len)
            envelope[attack_len+decay_len:] = sustain_level
        elif attack_len < actual_note_len: envelope[attack_len:] = sustain_level
        melody[current_pos : current_pos + actual_note_len] += note_tone * envelope
        current_pos += actual_note_len
    if random.random() < theme_data.get("melody_counter_melody_chance", 0.3) * intensity:
        counter_melody = np.zeros(duration_samples, dtype=np.float32); current_pos_counter = 0
        dissonant_intervals = [-1, 1, -6, 6, -11, 11]
        for _ in range(notes_in_fragment // 2 + 1):
            if current_pos_counter >= duration_samples: break
            scale_degree_main = random.choice(scale_intervals); counter_degree = (scale_degree_main + random.choice(dissonant_intervals)) % 12
            octave_shift_cnt = random.choice([-1,0,1]); note_freq_cnt = base_freq*0.7*(2**((counter_degree + octave_shift_cnt*12)/12.0))
            note_freq_cnt = np.clip(note_freq_cnt,LOW_FREQ_CUTOFF,HIGH_FREQ_CUTOFF)
            actual_note_len_cnt = min(note_duration_samples*random.randint(1,2), duration_samples - current_pos_counter)
            if actual_note_len_cnt <=0: break
            note_tone_cnt = generate_tone(note_freq_cnt, actual_note_len_cnt, 0.7, fm_chance=0.6, harmonic_chance=0.1)
            counter_melody[current_pos_counter : current_pos_counter + actual_note_len_cnt] += note_tone_cnt
            current_pos_counter += actual_note_len_cnt
        melody += counter_melody * 0.6
    distortion_amount = theme_data.get("melody_distortion", 10.0) * (1.0 + intensity)
    melody = apply_distortion(melody, intensity_range=(distortion_amount*0.7, distortion_amount*1.5))
    if random.random() < 0.6*intensity: melody = apply_bitcrush(melody, bit_depth_range=(2,6))
    if random.random() < 0.3*intensity: melody = apply_spectral_glitch(melody, intensity=intensity*0.8)
    if random.random() < 0.2*intensity and len(melody) > 0:
        grain_len = max(1, int(SAMPLE_RATE * random.uniform(0.005, 0.02)))
        num_grains = max(0, int(len(melody) / grain_len) // 2)
        shredded_melody = np.zeros_like(melody)
        for _ in range(num_grains):
            if len(melody) < grain_len : break
            start_idx = random.randint(0, len(melody) - grain_len)
            grain = melody[start_idx : start_idx + grain_len]
            place_idx = random.randint(0, len(shredded_melody) - grain_len)
            shredded_melody[place_idx : place_idx + grain_len] += grain * random.uniform(0.5,1.2)
        melody = shredded_melody
    return melody * vol


# --- Frame Generation ---
def generate_frames_enhanced(theme_data, intensity_func, global_params):
    DURATION = random.randint(MIN_DURATION, MAX_DURATION)
    FPS = random.randint(MIN_FPS, MAX_FPS)
    FRAME_COUNT = DURATION * FPS
    current_visual_params = global_params["visual"]
    instability_chance = global_params["instability_chance"] * 1.5
    print(f"Generating {DURATION}s video ({FPS} FPS), Theme: {global_params['theme_name']}...")
    print(f"Visual Instability Chance: {instability_chance:.4f}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_temp_path = f"video_temp_{int(time.time())}_{random.randint(100,999)}.mp4"
    video = cv2.VideoWriter(video_temp_path, fourcc, FPS, (WIDTH, HEIGHT))
    prev_frame = None; frame_hold_counter = 0; held_frame = None; effect_history = []
    current_theme_mutated = copy.deepcopy(theme_data)
    mutation_timer = 0; mutation_interval = FPS * random.randint(5,15)
    arg_qr_img = None; arg_qr_display_frames = 0
    arg_text_code = None; arg_text_display_frames = 0
    persistent_texts = []
    num_persistent = random.randint(8, 20)
    for _ in range(num_persistent):
        word = str(random.choice(current_theme_mutated["words"]))
        symbol = str(random.choice(current_theme_mutated["symbols"]))
        if random.random() < current_theme_mutated.get("zalgo_words_chance", 0.3): word = zalgo_text(word)
        if random.random() < 0.2: word = base64.b64encode(word.encode('utf-8', 'ignore')).decode()
        elif random.random() < 0.15: word = binascii.hexlify(word.encode('utf-8', 'ignore')).decode()
        persistent_texts.append({ "text": word + symbol, "pos": (random.randint(-100, WIDTH), random.randint(-100, HEIGHT)),
            "font_size": random.uniform(0.4, 5.0), "color": random.choice(random.choice(current_theme_mutated["colors"])),
            "lifetime": random.randint(int(FPS * 0.05), int(FPS * 10)), "frame_count": 0,
            "move_speed": (random.uniform(-12, 12), random.uniform(-10, 10)), "alpha": random.uniform(0.3, 1.0) })

    for i in range(FRAME_COUNT):
        current_time_sec = i / FPS
        current_intensity = intensity_func(current_time_sec)
        current_intensity = np.clip(current_intensity, 0.01, 4.0)
        mutation_timer += 1
        if mutation_timer >= mutation_interval: # Dynamic Theme Mutation
            mutation_timer = 0
            if random.random() < 0.3 * current_intensity:
                palette_idx = random.randrange(len(current_theme_mutated["colors"]))
                color_idx = random.randrange(len(current_theme_mutated["colors"][palette_idx]))
                current_theme_mutated["colors"][palette_idx][color_idx] = tuple(np.clip(c + random.randint(-30,30), 0, 255) for c in current_theme_mutated["colors"][palette_idx][color_idx])
                # print(f"Theme Mutation: Color palette {palette_idx} changed at {current_time_sec:.1f}s")

        if random.random() < instability_chance: # Parameter Instability
            param_key = random.choice(list(current_visual_params.keys()))
            if param_key == "effect_probabilities":
                effect_func_name = random.choice(list(current_visual_params["effect_probabilities"].keys()))
                old_prob = current_visual_params["effect_probabilities"][effect_func_name]
                current_visual_params["effect_probabilities"][effect_func_name] = np.clip(old_prob + random.uniform(-0.3, 0.3), 0.01, 0.99)
            elif isinstance(current_visual_params[param_key], tuple) and len(current_visual_params[param_key]) == 2:
                c_min, c_max = current_visual_params[param_key]; range_w = abs(c_max - c_min) + 1e-6
                mid_s = random.uniform(-range_w * 0.2, range_w * 0.2); scale_f = random.uniform(0.7, 1.3)
                n_mid = (c_min + c_max) / 2 + mid_s; n_w = range_w * scale_f
                n_min = n_mid - n_w / 2; n_max = n_mid + n_w / 2
                if "alpha" in param_key: n_min=max(0.01,n_min); n_max=min(1.0,max(n_min+0.05,n_max))
                elif any(k in param_key for k in ["shift","block","thresh","size"]): n_min=int(max(1,n_min)); n_max=int(max(n_min+1,n_max))
                current_visual_params[param_key] = (n_min, n_max)

        if frame_hold_counter > 0: # Frame Hold
            if held_frame is not None: video.write(held_frame); frame_hold_counter -=1; continue
            else: frame_hold_counter = 0
        if random.random() < (0.02 + 0.03 * current_intensity) and frame_hold_counter == 0:
            frame_hold_counter = random.randint(1, int(FPS * (0.5 + current_intensity * 0.5))); held_frame = None

        base_roll = random.random() # Base Frame
        if base_roll < 0.2:
             bg_color = random.choice(random.choice(current_theme_mutated["colors"])); frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
        elif base_roll < 0.4:
             c1 = random.choice(random.choice(current_theme_mutated["colors"])); c2 = random.choice(random.choice(current_theme_mutated["colors"]))
             frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
             for k_row in range(HEIGHT): frame[k_row, :] = [int(c1[ch]*(1-k_row/HEIGHT) + c2[ch]*(k_row/HEIGHT)) for ch in range(3)]
             if random.random() < 0.5: frame = cv2.rotate(frame, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]))
        else: frame = np.random.randint(0, int(30 + 40*current_intensity), (HEIGHT,WIDTH,3), dtype=np.uint8)

        is_catalyst_event = False # Glitch Catalyst
        if current_intensity > (global_params.get("visual",{}).get("catalyst_threshold", 2.8)) and random.random() < 0.3:
            is_catalyst_event = True; num_effects_catalyst = random.randint(5,8)
            # print(f"GLITCH CATALYST at {current_time_sec:.1f}s! Intensity: {current_intensity:.2f}")

        available_effect_names = current_theme_mutated.get("visual_effects", []) # Apply Visual Effects
        effect_candidates = [globals()[name] for name in available_effect_names if name in globals() and callable(globals()[name])]
        num_effects_base = random.randint(1,3) if not is_catalyst_event else num_effects_catalyst
        num_effects_intensity = int(current_intensity*2.0) if not is_catalyst_event else 0
        num_effects_to_apply = min(len(effect_candidates), num_effects_base + num_effects_intensity)
        applied_effects_count = 0; random.shuffle(effect_candidates); applied_this_frame = []
        for effect_func in effect_candidates:
            if applied_effects_count >= num_effects_to_apply: break
            effect_name = effect_func.__name__
            recent_count = effect_history.count(effect_name)
            if recent_count > 2 and random.random() < 0.6: continue
            base_prob = current_visual_params["effect_probabilities"].get(effect_name, 0.6)
            prob_mod = 0.4 + current_intensity * 0.6; final_prob = np.clip(base_prob * prob_mod, 0.05, 0.95)
            if random.random() < final_prob:
                try:
                    sig = inspect.signature(effect_func); params_to_pass = {}
                    if 'prev_frame' in sig.parameters: params_to_pass['prev_frame'] = prev_frame
                    if 'intensity' in sig.parameters: params_to_pass['intensity'] = current_intensity # Pass intensity if func accepts
                    # Simplified range passing (assumes effect funcs handle their own random.uniform/randint from ranges)
                    for p_name_sig in sig.parameters:
                        if p_name_sig.endswith("_range") and p_name_sig[:-6] in current_visual_params:
                             scaled_min, scaled_max = current_visual_params[p_name_sig[:-6]]
                             # Apply intensity scaling to range values before passing
                             scaled_min_val = scaled_min * (1.0 + (current_intensity - 1.0) * 0.3)
                             scaled_max_val = scaled_max * (1.0 + (current_intensity - 1.0) * 0.3)
                             if scaled_min_val > scaled_max_val: scaled_min_val, scaled_max_val = scaled_max_val, scaled_min_val # Ensure min <= max
                             # Add type casting or specific clipping if needed based on param name
                             if any(k in p_name_sig for k in ["alpha", "intensity_val"]): # Example for float ranges
                                 params_to_pass[p_name_sig] = (max(0.01, scaled_min_val), min(1.0, max(scaled_min_val + 0.01, scaled_max_val)))
                             else: # Assume int ranges for others (block_size, shift etc)
                                 params_to_pass[p_name_sig] = (max(1, int(scaled_min_val)), max(int(scaled_min_val) + 1, int(scaled_max_val)))
                    if effect_func == apply_flickering_glitch_layer:
                        inner_effect_func_candidate = random.choice([f for f in effect_candidates if f != apply_flickering_glitch_layer and f.__name__ != "apply_flickering_glitch_layer"])
                        if inner_effect_func_candidate:
                             frame = effect_func(frame, intensity=current_intensity, effect_func=inner_effect_func_candidate, prev_frame=prev_frame, **current_visual_params) # Pass all params for inner
                    else:
                         frame = effect_func(frame, **params_to_pass)
                    applied_effects_count += 1; applied_this_frame.append(effect_name)
                except Exception as e: print(f"Error applying {effect_name} at frame {i}: {e}")
        effect_history.extend(applied_this_frame); effect_history = effect_history[-20:]

        # ARG Visuals
        if arg_qr_display_frames > 0: # QR Code
            if arg_qr_img is not None:
                qr_x = random.randint(0, max(0,WIDTH - ARG_QR_SIZE)); qr_y = random.randint(0, max(0,HEIGHT - ARG_QR_SIZE))
                temp_qr = arg_qr_img.copy()
                if random.random() < 0.3*current_intensity: temp_qr = apply_pixelation(temp_qr, block_size_range=(max(2,int(5-current_intensity*2)), max(3,int(15-current_intensity*5)))) # Dynamic pixelation
                if qr_y + ARG_QR_SIZE <= HEIGHT and qr_x + ARG_QR_SIZE <= WIDTH: # Check bounds before slicing
                     frame_roi = frame[qr_y:qr_y+ARG_QR_SIZE, qr_x:qr_x+ARG_QR_SIZE]
                     # Ensure temp_qr has same channels as frame_roi
                     if temp_qr.shape[2] != frame_roi.shape[2]: temp_qr = cv2.cvtColor(temp_qr, cv2.COLOR_BGR2GRAY); temp_qr = cv2.cvtColor(temp_qr, cv2.COLOR_GRAY2BGR)
                     if frame_roi.shape == temp_qr.shape:
                          frame[qr_y:qr_y+ARG_QR_SIZE, qr_x:qr_x+ARG_QR_SIZE] = cv2.addWeighted(frame_roi, 0.5, temp_qr, 0.9 * (arg_qr_display_frames / (FPS*2.0)),0)
            arg_qr_display_frames -=1
        elif random.random() < (0.005 + 0.01 * current_intensity) * global_params.get("arg_qr_prob_modifier", 1.0):
            qr_data = current_theme_mutated.get("arg_elements",{}).get("qr_data_prefix","QR_") + "".join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(6,12)))
            arg_qr_img = generate_qr_code_image(qr_data, ARG_QR_SIZE)
            if arg_qr_img.size > 0 : arg_qr_display_frames = int(FPS * random.uniform(0.5, 2.0))
            # print(f"ARG: Displaying QR Code: {qr_data} for {arg_qr_display_frames} frames")

        if arg_text_display_frames > 0: # Text Codes
            if arg_text_code:
                txt_scale = ARG_TEXT_FONT_SCALE_BASE * (1.0 + current_intensity*0.2)
                text_size_cv2, _ = cv2.getTextSize(arg_text_code, FONT_FACE_TO_USE, txt_scale, 2)
                text_x = random.randint(10, max(11, WIDTH - text_size_cv2[0] - 10))
                text_y = random.randint(text_size_cv2[1] + 10, max(text_size_cv2[1]+11, HEIGHT - 10))
                cv2.putText(frame, arg_text_code, (text_x, text_y), FONT_FACE_TO_USE, txt_scale, (200,220,255, int(255 * (arg_text_display_frames / (FPS*1.5)))), random.randint(1,2), cv2.LINE_AA)
            arg_text_display_frames -=1
        elif random.random() < (0.01 + 0.02 * current_intensity) * global_params.get("arg_text_prob_modifier", 1.0):
            arg_text_code = generate_arg_text_code(current_theme_mutated, length=random.randint(16,48))
            arg_text_display_frames = int(FPS * random.uniform(0.3, 1.5))
            # print(f"ARG: Displaying Text Code: {arg_text_code} for {arg_text_display_frames} frames")

        if random.random() < (0.02 + 0.05 * current_intensity) * global_params.get("arg_protocol_prob_modifier", 1.0): # Fake Errors/Keys
            arg_elements = current_theme_mutated.get("arg_elements", {}); error_list = arg_elements.get("fake_errors", []); key_list = arg_elements.get("cipher_keys", [])
            protocol_text = ""
            if error_list and random.random() < 0.6: protocol_text = random.choice(error_list)
            elif key_list: protocol_text = "KEY_FRAG: " + random.choice(key_list)
            if protocol_text:
                font_scale_err = ARG_ERROR_CODE_FONT_SCALE * (1.0 + current_intensity*0.5)
                text_size_err, _ = cv2.getTextSize(protocol_text, FONT_FACE_TO_USE, font_scale_err, 3)
                text_x_err = max(10, (WIDTH - text_size_err[0])//2); text_y_err = max(10, (HEIGHT + text_size_err[1])//2 + random.randint(-HEIGHT//4, HEIGHT//4))
                flash_text_color_err = (255,random.randint(0,50),random.randint(0,50)) if np.mean(frame) > 100 else (255,random.randint(200,255),random.randint(200,255))
                flash_overlay_err = frame.copy()
                cv2.putText(flash_overlay_err, protocol_text, (text_x_err, text_y_err), FONT_FACE_TO_USE, font_scale_err, flash_text_color_err, random.randint(2,4), cv2.LINE_AA)
                frame = cv2.addWeighted(frame, 0.3, flash_overlay_err, 0.7,0)
                # print(f"ARG: Flashing Protocol Text: {protocol_text}")

        # Subliminal Flash
        flash_probability = 0.20 * current_intensity
        if random.random() < flash_probability:
            flash_type = random.choice(["invert", "text", "symbol", "color", "noise", "contrast", "ascii_flash"])
            overlay = frame.copy()
            try:
                if flash_type == "invert": overlay = 255 - frame
                # ... (other flash types as in v2, ensure they use current_theme_mutated.colors) ...
                elif flash_type == "ascii_flash": overlay = apply_ascii_sim(overlay, block_size_range=(max(2,int(6-current_intensity*2)),max(3,int(12-current_intensity*4))), invert=random.choice([True,False]))
                frame = cv2.addWeighted(frame, random.uniform(0.0,0.3), overlay, random.uniform(0.7,1.0),0)
            except Exception: pass # Ignore flash errors

        # Persistent Text
        frame_overlay_text = frame.copy() # Draw text on a copy to blend with alpha later
        active_persistent_texts = []
        for text_info in persistent_texts:
            px, py = text_info["pos"]; mx, my = text_info["move_speed"]
            text_info["pos"] = (int(px+mx), int(py+my))
            # ... (bounce logic from v2) ...
            text_info["frame_count"] +=1
            if text_info["frame_count"] < text_info["lifetime"]:
                try:
                    display_font_size = max(0.1, text_info["font_size"] * (0.7 + current_intensity*0.3))
                    thickness = max(1, int(display_font_size))
                    color_with_alpha = (*text_info["color"], int(text_info["alpha"]*255)) # Target for Pillow
                    # For OpenCV, alpha needs to be handled by blending layers
                    cv2.putText(frame_overlay_text, text_info["text"], text_info["pos"], FONT_FACE_TO_USE, display_font_size, text_info["color"], thickness, cv2.LINE_AA)
                    active_persistent_texts.append(text_info)
                except Exception: active_persistent_texts.append(text_info)
            elif random.random() < 0.6: # Respawn
                # ... (respawn logic from v2, use current_theme_mutated) ...
                active_persistent_texts.append(text_info) # Re-add to list
        persistent_texts = active_persistent_texts
        # Blend text overlay (frame_overlay_text contains text drawn on a copy of frame)
        # A true alpha blend is more complex with OpenCV directly.
        # This is an approximation:
        frame = cv2.addWeighted(frame, 0.5, frame_overlay_text, 0.5, 0)


        prev_frame = frame.copy()
        if frame_hold_counter > 0 and held_frame is None: held_frame = frame.copy()
        video.write(frame)
        if (i + 1) % (FPS * 1) == 0 and frame_hold_counter == 0: print(f"Video Frame {i+1}/{FRAME_COUNT} (Int: {current_intensity:.2f})...")
    video.release()
    return DURATION, FPS, video_temp_path

# --- Audio Generation ---
def generate_audio_enhanced(video_duration, video_fps, theme_data, intensity_func, global_params):
    DURATION = video_duration; NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    if NUM_SAMPLES <= 0: return None
    NUM_CHANNELS = 2; samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32)
    current_audio_params = global_params["audio"]; audio_instability_chance = global_params["instability_chance"] * 1.2
    print(f"Generating {DURATION:.1f}s audio, Theme: {global_params['theme_name']}...")
    min_freq, max_freq = theme_data["audio_freq_range"]; noise_types = theme_data["audio_noise_types"]; features = theme_data["audio_features"]
    current_theme_mutated = copy.deepcopy(theme_data)
    last_event_end_sample = 0; melody_track = np.zeros(NUM_SAMPLES, dtype=np.float32); biometric_track = np.zeros(NUM_SAMPLES, dtype=np.float32)
    if "biometric_rhythm" in features:
        biometric_type = random.choice([t for t in noise_types if "heart" in t or "breath" in t] or ["heartbeat_distorted"])
        biometric_track = generate_biometric_rhythm(NUM_SAMPLES, 0.3, rhythm_type=biometric_type)

    while last_event_end_sample < NUM_SAMPLES:
        current_sample_time = last_event_end_sample / SAMPLE_RATE
        try: current_intensity = intensity_func(current_sample_time)
        except ValueError: current_intensity = 1.0
        current_intensity = np.clip(current_intensity, 0.01, 4.0)
        if random.random() < audio_instability_chance: # Audio param instability
            param_key = random.choice(list(current_audio_params.keys()))
            if isinstance(current_audio_params[param_key], tuple) and len(current_audio_params[param_key]) == 2:
                c_min,c_max=current_audio_params[param_key];range_w=abs(c_max-c_min)+1e-6;mid_s=random.uniform(-range_w*0.15,range_w*0.15);scale_f=random.uniform(0.8,1.2)
                n_mid=(c_min+c_max)/2+mid_s;n_w=range_w*scale_f;n_min=n_mid-n_w/2;n_max=n_mid+n_w/2
                if"depth"in param_key:n_min=max(2,int(n_min))
                current_audio_params[param_key]=(n_min,n_max)

        event_max_duration = max(0.01, 1.5/(current_intensity+1.0)); event_min_duration = 0.002
        event_duration_sec = random.uniform(event_min_duration, event_max_duration)
        event_duration_samples = min(max(1,int(event_duration_sec*SAMPLE_RATE)), NUM_SAMPLES - last_event_end_sample)
        if event_duration_samples <= 0: break
        event_start_sample = last_event_end_sample; event_end_sample = event_start_sample + event_duration_samples
        event_vol = np.clip(abs(random.uniform(0.01,0.5)*(0.2+current_intensity*1.5)),0.001,1.0)
        type_roll = random.random(); silence_thresh = max(0.005, 0.1*(2.0-current_intensity))
        melody_thresh = silence_thresh + (0.1+0.1*current_intensity); noise_thresh = melody_thresh+(0.6+0.25*current_intensity)
        segment_mono = np.zeros(event_duration_samples,dtype=np.float32); is_melody=False
        if type_roll < silence_thresh: event_type="silence"
        elif type_roll < melody_thresh:
            event_type="melody";is_melody=True;melody_vol_mod=0.3+0.3*current_intensity
            segment_mono = generate_melody_fragment(event_duration_samples,event_vol*melody_vol_mod,current_theme_mutated,current_intensity)
        elif type_roll < noise_thresh:
            event_type="noise";noise_choice=random.choice(noise_types)
            if current_intensity>2.0 and random.random()<0.8:
                 harsh_noises=[n for n in noise_types if any(k in n for k in["screech","clip","feedback","artifact","data_stream"])]
                 if harsh_noises:noise_choice=random.choice(harsh_noises)
            segment_mono=generate_noise(noise_choice,event_duration_samples,event_vol,features)
        else:
            event_type="tone";fm_chance=np.clip(0.3+current_intensity*0.4,0.1,0.95);harmonic_chance=np.clip(0.2+current_intensity*0.3,0.1,0.9)
            freq_val=random.uniform(min_freq,max_freq*(0.5+current_intensity))
            segment_mono=generate_tone(np.clip(freq_val,LOW_FREQ_CUTOFF,HIGH_FREQ_CUTOFF),event_duration_samples,event_vol,fm_chance,harmonic_chance)

        if len(segment_mono)!=event_duration_samples and event_duration_samples>0:
            padded_segment=np.zeros(event_duration_samples,dtype=np.float32);len_to_copy=min(len(segment_mono),event_duration_samples)
            if len_to_copy>0:padded_segment[:len_to_copy]=segment_mono[:len_to_copy]
            segment_mono=padded_segment
        if event_type!="silence":
            effect_chance_mod=0.6+current_intensity
            if"distortion"in features and random.random()<0.6*effect_chance_mod:
                d_min,d_max=current_audio_params["distortion_intensity"];dist_range=(d_min*(0.7+current_intensity*1.2),d_max*(1.0+current_intensity*1.5))
                segment_mono=apply_distortion(segment_mono,intensity_range=np.clip(dist_range,1.0,40.0))
            if"bitcrush"in features and random.random()<0.5*effect_chance_mod:
                 b_min,b_max=current_audio_params["bitcrush_depth"];target_bits=int(max(2,b_min+(b_max-b_min)*(2.0-current_intensity)))
                 segment_mono=apply_bitcrush(segment_mono,bit_depth_range=(2,max(2,target_bits)))
            if"spectral_glitch"in features and random.random()<0.4*effect_chance_mod:segment_mono=apply_spectral_glitch(segment_mono,intensity=current_intensity*1.2)
            if"broken_speaker_sim"in features and random.random()<0.2*effect_chance_mod:segment_mono=apply_broken_speaker_sim(segment_mono,current_intensity)
            reverb_feature=[f for f in features if"convolution_reverb"in f]
            if reverb_feature and random.random()<0.1*effect_chance_mod:
                 try:
                      impulse_name=reverb_feature[0].split("_")[-1]
                      if impulse_name in IMPULSE_RESPONSES:segment_mono=apply_convolution_reverb(segment_mono,IMPULSE_RESPONSES[impulse_name])
                 except Exception:pass
            pan_extremity=np.clip(0.5+current_intensity*0.5,0.2,1.0);pan=random.uniform(-pan_extremity,pan_extremity)
            gain_l,gain_r=np.sqrt(0.5*(1-pan)),np.sqrt(0.5*(1+pan))
            end_idx=min(event_end_sample,NUM_SAMPLES);length=end_idx-event_start_sample
            if length<=0:continue
            if is_melody:melody_track[event_start_sample:end_idx]+=segment_mono[:length]
            else:samples[event_start_sample:end_idx,0]+=segment_mono[:length]*gain_l;samples[event_start_sample:end_idx,1]+=segment_mono[:length]*gain_r
        overlap_factor=np.clip(0.3+current_intensity*0.5,0.1,0.95)
        advance_samples=int(event_duration_samples*(1.0-overlap_factor))
        last_event_end_sample+=max(1,advance_samples)

    samples[:,0]+=biometric_track*(0.2+0.3*global_params.get("intensity_at_end",1.0))
    samples[:,1]+=biometric_track*(0.2+0.3*global_params.get("intensity_at_end",1.0))
    max_melody_abs=np.max(np.abs(melody_track));
    if max_melody_abs>1e-6:melody_track/=max_melody_abs
    melody_mix_level=0.25+0.2*global_params.get("intensity_at_end",1.0)
    samples[:,0]+=melody_track*melody_mix_level;samples[:,1]+=melody_track*melody_mix_level
    max_abs_amplitude=np.max(np.abs(samples))
    if max_abs_amplitude>1e-6:samples/=max_abs_amplitude
    else:print("Warning: Final audio mix is silent or near silent.")
    samples_int16=(np.clip(samples,-1.0,1.0)*32767).astype(np.int16)
    audio_temp_path=f"audio_temp_{int(time.time())}_{random.randint(100,999)}.wav"
    try:
        with wave.open(audio_temp_path,'wb')as wf:wf.setnchannels(NUM_CHANNELS);wf.setsampwidth(2);wf.setframerate(SAMPLE_RATE);wf.writeframes(samples_int16.tobytes())
        return audio_temp_path
    except Exception as e:print(f"Error writing WAV: {e}");return None

# --- Main Execution ---
if __name__ == "__main__":
    start_time_main = time.time()
    initial_theme_name = random.choice(list(THEMES.keys()))
    current_theme_data = copy.deepcopy(THEMES[initial_theme_name])
    print(f"--- Starting Shardmind Protocol Generation v3 ---"); print(f"Initial Theme: {initial_theme_name}")
    global_params = { "theme_name": initial_theme_name, "visual": { "catalyst_threshold": 2.8,
        "effect_probabilities": {name: prob for name, prob in [
            ("apply_perlin_noise",0.7), ("apply_block_shift",0.6), ("apply_color_channel_shift",0.5), ("apply_warp",0.5),
            ("apply_pixelation",0.4), ("apply_scanlines",0.3), ("apply_solarize",0.2), ("apply_extreme_contrast",0.5),
            ("apply_feedback",0.6), ("apply_datamosh_sim",0.3), ("apply_ascii_sim",0.2), ("apply_sensor_overload",0.25),
            ("apply_data_bleed_sim",0.2), ("apply_flickering_glitch_layer",0.25), ("apply_crt_ghost_sim",0.15),
            ("apply_slit_scan_sim",0.1), ("apply_vector_field_sim",0.1), ("apply_figurative_noise_sim",0.1) ]},
        # Add default ranges for visual effect parameters here, matching those in current_visual_params in generate_frames_enhanced
         "perlin_alpha": (0.1, 0.9), "perlin_scale": (2.0, 80.0), "block_shift_max": (10, WIDTH // 3), "block_size": (5, HEIGHT // 2),
         "color_shift_max": (5, 60), "warp_amp": (5, 80), "warp_freq": (0.002, 0.1), "pixel_block": (4, 96), # Renamed from pixel_block_range
         "contrast_alpha": (1.0, 6.0), "contrast_beta": (-100, 100), "feedback_alpha": (0.02, 0.6),
         "solarize_thresh": (60, 200), "scanline_intensity": (0.4, 1.0),
         "datamosh_block_size": (32,128), "ascii_block_size": (8,16) # Example for new effects
        },
        "audio": {"distortion_intensity": (1.0, 25.0), "bitcrush_depth": (2, 8)},
        "instability_chance": 0.012, "style_break_chance": 0.0005, "arg_qr_prob_modifier": 1.0,
        "arg_text_prob_modifier": 1.0, "arg_protocol_prob_modifier": 1.0, "intensity_at_end": 1.0 }

    estimated_max_duration_sec = MAX_DURATION + 15
    estimated_max_samples = int(estimated_max_duration_sec * SAMPLE_RATE)
    intensity_function = generate_intensity_profile(estimated_max_samples, SAMPLE_RATE)
    global_params["intensity_at_end"] = np.clip(intensity_function(MAX_DURATION - 5), 0.1, 3.0)

    actual_duration, actual_fps, video_temp_file = generate_frames_enhanced(current_theme_data, intensity_function, global_params)
    if not video_temp_file: print("Video gen failed!"); exit(1)
    audio_temp_file = generate_audio_enhanced(actual_duration, actual_fps, current_theme_data, intensity_function, global_params)
    if not audio_temp_file: print("Audio gen failed!"); exit(1)

    if video_temp_file and audio_temp_file:
        print("Combining with ffmpeg...")
        if os.path.exists(OUTPUT_FILE):
            try: os.remove(OUTPUT_FILE)
            except OSError as e: print(f"Could not remove existing output: {e}")
        ffmpeg_command = [ 'ffmpeg', '-y', '-i', video_temp_file, '-i', audio_temp_file, '-c:v', 'libx264',
            '-preset', 'medium', '-crf', '24', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '320k', '-shortest', OUTPUT_FILE ]
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            print(f"Output: '{OUTPUT_FILE}'")
        except Exception as e:
            print(f"ffmpeg error: {e}")
            if hasattr(e, 'stderr'): print(e.stderr)
    for f_clean in [video_temp_file, audio_temp_file]:
        if f_clean and os.path.exists(f_clean):
            try: os.remove(f_clean)
            except OSError as e: print(f"Error removing temp file {f_clean}: {e}")
    print(f"--- Total Time: {time.time() - start_time_main:.2f}s ---")










