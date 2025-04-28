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
from scipy.interpolate import interp1d # <<< Added for intensity profile
import subprocess # For ffmpeg call

# --- Configuration ---
WIDTH, HEIGHT = 640, 480
MIN_DURATION, MAX_DURATION = 20, 45 # Duration range in seconds
MIN_FPS, MAX_FPS = 2, 8 # EXTREMELY low FPS range for intense jerkiness
OUTPUT_FILE = "extreme_video.mp4"
HIGH_FREQ_CUTOFF = 20000 # Target high frequency limit (adjust based on hearing/equipment)
LOW_FREQ_CUTOFF = 20     # Target low frequency limit (lowered slightly)
SAMPLE_RATE = 44100      # Audio sample rate
time_step = 1.0 / SAMPLE_RATE # For audio calculations

# --- Thematic Elements ---
# Added more extreme symbols, audio types, and features
THEMES = {
    "Digital Decay EXTREME": {
        "colors": [
            [(0, 0, 0), (0, 255, 0), (255, 0, 0)], # Glitch Red/Green/Black
            [(255, 255, 255), (0, 0, 0), (100, 100, 100)], # Stark Contrast
            [(0, 0, 50), (0, 0, 255), (255, 255, 0)], # Corrupted Blue/Yellow
        ],
        "words": ["ERROR", "CORRUPT", "FRAGMENT", "NULL", "VOID", "DELETE", "SEGFAULT", "BUFFER", "OVERFLOW", "STATIC", "LOST", "SIGNAL", "404", "0xDEADBEEF", "Ŗ̸͈̪E̴̮͊S̷̩͂E̸̡ͮT̷͓ͨ", "U̸͖͒N̴̩ͥR̷͉ͮȆ̷̺Ȃ̶͜D̶̩̎A̸͚ͣB̷̻ͥL̴̯͘E̸̹͐", base64.b64encode(b"FEAR").decode(), binascii.hexlify(b"PAIN").decode()],
        "symbols": ["▌", "█", "░", "▒", "▓", "⚠️", "⚡", "☣️", "中断", "异常", "뷁", "௹", "", "//", "%", "&", "*", "!", "?", "01101100011011110111001101110100"], # Added more symbols, unicode chaos
        "audio_freq_range": (40, 16000), # Wider base range
        "audio_noise_types": ["white", "glitch", "static", "digital_artifact", "screech"], # Added harsher noises
        "audio_features": ["stutter", "bitcrush", "clicks", "extreme_panning", "distortion", "extreme_pitch"] # Added more intense features
    },
    "Organic Corruption EXTREME": {
        "colors": [
            [(50, 5, 5), (220, 40, 40), (50, 80, 40)], # Darker Fleshy Tones
            [(10, 20, 5), (40, 80, 20), (200, 200, 180)], # Deeper Decay
            [(0, 0, 0), (150, 0, 10), (255, 80, 80)], # Congealed Blood/Viscera
        ],
        "words": ["GROW", "DECAY", "CONSUME", "INFECT", "MUTATE", "FLESH", "BONE", "ROT", "SPORE", "INSIDE", "BREATHE", "PULSATE", "WRITHE", "SWELL", "BURST", "T̸̜̔H̸̻̋R̶̻͐O̵̺͒Ḇ̴͊", "內部", "곪다"], # Internal(Chinese), Fester(Korean)
        "symbols": ["Ø", "§", " संक्रमित", "تلوث", "🦠", "🍄", "🦴", "👁️‍🗨️", "〰", "~~", "...", "呼吸", "∬", "∯", "⌬", "⏳"], # Added more abstract/medical/decay symbols
        "audio_freq_range": (20, 1000), # Lower base range
        "audio_noise_types": ["brown", "squelch", "heartbeat", "breathing", "wet_clicks", "sub_bass"], # Added sub_bass and wet clicks
        "audio_features": ["wet_sounds", "slow_lfo", "dissonance", "low_rumble", "distortion", "extreme_pitch", "sub_bass_throb"]
    },
    "Cosmic Horror EXTREME": {
        "colors": [
            [(0, 0, 5), (50, 0, 80), (150, 150, 255)], # Colder Deep Space
            [(5, 0, 0), (0, 20, 20), (255, 100, 0)], # Starker Nebula
            [(0, 0, 0), (255, 255, 255), (5, 5, 5)], # Absolute Void
        ],
        "words": ["BEYOND", "VOID", "STARS", "MADNESS", "UNKNOWN", "ELDRITCH", "ENTITY", "SILENCE", "WATCHING", "BELOW", "ABYSS", "UNFOLD", "SEE", "NOTHING", "Ȉ̸̮Ț̴͘_̸̤͋I̶̻͋S̴̩̕_̴͕̽H̷̙̿E̸̻̅Ŕ̴͔E̵̥͌", "無", "深淵"], # Nothingness(Chinese), Abyss(Japanese)
        "symbols": ["★", "☆", "✶", "✡", "♮", "♄", "♃", "☉", "☿", "♁", "∝", "∫", "∇", "∞", "⋮", "⊙", "⊗", "∑", "Ж", "Ѫ"], # Added more symbols, some Cyrillic/obscure math
        "audio_freq_range": (15, HIGH_FREQ_CUTOFF + 2000), # VERY WIDE range potential
        "audio_noise_types": ["pink", "deep_drone", "radio_interference", "void_silence", "screech", "sub_bass", "static"],
        "audio_features": ["shepard_tone", "extreme_pitch", "heavy_reverb", "extreme_panning", "random_bursts", "distortion", "clipping_sim"]
    },
    "Subconscious Echoes EXTREME": {
        "colors": [
            [(80, 80, 100), (220, 220, 240), (30, 30, 50)], # Harsher Dreamlike
            [(0, 0, 0), (255, 50, 0), (0, 50, 255)], # More Violent Contrasts
            [(180, 180, 180), (50, 50, 50), (220, 220, 220)], # Starker Fog
        ],
        "words": ["REMEMBER", "FORGET", "DREAM", "LOOP", "ECHO", "MIRROR", "SHADOW", "WHO", "WHY", "LOST", "AGAIN", "TRAPPED", "FACE", "NEVER", "W̴̼͗Ḫ̴̏Á̸̤T̷̮̚_̸͎̂W̸̫͐Ȁ̷̙S̶̻̐", "잊다", "什么"], # Forget(Korean), What(Chinese)
        "symbols": ["?", "¿", " திரும்பத்திரும்ப", "繰り返す", "≡", "∽", "◌", "()", "...", "≠", "||", "⚰️", "🎭", "🗝️", "", "௸", "؟"], # Added mask, key, more obscure symbols
        "audio_freq_range": (80, 14000),
        "audio_noise_types": ["whispers", "filtered_noise", "reversed_sounds", "tape_hiss", "glitch", "digital_artifact"],
        "audio_features": ["delay", "reverb", "fading_melody", "extreme_panning", "stutter", "bitcrush", "distortion"]
    }
}

# --- Font Configuration (Optional) ---
FONT_PATHS = [
    # Linux/MacOS common paths (adjust if needed)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Arial.ttf",
     # Windows common paths (adjust if needed)
    "C:/Windows/Fonts/arial.ttf",
    # Add paths to specific unsettling or broad Unicode fonts if you have them
    # "NotoSans-Regular.ttf", # Example: Download Noto Sans from Google Fonts
    # "WEIRD_FONT.ttf",
]
AVAILABLE_FONTS = [f for f in FONT_PATHS if os.path.exists(f)]
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX # Fallback if no TTF fonts work
# TTF Font selection logic (currently unused in main drawing, but available)
SELECTED_TTF_FONT = random.choice(AVAILABLE_FONTS) if AVAILABLE_FONTS else None
print(f"Available TTF fonts found: {AVAILABLE_FONTS}")
print(f"Selected TTF font for potential use: {SELECTED_TTF_FONT}")


# Initialize pygame mixer
try:
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024)
except pygame.error as e:
    print(f"Pygame mixer init failed (might be normal in some envs, not essential for file output): {e}")


# --- NEW HELPER FUNCTION for Shared Intensity Profile ---
def generate_intensity_profile(duration_samples, sample_rate, scale=20.0, octaves=4, persistence=0.6, lacunarity=2.0):
    """Generates a Perlin noise-based intensity profile over time using 1D noise."""
    duration_seconds = duration_samples / sample_rate
    # Generate noise points slightly more frequently than needed for smooth interpolation
    num_points = int(duration_seconds * 10) # 10 points per second seems reasonable
    if num_points < 2: num_points = 2 # Need at least two points for interpolation
    profile = np.zeros(num_points)
    base = random.randint(0, 100) # Random offset for noise

    # Use 1D Perlin noise
    for i in range(num_points):
        profile[i] = noise.pnoise1(i / scale,
                                   octaves=octaves,
                                   persistence=persistence,
                                   lacunarity=lacunarity,
                                   base=base)

    # Normalize the profile: map it to a usable range (e.g., 0.2 to 1.5)
    min_val, max_val = np.min(profile), np.max(profile)
    if max_val - min_val > 1e-6: # Avoid division by zero if noise is flat
        normalized_profile = (profile - min_val) / (max_val - min_val) # Normalize to 0-1
        # Scale to desired output range, defining min/max intensity influence
        intensity_min = 0.2
        intensity_max = 1.5 # Allows intensity to boost parameters up to 150%
        scaled_profile = intensity_min + normalized_profile * (intensity_max - intensity_min)
    else:
        scaled_profile = np.full(num_points, 1.0) # Default intensity if noise is flat

    # Create an interpolation function for easy lookup at any time point
    profile_times = np.linspace(0, duration_seconds, num_points)
    # Use linear interpolation. 'bounds_error=False' prevents errors at edges,
    # 'fill_value' uses the edge values for times slightly outside the generated range.
    intensity_func = interp1d(profile_times, scaled_profile, kind='linear', bounds_error=False,
                              fill_value=(scaled_profile[0], scaled_profile[-1]))

    print(f"Generated shared intensity profile: {len(profile)} points over {duration_seconds:.1f}s")
    return intensity_func


# --- EXTREME Visual Glitch Functions ---

def apply_perlin_noise(frame, alpha_range=(0.1, 0.8), scale_range=(3.0, 60.0), oct_range=(2, 8)):
    """Applies more variable Perlin noise overlay."""
    alpha = random.uniform(*alpha_range)
    scale = random.uniform(*scale_range)
    octaves = random.randint(*oct_range)
    persistence = random.uniform(0.3, 0.8)
    lacunarity = random.uniform(1.5, 3.0)
    seed = random.randint(0, 1000)

    height, width = frame.shape[:2]
    gray_noise = np.zeros((height, width))

    # Generate Perlin noise values for each pixel
    # Vectorized approach might be faster but direct loop is clearer here
    for i in range(height):
        for j in range(width):
            gray_noise[i][j] = noise.pnoise2(i/scale, j/scale,
                                             octaves=octaves,
                                             persistence=persistence,
                                             lacunarity=lacunarity,
                                             base=seed)

    # Normalize noise to 0-255 and convert to BGR
    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX)
    colored_noise = cv2.cvtColor(colored_noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Random inversion for extra chaos
    if random.random() < 0.3:
        colored_noise = 255 - colored_noise

    return cv2.addWeighted(frame, 1 - alpha, colored_noise, alpha, 0)

def apply_block_shift(frame, max_shift_range=(20, WIDTH // 4), block_size_range=(10, HEIGHT // 3), num_blocks_range=(10, 50)):
    """Shifts more random blocks more drastically."""
    max_shift = random.randint(*max_shift_range)
    block_size_max = random.randint(*block_size_range)
    num_blocks = random.randint(*num_blocks_range)

    height, width = frame.shape[:2]
    temp_frame = frame.copy()

    for _ in range(num_blocks):
        bh = random.randint(5, block_size_max)
        bw = random.randint(5, block_size_max)
        # Ensure block doesn't go out of bounds during selection
        if height - bh <= 0 or width - bw <= 0: continue # Skip if block size is too large
        y = random.randint(0, height - bh -1) # Corrected upper bound
        x = random.randint(0, width - bw -1)  # Corrected upper bound

        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)

        # Calculate target top-left corner, clipping to frame boundaries
        target_y_start = np.clip(y + shift_y, 0, height - bh)
        target_x_start = np.clip(x + shift_x, 0, width - bw)

        # Extract the block
        block = frame[y:y+bh, x:x+bw].copy()
        # Place the block - potential for overlap/overwriting is part of the effect
        temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block

    return temp_frame

def apply_color_channel_shift(frame, max_shift_range=(10, 40)):
    """Shifts R, G, B channels more independently."""
    max_shift = random.randint(*max_shift_range)
    temp_frame = frame.copy()
    height, width = frame.shape[:2] # Get dimensions once

    for i in range(3): # Iterate through B, G, R channels
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        channel = frame[:,:,i]
        # Use np.roll for efficient circular shifting
        shifted_channel = np.roll(channel, shift_y, axis=0)
        shifted_channel = np.roll(shifted_channel, shift_x, axis=1)
        temp_frame[:,:,i] = shifted_channel
    return temp_frame

def apply_warp(frame, amplitude_range=(10, 40), freq_range=(0.005, 0.08)):
    """Applies a more intense wave warp distortion."""
    rows, cols = frame.shape[:2]
    img_output = np.zeros(frame.shape, dtype=frame.dtype)

    # Random parameters for the warp effect
    amplitude_x = random.uniform(*amplitude_range)
    frequency_x = random.uniform(*freq_range)
    amplitude_y = random.uniform(*amplitude_range) * random.uniform(0.5, 1.5) # Independent Y amplitude scaling
    frequency_y = random.uniform(*freq_range) * random.uniform(0.5, 1.5) # Independent Y frequency scaling
    phase_x = random.uniform(0, 2 * math.pi)
    phase_y = random.uniform(0, 2 * math.pi)

    # Create map arrays for remap (potentially faster than looping)
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(amplitude_x * math.sin(2 * math.pi * i * frequency_x + phase_x))
            offset_y = int(amplitude_y * math.cos(2 * math.pi * j * frequency_y + phase_y)) # Use cos for variation

            # Calculate source coordinates, clipping to boundaries
            src_x = np.clip(j + offset_x, 0, cols - 1)
            src_y = np.clip(i + offset_y, 0, rows - 1)

            map_x[i, j] = src_x
            map_y[i, j] = src_y

    # Apply the remapping
    img_output = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR) # Use linear interpolation

    return img_output

def apply_feedback(frame, prev_frame, alpha_range=(0.05, 0.3)):
    """Blends the current frame with a potentially modified previous frame."""
    if prev_frame is None:
        return frame # No feedback on the first frame
    alpha = random.uniform(*alpha_range)
    modified_prev = prev_frame

    # Optional: Add random slight transformation to the feedback frame
    if random.random() < 0.1:
        rows, cols = modified_prev.shape[:2]
        angle = random.uniform(-2, 2)
        scale = random.uniform(0.98, 1.02)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)
        modified_prev = cv2.warpAffine(modified_prev, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)

def apply_pixelation(frame, block_size_range=(8, 48)):
    """Applies a pixelation effect with variable block size."""
    height, width = frame.shape[:2]
    block_size = random.randint(*block_size_range)
    block_size = max(2, block_size) # Ensure block size is at least 2

    # Calculate target pixelated dimensions
    pixel_w, pixel_h = width // block_size, height // block_size
    # Ensure dimensions are at least 1x1
    pixel_w = max(1, pixel_w)
    pixel_h = max(1, pixel_h)

    # Resize down using nearest neighbor interpolation
    temp = cv2.resize(frame, (pixel_w, pixel_h), interpolation=cv2.INTER_NEAREST)
    # Resize back up to original size, also using nearest neighbor
    pixelated_frame = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_frame

def apply_scanlines(frame, intensity_range=(0.5, 0.9), thickness_range=(1, 3), color_variation=30):
    """Applies more prominent scanlines."""
    intensity = random.uniform(*intensity_range)
    thickness = random.randint(*thickness_range)
    temp_frame = np.zeros_like(frame) # Create a separate layer for scanlines
    height, width = frame.shape[:2]
    base_color_val = random.randint(0, 50) # Dark lines

    for y in range(0, height, thickness * 2): # Draw lines every other step
        line_color_val = base_color_val + random.randint(-color_variation, color_variation)
        line_color = tuple(np.clip([line_color_val]*3, 0, 255)) # Ensure color is valid BGR tuple
        cv2.line(temp_frame, (0, y), (width, y), line_color, thickness)

    # Blend the scanline layer over the original frame
    # Adjust gamma (the last parameter) to control blending darkness
    return cv2.addWeighted(frame, 1.0, temp_frame, intensity, -int(255 * intensity * 0.2)) # Less aggressive darkening

def apply_solarize(frame, threshold_range=(80, 180)):
    """Inverts pixels above a random threshold."""
    threshold = random.randint(*threshold_range)
    # Convert to grayscale to find the mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Invert pixels in the original frame where the mask is white
    solarized_frame = frame.copy()
    solarized_frame[mask == 255] = 255 - solarized_frame[mask == 255]
    return solarized_frame

def apply_extreme_contrast(frame, alpha_range=(1.5, 3.5), beta_range=(-60, 60)):
    """Applies very high contrast adjustments."""
    alpha = random.uniform(*alpha_range) # Contrast control
    beta = random.randint(*beta_range)   # Brightness control
    # Use convertScaleAbs for speed and automatic clipping to 0-255
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


# --- EXTREME Audio Effect/Generation Functions ---

def apply_distortion(data, intensity_range=(1.5, 5.0)):
    """Applies hard clipping distortion."""
    intensity = random.uniform(*intensity_range)
    # Clip audio samples, scaling by intensity first
    return np.clip(data * intensity, -0.98, 0.98) # Clip slightly below full scale

def apply_bitcrush(data, bit_depth_range=(4, 12)):
    """Simulates bitcrushing by quantizing audio samples."""
    bits = random.randint(*bit_depth_range)
    if bits >= 16: return data # No effect if bit depth is high
    # Calculate the number of quantization steps based on bit depth
    steps = 2**(bits - 1) # For signed values centered around zero
    # Quantize: scale to integer range, round, scale back to float range
    max_val = 1.0 # Assuming input data is already in [-1, 1]
    quantized = np.round(data * steps) / steps
    return quantized

def generate_tone(freq, duration_samples, vol, fm_chance=0.5, harmonic_chance=0.4):
    """Generates a tone with potential FM or harmonics and sharp envelope."""
    t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
    wave = np.zeros(duration_samples)

    # Add Frequency Modulation (FM)
    if random.random() < fm_chance:
        mod_freq = freq * random.uniform(0.1, 5.0) # Wider FM frequency ratio
        mod_depth = vol * random.uniform(1.0, 8.0) # Wider FM modulation depth (relative to vol)
        wave = np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))
    # Add Harmonics
    elif random.random() < harmonic_chance:
        harmonic_count = random.randint(1, 4)
        wave = np.sin(2 * np.pi * freq * t) # Fundamental frequency
        for h in range(2, harmonic_count + 2): # Add 2nd, 3rd, etc. harmonics
            harmonic_vol = random.uniform(0.1, 0.5) / h # Decrease volume for higher harmonics
            phase_shift = random.uniform(0, np.pi) # Random phase shift
            wave += harmonic_vol * np.sin(2 * np.pi * freq * h * t + phase_shift)
        # Basic normalization attempt for harmonics
        wave = wave / (1 + harmonic_count * 0.2)
    # Default: Simple Sine Wave
    else:
        wave = np.sin(2 * np.pi * freq * t)

    wave *= vol # Apply overall volume

    # Apply a sharp ADSR-like envelope
    attack_len = min(int(SAMPLE_RATE * 0.005), duration_samples // 3) # Very short attack
    decay_len = min(int(SAMPLE_RATE * random.uniform(0.05, 0.3)), duration_samples - attack_len) # Short decay
    sustain_level = random.uniform(0.1, 0.7)
    release_len = min(int(SAMPLE_RATE * random.uniform(0.05, 0.4)), duration_samples - attack_len - decay_len) # Variable release

    envelope = np.ones(duration_samples)
    if attack_len > 0:
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
    if decay_len > 0:
        envelope[attack_len:attack_len+decay_len] = np.linspace(1, sustain_level, decay_len)
    # Apply sustain level and exponential release tail
    sustain_start = attack_len + decay_len
    if release_len > 0 and sustain_start + release_len <= duration_samples:
         envelope[sustain_start:sustain_start+release_len] = sustain_level * np.exp(-np.linspace(0, 5, release_len))
         envelope[sustain_start+release_len:] = 0 # Ensure silence after release
    elif sustain_start < duration_samples: # If no distinct release phase fits, just hold sustain or start decay
         envelope[sustain_start:] = sustain_level * np.exp(-np.linspace(0, 5, duration_samples - sustain_start))


    return wave * envelope

def generate_noise(noise_type, duration_samples, vol, features=[]): # Pass features for context
    """Generates various types of noise."""
    if noise_type == "white":
        return vol * (2 * np.random.random(duration_samples) - 1)

    elif noise_type == "pink":
        # Simple approximation using filtered cumulative sum of white noise
        wn = 2 * np.random.random(duration_samples) - 1
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
        wn = 2 * np.random.random(duration_samples) - 1
        brown = np.cumsum(wn)
        # Normalize the Brownian noise
        max_abs = np.max(np.abs(brown))
        if max_abs > 1e-6: brown /= max_abs
        return vol * brown

    elif noise_type in ["glitch", "clicks", "wet_clicks"]:
        noise = np.zeros(duration_samples)
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
                t_click = np.arange(res_len) / SAMPLE_RATE
                click_env = np.exp(-np.linspace(0, 15, res_len)) # Faster decay
                click_wave = amp * np.sin(2 * np.pi * click_freq * t_click) * click_env
                end_pos = min(pos + res_len, duration_samples)
                actual_len = end_pos - pos
                if actual_len > 0:
                     noise[pos:end_pos] += click_wave[:actual_len] # Additive, careful not to clip excessively

        return noise # No extra vol multiplication, amp handled per click

    elif noise_type == "static":
        # Mix white noise with some filtering and crackle modulation
        wn = 2 * np.random.random(duration_samples) - 1
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
        noise = vol * (2 * np.random.random(duration_samples) - 1)
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
        mod_depth = vol * random.uniform(5, 20) # Deep modulation
        t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
        wave = np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))
        # Add a bit of white noise for harshness
        wave += 0.3 * (2 * np.random.random(duration_samples) - 1)
        # Normalize roughly to prevent excessive clipping
        max_abs = np.max(np.abs(wave))
        if max_abs > 1e-6: wave /= max_abs
        return vol * wave

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
        return vol * wave

    # --- Theme Specific Noises (Examples) ---
    elif noise_type == "heartbeat":
         # Simulate heartbeat rhythm with low-frequency thuds
         bpm = random.uniform(40, 100) # Beats per minute
         bps = bpm / 60.0
         beat_interval_samples = int(SAMPLE_RATE / bps)
         thud_len = int(SAMPLE_RATE * 0.08) # Short thud sound
         thud_freq = random.uniform(40, 80)
         t_thud = np.linspace(0, thud_len * time_step, thud_len, endpoint=False)
         thud_env = np.exp(-np.linspace(0, 8, thud_len)) # Exponential decay
         thud_sound = np.sin(2*np.pi*thud_freq*t_thud) * thud_env

         noise = np.zeros(duration_samples)
         current_sample = 0
         while current_sample + thud_len < duration_samples:
             noise[current_sample : current_sample + thud_len] += thud_sound * vol
             # Add a smaller second beat?
             if random.random() < 0.7:
                 second_beat_offset = int(beat_interval_samples * random.uniform(0.3, 0.5))
                 if current_sample + second_beat_offset + thud_len < duration_samples:
                     noise[current_sample + second_beat_offset : current_sample + second_beat_offset + thud_len] += thud_sound * vol * 0.6
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
        alpha = 0.1 + filter_mod * 0.8 # Vary filter coefficient (0.1 to 0.9)

        filtered_breath = np.zeros_like(base_noise)
        for n in range(1, duration_samples):
             # Simple time-varying low-pass filter
             current_alpha = np.clip(0.1 + filter_mod[n] * 0.8, 0.01, 0.99) # Update alpha per sample
             filtered_breath[n] = current_alpha * filtered_breath[n-1] + (1 - current_alpha) * base_noise[n]

        return vol * filtered_breath * amp_mod

    # Add other specific noise types here (squelch, whispers, etc.) as needed

    else: # Default fallback to white noise
        print(f"Warning: Unknown noise type '{noise_type}', using white noise.")
        return vol * (2 * np.random.random(duration_samples) - 1)


# --- EXTREME Frame Generation (Enhanced with Intensity & Instability) ---
def generate_frames_enhanced(theme_data, intensity_func):
    """Generates video frames, influenced by intensity and with parameter instability."""
    DURATION = random.randint(MIN_DURATION, MAX_DURATION)
    FPS = random.randint(MIN_FPS, MAX_FPS)
    FRAME_COUNT = DURATION * FPS

    # --- Parameter Instability Tracking ---
    # Store current ranges/probabilities that *might* change during generation
    # Initialized with reasonable defaults, possibly derived from config or theme later
    current_params = {
        "perlin_alpha": (0.1, 0.8), "perlin_scale": (3.0, 60.0),
        "block_shift_max": (20, WIDTH // 4), "block_size": (10, HEIGHT // 3),
        "color_shift_max": (10, 40),
        "warp_amp": (10, 40), "warp_freq": (0.005, 0.08),
        "pixel_block": (8, 48),
        "contrast_alpha": (1.5, 3.5), "contrast_beta": (-60, 60),
        "feedback_alpha": (0.1, 0.4), # Base range for feedback blend
        "solarize_thresh": (80, 180),
        "scanline_intensity": (0.5, 0.9),
        "effect_probabilities": { # Store base probabilities for each effect
             apply_perlin_noise: 0.7,
             apply_block_shift: 0.6,
             apply_color_channel_shift: 0.6,
             apply_warp: 0.5,
             apply_pixelation: 0.4,
             apply_scanlines: 0.3,
             apply_solarize: 0.2,
             apply_extreme_contrast: 0.3,
             apply_feedback: 0.5 # Base chance for feedback loop
         }
    }
    instability_chance = 0.005 # Small chance per frame to randomly change a parameter range/probability

    print(f"Generating EXTREME {DURATION}-second video ({FPS} FPS) with theme: {current_theme_name}...")
    print(f"Using shared intensity profile and parameter instability (chance: {instability_chance:.3f}).")

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Codec
    video_temp_path = "video_temp.mp4"
    video = cv2.VideoWriter(video_temp_path, fourcc, FPS, (WIDTH, HEIGHT))

    prev_frame = None
    frame_hold_counter = 0
    held_frame = None

    # Persistent texts setup (more numerous and erratic)
    persistent_texts = []
    for _ in range(random.randint(3, 8)): # More persistent elements
        word = random.choice(theme_data["words"])
        symbol = random.choice(theme_data["symbols"])
        # Randomly encode some words for extra visual noise
        if random.random() < 0.15: word = base64.b64encode(word.encode()).decode()
        elif random.random() < 0.10: word = binascii.hexlify(word.encode()).decode()

        persistent_texts.append({
            "text": word + symbol,
            "pos": (random.randint(10, WIDTH - 100), random.randint(20, HEIGHT - 20)),
            "font_size": random.uniform(0.4, 3.0), # Wider size range
            "color": tuple(random.randint(0, 255) for _ in range(3)),
            # "angle": random.uniform(-90, 90), # Rotation adds complexity, omit for now
            "lifetime": random.randint(int(FPS * 0.2), int(FPS * 6)), # More variance
            "frame_count": 0,
            "move_speed": (random.uniform(-4, 4), random.uniform(-3, 3)) # Faster movement
        })

    # --- Main Frame Generation Loop ---
    for i in range(FRAME_COUNT):
        current_time_sec = i / FPS

        # --- Get Intensity from Shared Profile ---
        current_intensity = intensity_func(current_time_sec)
        # Clamp intensity to a reasonable multiplier range (e.g., 0.1x to 2.0x)
        current_intensity = np.clip(current_intensity, 0.1, 2.0)

        # --- Parameter Instability Check ---
        if random.random() < instability_chance:
            param_key_to_change = random.choice(list(current_params.keys()))

            if param_key_to_change == "effect_probabilities":
                # Slightly alter one effect's base probability
                effect_func = random.choice(list(current_params["effect_probabilities"].keys()))
                old_prob = current_params["effect_probabilities"][effect_func]
                # Change probability by a random amount, keeping it within bounds
                new_prob = np.clip(old_prob + random.uniform(-0.2, 0.2), 0.05, 0.95)
                current_params["effect_probabilities"][effect_func] = new_prob
                # print(f"INSTABILITY: Changed {effect_func.__name__} probability to {new_prob:.2f} at {current_time_sec:.1f}s") # Optional Debug
            elif isinstance(current_params[param_key_to_change], tuple) and len(current_params[param_key_to_change]) == 2:
                # If it's a range tuple (min, max), slightly alter it
                current_min, current_max = current_params[param_key_to_change]
                range_width = current_max - current_min
                # Shift the midpoint slightly
                mid_shift = random.uniform(-range_width * 0.1, range_width * 0.1)
                # Scale the range width slightly
                scale_factor = random.uniform(0.9, 1.1)
                new_mid = (current_min + current_max) / 2 + mid_shift
                new_width = range_width * scale_factor
                new_min = new_mid - new_width / 2
                new_max = new_mid + new_width / 2
                # Ensure min < max and apply reasonable bounds if necessary (example for alpha)
                if "alpha" in param_key_to_change:
                     new_min = max(0.01, new_min)
                     new_max = max(new_min + 0.1, new_max) # Ensure some range
                # Add similar bounds checks for other params (e.g., shift ranges)
                current_params[param_key_to_change] = (new_min, new_max)
                # print(f"INSTABILITY: Changed {param_key_to_change} range to ({new_min:.2f}, {new_max:.2f}) at {current_time_sec:.1f}s") # Optional Debug
            # Add logic for other types of parameters if needed

        # --- Frame stutter/hold simulation ---
        if frame_hold_counter > 0:
            if held_frame is not None:
                video.write(held_frame)
                frame_hold_counter -= 1
                if (i + 1) % (FPS * 5) == 0: print(f"Generated { (i + 1) / FPS :.1f} seconds of video (holding)...")
                continue # Skip generating a new frame
            else: # Reset if held_frame is somehow None (shouldn't happen)
                frame_hold_counter = 0

        # Chance to hold the *next* frame to be generated
        if random.random() < 0.02 and frame_hold_counter == 0: # Only trigger if not already holding
            frame_hold_counter = random.randint(1, int(FPS * 0.5)) # Hold for up to half a second
            held_frame = None # Clear previous held frame


        # --- Base frame ---
        if random.random() < 0.4: # Solid color background sometimes
            bg_color = random.choice(random.choice(theme_data["colors"]))
            frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
        else: # Black background otherwise
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # --- Apply Layers of EXTREME Effects (Influenced by Intensity) ---
        effect_stack = list(current_params["effect_probabilities"].keys())
        random.shuffle(effect_stack)

        # Determine number of effects to apply, influenced by intensity
        num_base_effects = random.randint(1, 3) # Minimum number of effects
        num_intensity_effects = int(current_intensity * 1.5) # More intensity = more effects attempted
        num_effects_to_apply = min(len(effect_stack), num_base_effects + num_intensity_effects)

        applied_effects_count = 0
        for effect_func in effect_stack:
            if applied_effects_count >= num_effects_to_apply:
                break # Stop if we've applied enough effects

            # Get current base probability for this effect
            base_probability = current_params["effect_probabilities"].get(effect_func, 0.5)
            # Intensity boosts the chance of applying the effect
            # Modifier scales from ~0.5 (low intensity) to ~1.5 (high intensity) around 1.0
            probability_modifier = 0.5 + current_intensity * 0.5
            final_probability = np.clip(base_probability * probability_modifier, 0.0, 1.0)

            if random.random() < final_probability:
                try:
                    # --- Apply effect, scaling parameters by current_intensity ---
                    # Retrieve current parameter ranges from the (potentially unstable) current_params dict
                    # Apply intensity scaling to the *ranges* before picking a value
                    if effect_func == apply_perlin_noise:
                        alpha_range = (current_params["perlin_alpha"][0],
                                       current_params["perlin_alpha"][1] * current_intensity)
                        scale_range = (current_params["perlin_scale"][0],
                                       current_params["perlin_scale"][1] * current_intensity)
                        frame = apply_perlin_noise(frame, alpha_range=np.clip(alpha_range, 0.01, 1.0),
                                                   scale_range=np.clip(scale_range, 2.0, 100.0))
                    elif effect_func == apply_block_shift:
                        shift_range = (int(current_params["block_shift_max"][0] * current_intensity),
                                       int(current_params["block_shift_max"][1] * current_intensity))
                        block_range = (int(current_params["block_size"][0]),
                                       int(current_params["block_size"][1] * current_intensity))
                        frame = apply_block_shift(frame, max_shift_range=np.clip(shift_range, 5, WIDTH//2),
                                                  block_size_range=np.clip(block_range, 5, HEIGHT//2))
                    elif effect_func == apply_color_channel_shift:
                        shift_range = (int(current_params["color_shift_max"][0]),
                                       int(current_params["color_shift_max"][1] * current_intensity))
                        frame = apply_color_channel_shift(frame, max_shift_range=np.clip(shift_range, 1, 100))
                    elif effect_func == apply_warp:
                        amp_range = (current_params["warp_amp"][0] * current_intensity,
                                     current_params["warp_amp"][1] * current_intensity)
                        frame = apply_warp(frame, amplitude_range=np.clip(amp_range, 1, 100))
                    elif effect_func == apply_pixelation:
                        block_range = (int(current_params["pixel_block"][0]),
                                       int(current_params["pixel_block"][1] * current_intensity))
                        frame = apply_pixelation(frame, block_size_range=np.clip(block_range, 2, 100))
                    elif effect_func == apply_scanlines:
                         intensity_range = (current_params["scanline_intensity"][0],
                                            current_params["scanline_intensity"][1] * current_intensity)
                         frame = apply_scanlines(frame, intensity_range=np.clip(intensity_range, 0.1, 1.0))
                    elif effect_func == apply_solarize:
                         frame = apply_solarize(frame, threshold_range=current_params["solarize_thresh"]) # No intensity scaling here?
                    elif effect_func == apply_extreme_contrast:
                        alpha_range = (current_params["contrast_alpha"][0] * current_intensity,
                                       current_params["contrast_alpha"][1] * current_intensity)
                        frame = apply_extreme_contrast(frame, alpha_range=np.clip(alpha_range, 0.5, 10.0),
                                                      beta_range=current_params["contrast_beta"])
                    elif effect_func == apply_feedback:
                        alpha_range = (current_params["feedback_alpha"][0] * current_intensity,
                                       current_params["feedback_alpha"][1] * current_intensity)
                        frame = apply_feedback(frame, prev_frame, alpha_range=np.clip(alpha_range, 0.01, 0.8))
                    else:
                        # Fallback for any effects not explicitly handled above
                        frame = effect_func(frame)

                    applied_effects_count += 1
                except Exception as e:
                    print(f"Error applying {effect_func.__name__} at frame {i}: {e}")


        # --- Subliminal Flash (Maybe also scale probability/intensity?) ---
        flash_probability = 0.12 * current_intensity # More flashes when intense
        if random.random() < flash_probability:
            flash_type = random.choice(["invert", "text", "symbol", "color", "noise", "contrast"])
            overlay = frame.copy() # Work on a copy for the flash
            try:
                if flash_type == "invert":
                    overlay = 255 - frame
                elif flash_type == "color":
                    flash_col = random.choice(random.choice(theme_data["colors"]))
                    overlay = np.full((HEIGHT, WIDTH, 3), flash_col, dtype=np.uint8)
                elif flash_type == "noise":
                    noise_overlay = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
                    overlay = cv2.addWeighted(frame, 0.3, noise_overlay, 0.7, 0)
                elif flash_type == "contrast":
                     overlay = apply_extreme_contrast(frame, alpha_range=(3.0 * current_intensity, 6.0 * current_intensity), beta_range=(-80, 80))
                else: # text or symbol
                    sub_text = random.choice(theme_data["words"]) if flash_type == "text" else random.choice(theme_data["symbols"])
                    font_scale = random.uniform(4.0, 8.0) * (0.5 + current_intensity * 0.5) # Scale size with intensity
                    thickness = random.randint(4, 8)
                    text_size, _ = cv2.getTextSize(sub_text, DEFAULT_FONT, font_scale, thickness)
                    text_x = max(0, (WIDTH - text_size[0]) // 2)
                    text_y = max(0, (HEIGHT + text_size[1]) // 2)
                    flash_col = tuple(random.randint(0, 255) for _ in range(3))
                    # Add outline/shadow for visibility
                    cv2.putText(overlay, sub_text, (text_x+3, text_y+3), DEFAULT_FONT, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
                    cv2.putText(overlay, sub_text, (text_x, text_y), DEFAULT_FONT, font_scale, flash_col, thickness, cv2.LINE_AA)

                # Blend the flash frame briefly and strongly
                frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)
            except Exception as e:
                 print(f"Error during subliminal flash ({flash_type}): {e}") # Catch potential rendering errors


        # --- Persistent and Fleeting Text ---
        current_persistent_texts = []
        frame_overlay = frame.copy() # Draw text on an overlay to manage alpha blending later
        # Update and draw persistent texts
        for text_info in persistent_texts:
            # Update position
            px, py = text_info["pos"]
            mx, my = text_info["move_speed"]
            text_info["pos"] = (int(px + mx), int(py + my))

            # More erratic bounce/wrap behavior
            pos_x, pos_y = text_info["pos"]
            text_width_estimate = int(text_info["font_size"] * 20 * len(text_info["text"])) # Rough estimate
            text_height_estimate = int(text_info["font_size"] * 40)
            if not (0 < pos_x < WIDTH - text_width_estimate):
                text_info["move_speed"] = (-mx * random.uniform(0.8, 1.2), my + random.uniform(-1, 1))
                if random.random() < 0.1: text_info["pos"] = (random.randint(10, WIDTH-100), pos_y) # Random reposition X
            if not (text_height_estimate < pos_y < HEIGHT - text_height_estimate):
                 text_info["move_speed"] = (mx + random.uniform(-1, 1), -my * random.uniform(0.8, 1.2))
                 if random.random() < 0.1: text_info["pos"] = (pos_x, random.randint(20, HEIGHT-20)) # Random reposition Y

            # Clip position just in case
            text_info["pos"] = (np.clip(text_info["pos"][0], 0, WIDTH - 50), np.clip(text_info["pos"][1], 20, HEIGHT - 20))
            text_info["frame_count"] += 1

            if text_info["frame_count"] < text_info["lifetime"]:
                try:
                    font_face = DEFAULT_FONT
                    # Scale font size slightly with intensity?
                    display_font_size = text_info["font_size"] * (0.8 + current_intensity * 0.2)
                    cv2.putText(frame_overlay, text_info["text"], text_info["pos"], font_face,
                                display_font_size, text_info["color"], random.randint(1, 4), cv2.LINE_AA)
                    current_persistent_texts.append(text_info)
                except Exception as e:
                    # print(f"Error drawing persistent text: {e}") # Avoid spamming logs
                    current_persistent_texts.append(text_info) # Keep it anyway
            else: # Lifetime expired
                if random.random() < 0.5: # Higher chance to respawn
                    text_info["pos"] = (random.randint(10, WIDTH - 100), random.randint(20, HEIGHT - 20))
                    text_info["frame_count"] = 0
                    text_info["lifetime"] = random.randint(int(FPS * 0.2), int(FPS * 6))
                    # Change text on respawn
                    word = random.choice(theme_data["words"])
                    symbol = random.choice(theme_data["symbols"])
                    if random.random() < 0.1: word = base64.b64encode(word.encode()).decode() # Re-encode sometimes
                    text_info["text"] = word + symbol
                    current_persistent_texts.append(text_info)

        persistent_texts = current_persistent_texts

        # Add more fleeting text, potentially more when intense
        if random.random() > 0.3 / max(0.5, current_intensity): # More likely when intense
            num_fleet = random.randint(1, int(2 + current_intensity * 2)) # Multiple fleeting texts
            for _ in range(num_fleet):
                fleet_text = random.choice(theme_data["words"]) + " " + random.choice(theme_data["symbols"])
                if random.random() < 0.1: fleet_text = base64.b64encode(fleet_text.encode()).decode()
                font_size = random.uniform(0.3, 1.8) * (0.8 + current_intensity * 0.2)
                position = (random.randint(5, WIDTH - 80), random.randint(10, HEIGHT - 10))
                text_color = tuple(random.randint(0, 255) for _ in range(3))
                thickness = random.randint(1, 3)
                try:
                    cv2.putText(frame_overlay, fleet_text, position, DEFAULT_FONT, font_size, text_color, thickness, cv2.LINE_AA)
                except Exception as e:
                    pass # Ignore errors for fleeting text

        # Blend text overlay (adjust alpha for visibility)
        text_alpha = 0.9 # How opaque the text overlay is
        frame = cv2.addWeighted(frame, 1.0, frame_overlay, text_alpha, 0)


        # --- Store frame for feedback loop BEFORE potential hold ---
        prev_frame = frame.copy()

        # --- If this frame is designated to be held, store it ---
        if frame_hold_counter > 0 and held_frame is None:
            held_frame = frame.copy() # Store the final frame state

        # --- Write frame to video ---
        video.write(frame)

        # --- Progress indicator ---
        if (i + 1) % (FPS * 5) == 0 and frame_hold_counter == 0: # Update less often, only when not holding
            print(f"Generated { (i + 1) / FPS :.1f} seconds of video (Intensity: {current_intensity:.2f})...")


    video.release()
    print("Video generation complete.")
    return DURATION, FPS, video_temp_path # Return duration, FPS, and temp file path


# --- EXTREME Audio Generation (Enhanced with Intensity & Instability) ---
def generate_audio_enhanced(video_duration, video_fps, theme_data, intensity_func):
    """Generates EXTREME, dynamic stereo audio linked to intensity profile."""
    DURATION = video_duration
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    NUM_CHANNELS = 2

    samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32)

    # --- Audio Elements from Theme ---
    min_freq, max_freq = theme_data["audio_freq_range"]
    noise_types = theme_data["audio_noise_types"]
    features = theme_data["audio_features"] # Used by generate_noise/tone implicitly or explicitly

    # --- Parameter Instability (Audio - Optional, less developed) ---
    current_audio_params = {
         "distortion_intensity": (1.5, 5.0),
         "bitcrush_depth": (4, 12),
         # Add other audio parameter ranges if needed (e.g., panning range, reverb time)
    }
    audio_instability_chance = 0.001 # Lower chance for audio parameter changes?

    print(f"Generating {DURATION:.1f}s of audio linked to intensity profile...")

    # --- Generate Audio Event by Event (Influenced by Intensity) ---
    last_event_end_sample = 0
    while last_event_end_sample < NUM_SAMPLES:

        current_sample_time = last_event_end_sample / SAMPLE_RATE
        # --- Get Intensity for this audio segment's start time ---
        current_intensity = intensity_func(current_sample_time)
        current_intensity = np.clip(current_intensity, 0.1, 2.0) # Clamp intensity

        # --- Optional: Audio Parameter Instability ---
        if random.random() < audio_instability_chance:
             # Add logic here to modify current_audio_params ranges if desired
             pass

        # --- Determine Event Properties based on Intensity ---
        # Intensity affects duration, volume, type choice, effect chance/amount

        # Shorter, potentially overlapping events when intensity is high
        event_max_duration = max(0.05, 1.5 / (current_intensity + 0.5)) # Inverse relationship
        event_min_duration = 0.01
        event_duration_sec = random.uniform(event_min_duration, event_max_duration)
        event_duration_samples = int(event_duration_sec * SAMPLE_RATE)

        # Ensure event doesn't overshoot total duration significantly
        event_duration_samples = min(event_duration_samples, NUM_SAMPLES - last_event_end_sample)
        if event_duration_samples <= 1: break # Stop if no room left for meaningful event

        event_start_sample = last_event_end_sample
        event_end_sample = event_start_sample + event_duration_samples

        # Higher base volume and higher peaks possible when intensity is high
        base_vol = random.uniform(0.05, 0.3)
        event_vol = np.clip(base_vol * (0.5 + current_intensity), 0.01, 0.95) # Scale volume, clip

        # Choose event type (tone, noise, or silence), biasing based on intensity
        type_roll = random.random()
        silence_thresh = 0.1 * (1.5 - current_intensity) # More silence when intensity low
        noise_thresh = 0.5 + 0.3 * current_intensity # More noise when intensity high

        if type_roll < silence_thresh:
            event_type = "silence"
        elif type_roll < noise_thresh:
            event_type = "noise"
        else:
            event_type = "tone"

        segment = np.zeros((event_duration_samples, NUM_CHANNELS), dtype=np.float32)
        segment_mono = np.zeros(event_duration_samples, dtype=np.float32)

        if event_type == "tone":
            # Intensity can push frequency higher or lower depending on theme? For now, just wider range.
            current_max_freq = max_freq * (0.8 + current_intensity * 0.4) # Scale max freq
            freq = random.uniform(min_freq, current_max_freq)
            freq = np.clip(freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF) # Clamp to absolute limits
            segment_mono = generate_tone(freq, event_duration_samples, event_vol)
        elif event_type == "noise":
            # Bias towards harsher noise types with higher intensity
            noise_choice = random.choice(noise_types)
            if current_intensity > 1.3 and "screech" in noise_types and random.random() < 0.4:
                 noise_choice = "screech"
            elif current_intensity > 1.2 and "digital_artifact" in noise_types and random.random() < 0.3:
                 noise_choice = "digital_artifact"
            elif current_intensity > 1.1 and "glitch" in noise_types and random.random() < 0.2:
                 noise_choice = "glitch"

            segment_mono = generate_noise(noise_choice, event_duration_samples, event_vol, features)
        # else: silence (segment_mono remains zeros)

        if event_type != "silence":
             # --- Apply Audio Effects (Influenced by Intensity) ---
             effect_chance_mod = current_intensity # Use intensity directly as modifier for effect chance

             if "distortion" in features and random.random() < 0.4 * effect_chance_mod:
                 dist_range = (current_audio_params["distortion_intensity"][0] * current_intensity,
                               current_audio_params["distortion_intensity"][1] * current_intensity)
                 segment_mono = apply_distortion(segment_mono, intensity_range=np.clip(dist_range, 1.0, 15.0)) # Clip range

             if "bitcrush" in features and random.random() < 0.3 * effect_chance_mod:
                  # More crushing (lower bit depth) when intense
                  max_bits = current_audio_params["bitcrush_depth"][1]
                  min_bits = current_audio_params["bitcrush_depth"][0]
                  # Calculate target bits, inversely related to intensity
                  target_bits = int(max(2, min_bits + (max_bits - min_bits) * (1.5 - current_intensity)))
                  target_bits = max(2, min(max_bits, target_bits)) # Clamp within allowed range
                  segment_mono = apply_bitcrush(segment_mono, bit_depth_range=(min_bits, target_bits))

             # Add other intensity-influenced effects here (stutter, pitch shift, reverb amount, etc.)

             # --- Panning (Influenced by Intensity) ---
             if "extreme_panning" in features:
                  # Panning range widens with intensity
                  pan_extremity = np.clip(0.5 * current_intensity, 0.1, 1.0) # Scale from 0.1 to 1.0
                  pan = random.uniform(-pan_extremity, pan_extremity) # -1 = Left, 0 = Center, 1 = Right
             else:
                  pan = random.uniform(-0.7, 0.7) # Default moderate panning

             # Apply panning using square root law for constant power
             gain_l = np.sqrt(0.5 * (1 - pan))
             gain_r = np.sqrt(0.5 * (1 + pan))
             segment[:, 0] = segment_mono * gain_l
             segment[:, 1] = segment_mono * gain_r

        # Add the generated segment to the main samples array
        # Use additive blending which might cause clipping but increases density/chaos
        # Ensure indices are within bounds
        end_idx = min(event_end_sample, NUM_SAMPLES)
        samples[event_start_sample:end_idx, :] += segment[:end_idx-event_start_sample, :]


        # Move time forward: allow slight overlap for density when intense
        overlap_factor = np.clip(0.1 * current_intensity, 0, 0.5) # Overlap up to 50% when very intense
        advance_samples = int(event_duration_samples * (1.0 - overlap_factor))
        last_event_end_sample += max(1, advance_samples) # Ensure progress

        # Optional Progress Indicator
        # if last_event_end_sample % (SAMPLE_RATE * 10) < advance_samples:
        #     print(f"Generated {last_event_end_sample / SAMPLE_RATE :.1f}s of audio (Intensity: {current_intensity:.2f})...")


    # --- Final Normalization ---
    max_abs_amplitude = np.max(np.abs(samples))
    if max_abs_amplitude > 1e-6:
        print(f"Normalizing audio (Max amplitude before: {max_abs_amplitude:.3f})")
        samples /= max_abs_amplitude
    else:
        print("Warning: Audio appears silent.")

    # Convert to 16-bit integers for WAV export
    samples_int16 = (samples * 32767).astype(np.int16)

    # --- Write to WAV file ---
    audio_temp_path = "audio_temp.wav"
    try:
        with wave.open(audio_temp_path, 'wb') as wf:
            wf.setnchannels(NUM_CHANNELS)
            wf.setsampwidth(2) # 2 bytes for 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples_int16.tobytes())
        print(f"Temporary audio saved to {audio_temp_path}")
        return audio_temp_path
    except Exception as e:
        print(f"Error writing WAV file: {e}")
        return None


# --- Main Execution Logic ---
if __name__ == "__main__":
    # 1. Choose Theme
    current_theme_name = random.choice(list(THEMES.keys()))
    # Or select manually:
    # current_theme_name = "Cosmic Horror EXTREME"
    theme_data = THEMES[current_theme_name]
    print(f"--- Starting EXTREME Generation ---")
    print(f"Selected Theme: {current_theme_name}")

    # 2. Generate Shared Intensity Profile
    # Estimate max duration for profile generation, add buffer
    estimated_max_duration_sec = MAX_DURATION + 5
    estimated_max_samples = int(estimated_max_duration_sec * SAMPLE_RATE)
    intensity_function = generate_intensity_profile(estimated_max_samples, SAMPLE_RATE)

    # 3. Generate Video Frames (using intensity function)
    actual_duration, actual_fps, video_temp_file = generate_frames_enhanced(theme_data, intensity_function)
    print(f"Video generation yielded: Duration={actual_duration}s, FPS={actual_fps}")

    # 4. Generate Audio (using intensity function and actual video duration/fps)
    audio_temp_file = generate_audio_enhanced(actual_duration, actual_fps, theme_data, intensity_function)

    # 5. Combine video and audio using ffmpeg
    if video_temp_file and audio_temp_file:
        print("Combining video and audio using ffmpeg...")
        # Ensure output file doesn't exist or ffmpeg might hang
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)

        # Construct ffmpeg command
        # -y: overwrite output without asking
        # -i video: input video file
        # -i audio: input audio file
        # -c:v copy: copy video stream without re-encoding (fast)
        # -c:a aac: encode audio to AAC (common for mp4)
        # -b:a 192k: audio bitrate
        # -shortest: finish encoding when the shortest input stream ends (video or audio)
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', video_temp_file,
            '-i', audio_temp_file,
            '-c:v', 'copy',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest',
            OUTPUT_FILE
        ]

        try:
            # Run ffmpeg command
            result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            print("ffmpeg Output:", result.stdout)
            print("ffmpeg Error (if any):", result.stderr)
            print(f"Successfully combined streams into '{OUTPUT_FILE}'")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed with error code {e.returncode}")
            print("ffmpeg Output:", e.stdout)
            print("ffmpeg Error:", e.stderr)
            print("Check if ffmpeg is installed and in your system's PATH.")
        except FileNotFoundError:
             print("Error: ffmpeg command not found. Please install ffmpeg and ensure it's in your PATH.")
        except Exception as e:
            print(f"An unexpected error occurred during ffmpeg execution: {e}")

        # 6. Clean up temporary files
        print("Cleaning up temporary files...")
        try:
            if os.path.exists(video_temp_file): os.remove(video_temp_file)
            if os.path.exists(audio_temp_file): os.remove(audio_temp_file)
            print("Temporary files removed.")
        except Exception as e:
            print(f"Error removing temporary files: {e}")

    else:
        print("Skipping ffmpeg combination due to errors in video or audio generation.")

    print(f"--- Generation Process Finished ---")
