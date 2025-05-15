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
    zalgo_ b = [chr(i) for i in range(0x0300, 0x036F + 1)]
    zalgo_down = [chr(i) for i in range(0x0300, 0x036F + 1)] # Can reuse or use different ranges
    zalgo_mid = [chr(i) for i in range(0x0300, 0x036F + 1)]
    all_zalgo = zalgo_ ऊपर + zalgo_down + zalgo_mid
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
        "visual_effects": ["apply_perlin_noise", "apply_warp", "apply_feedback", "apply_data_bleed_sim", "apply_solarize", "apply_crt_ghost_sim", "apply_figurative_noise_sim"],
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
    # Add more themes like "Quantum Foam Simulation", "Urban Signal Ghosts"
}

# --- Scales for Buried Melody ---
SCALES = {
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": list(range(12)),
    "major": [0, 2, 4, 5, 7, 9, 11],
    "locrian": [0, 1, 3, 5, 6, 8, 10], # Very dissonant
    "augmented": [0, 4, 8], # Symmetrical, unsettling
    "whole_tone": [0, 2, 4, 6, 8, 10], # Dreamlike, ambiguous
}

# --- Font Configuration ---
# (FONT_PATHS, AVAILABLE_FONTS, DEFAULT_FONT, SELECTED_TTF_FONT, FONT_TO_USE, FONT_FACE_TO_USE remain mostly the same)
# ... (Assume font setup from previous version) ...
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/arialuni.ttf", "C:/Windows/Fonts/arial.ttf",
]
AVAILABLE_FONTS = [f for f in FONT_PATHS if os.path.exists(f)]
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
SELECTED_TTF_FONT = random.choice(AVAILABLE_FONTS) if AVAILABLE_FONTS else None
print(f"Available TTF fonts: {AVAILABLE_FONTS}")
FONT_FACE_TO_USE = DEFAULT_FONT # Keep it simple for now, OpenCV FreeType can be tricky
# For ARG text, we might use Pillow for better font control if needed.

# --- Pygame Mixer Init ---
try:
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=4096) # Larger buffer
except pygame.error as e:
    print(f"Pygame mixer init failed (non-critical for file output): {e}")

# --- Intensity Profile Generation ---
def generate_intensity_profile(duration_samples, sample_rate, scale=30.0, octaves=6, persistence=0.5, lacunarity=2.0):
    """Generates a highly dynamic Perlin noise-based intensity profile."""
    # ... (same as v2, but ensure parameters allow for sharp peaks and deep lulls) ...
    duration_seconds = duration_samples / sample_rate
    num_points = int(duration_seconds * 20) # Even more points for finer control
    if num_points < 2: num_points = 2
    profile = np.zeros(num_points)
    base = random.randint(0, 1000)

    for i in range(num_points):
        # Add another noise layer for more complexity
        n1 = noise.pnoise1(i / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base)
        n2 = noise.pnoise1(i / (scale*0.3) + 1000, octaves=3, persistence=0.7, lacunarity=1.8, base=base+1) # Faster variations
        profile[i] = (n1 * 0.7) + (n2 * 0.3)


    min_val, max_val = np.min(profile), np.max(profile)
    if max_val - min_val > 1e-6:
        normalized_profile = (profile - min_val) / (max_val - min_val)
        intensity_min = 0.05 # Allow very deep lulls
        intensity_max = 3.5  # Allow extremely high peaks for "Glitch Catalyst"
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
    """Generates a QR code image using the qrcode library."""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=max(1, size // 20), # Adjust box_size based on final image size
            border=1,
        )
        qr.add_data(data)
        qr.make(fit=True)
        img = qr.make_image(fill_color="white", back_color="black").convert('RGB')
        # Resize to target size using Pillow
        img = img.resize((size, size), Image.NEAREST)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # Convert PIL to OpenCV
    except Exception as e:
        print(f"Error generating QR code: {e}")
        return np.zeros((size, size, 3), dtype=np.uint8) # Return black square on error

def generate_arg_text_code(theme_data, length=16):
    """Generates a random text-based code (Base64, Hex, Binary)."""
    code_type = random.choice(theme_data.get("arg_elements", {}).get("text_code_types", ["hex"]))
    raw_data = os.urandom(length // 2 if code_type != "binary_short" else length // 8) # Adjust length for type

    if code_type == "base64":
        return base64.b64encode(raw_data).decode('utf-8')[:length]
    elif code_type == "hex":
        return binascii.hexlify(raw_data).decode('utf-8')[:length]
    elif code_type == "binary_short":
        return "".join(format(byte, '08b') for byte in raw_data)[:length]
    return "NULL_CODE"

# --- Visual Glitch Functions (Enhancements & New) ---
# ... (Keep apply_perlin_noise, apply_block_shift, etc. from v2, but adjust ranges for more abrasiveness)
# ... (Many existing visual effects from v2 are good foundations)

def apply_sensor_overload(frame, intensity):
    """Simulates sensor overload with extreme chromatic aberration and blown highlights."""
    output = frame.copy()
    # Extreme Chromatic Aberration
    max_shift = int(10 + 40 * intensity) # Scale shift with intensity
    temp_aberration = output.copy()
    for i in range(3):
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        channel = output[:,:,i]
        shifted = np.roll(np.roll(channel, shift_y, axis=0), shift_x, axis=1)
        # Blend shifted channel back with some transparency or make it dominant
        temp_aberration[:,:,i] = cv2.addWeighted(channel, 0.3, shifted, 0.7,0)
    output = temp_aberration

    # Blown Highlights
    if random.random() < 0.5 * intensity:
        alpha = 2.0 + 3.0 * intensity # Extreme contrast
        beta = random.randint(-50, 50) * intensity # Brightness boost
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)
        # Add bloom/persistence to highlights
        bright_mask = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(bright_mask, 230, 255, cv2.THRESH_BINARY)
        bloom = cv2.GaussianBlur(output, (0,0), sigmaX=10 + 20*intensity, sigmaY=10 + 20*intensity)
        output = cv2.addWeighted(output, 1.0, bloom, 0.3 * intensity, 0, dst=output, dtype=cv2.CV_8U)

    # Simulated Lens Dirt/Cracks (simple overlay)
    if random.random() < 0.2 * intensity:
        overlay = np.zeros_like(output, dtype=np.uint8)
        for _ in range(random.randint(1,5)): # Dirt specks
            cv2.circle(overlay, (random.randint(0,WIDTH), random.randint(0,HEIGHT)), random.randint(1,5), (20,20,20), -1)
        if random.random() < 0.1: # Lens crack
            cv2.line(overlay, (random.randint(0,WIDTH), random.randint(0,HEIGHT)), (random.randint(0,WIDTH), random.randint(0,HEIGHT)), (10,10,10), random.randint(1,2))
        output = cv2.addWeighted(output, 0.9, overlay, 0.3,0)
    return output

def apply_data_bleed_sim(frame, intensity):
    """Simulates colors bleeding from shifted blocks."""
    output = frame.copy()
    num_bleeds = random.randint(5, int(20 * intensity))
    for _ in range(num_bleeds):
        # Choose a source block
        bh = random.randint(int(HEIGHT*0.1), int(HEIGHT*0.4))
        bw = random.randint(int(WIDTH*0.1), int(WIDTH*0.4))
        y = random.randint(0, HEIGHT - bh -1)
        x = random.randint(0, WIDTH - bw -1)
        source_block = frame[y:y+bh, x:x+bw]
        avg_color = np.mean(source_block, axis=(0,1)).astype(np.uint8)

        # Choose a target area to bleed into
        bleed_strength = 0.2 + 0.6 * intensity * random.random()
        bleed_radius_x = random.randint(int(bw*0.5), int(bw*2))
        bleed_radius_y = random.randint(int(bh*0.5), int(bh*2))
        tx = np.clip(x + random.randint(-bw, bw), 0, WIDTH - bleed_radius_x)
        ty = np.clip(y + random.randint(-bh, bh), 0, HEIGHT - bleed_radius_y)

        target_area = output[ty:ty+bleed_radius_y, tx:tx+bleed_radius_x]
        # Create a color overlay and blend
        color_overlay = np.full_like(target_area, avg_color)
        output[ty:ty+bleed_radius_y, tx:tx+bleed_radius_x] = cv2.addWeighted(target_area, 1.0 - bleed_strength, color_overlay, bleed_strength, 0)
    return output

def apply_flickering_glitch_layer(frame, intensity, effect_func, prev_frame=None):
    """Applies an effect to a flickering transparent overlay."""
    if random.random() > 0.3 + 0.6 * intensity: # Control flicker rate
        return frame

    glitch_layer = frame.copy()
    # Apply the passed effect_func to the layer
    # Need to handle effects that require prev_frame
    sig = inspect.signature(effect_func)
    if 'prev_frame' in sig.parameters:
        glitch_layer = effect_func(glitch_layer, prev_frame if prev_frame is not None else frame)
    elif len(sig.parameters) == 1 or (len(sig.parameters) > 1 and all(p.default != inspect.Parameter.empty for p in list(sig.parameters.values())[1:])):
         # Assumes first param is 'frame', others have defaults or are ranges handled by wrapper
         try:
            glitch_layer = effect_func(glitch_layer) # Call with only frame if other params are optional
         except TypeError: # Fallback if it strictly needs ranges (this part is tricky)
            glitch_layer = effect_func(glitch_layer, alpha_range=(0.3,0.8), scale_range=(10,50)) # Example default ranges
    else: # If function has mandatory params beyond 'frame', this won't work well without specific handling
        return frame # Skip if effect signature is too complex for generic call


    # Make parts of the glitch_layer transparent (alpha mask)
    # For simplicity, create a random binary mask
    mask = np.random.randint(0, 2, size=(HEIGHT, WIDTH), dtype=np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask)

    # Apply mask: keep original frame where mask_inv is white, keep glitch_layer where mask is white
    fg = cv2.bitwise_and(glitch_layer, glitch_layer, mask=mask)
    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    return cv2.add(fg, bg)

def apply_figurative_noise_sim(frame, intensity):
    """Placeholder: Simulates noise forming fleeting figures (pareidolia)."""
    # This is very complex. A simple approach:
    # Generate Perlin noise, threshold it to get blobs, then slightly warp/distort these blobs.
    # For now, just a slightly more structured Perlin noise.
    return apply_perlin_noise(frame, alpha_range=(0.2, 0.5 * intensity), scale_range=(50, 150), oct_range=(4,6))


# --- Audio Effects (Enhancements & New) ---
# ... (Keep apply_distortion, apply_bitcrush, apply_spectral_glitch, apply_convolution_reverb from v2) ...

def apply_broken_speaker_sim(data, intensity):
    """Simulates a broken speaker with harsh filtering and distortion."""
    output = data.copy()
    # Harsh band-pass/stop filtering
    if random.random() < 0.7 * intensity:
        # Simple way: zero out some frequency bins (crude but fast)
        spectrum = fft(output)
        n = len(spectrum)
        if random.random() < 0.5: # Low-cut
            cut_freq_bin = int((random.uniform(500, 3000) / (SAMPLE_RATE / 2)) * (n / 2))
            spectrum[:cut_freq_bin] = 0
            spectrum[n-cut_freq_bin:] = 0
        else: # Mid-scoop or high-cut
            scoop_center_bin = int((random.uniform(1000, 6000) / (SAMPLE_RATE / 2)) * (n / 2))
            scoop_width_bin = int(scoop_center_bin * random.uniform(0.1, 0.5))
            spectrum[max(0,scoop_center_bin-scoop_width_bin):min(n//2, scoop_center_bin+scoop_width_bin)] = 0
            spectrum[n-min(n//2, scoop_center_bin+scoop_width_bin) : n-max(0,scoop_center_bin-scoop_width_bin)] = 0
        output = np.real(ifft(spectrum))

    # Heavy distortion
    output = apply_distortion(output, intensity_range=(5.0 + 10 * intensity, 15.0 + 20 * intensity))
    # Add crackling (short noise bursts)
    if random.random() < 0.5 * intensity:
        num_crackles = random.randint(10, 50)
        for _ in range(num_crackles):
            if len(output) == 0: break
            pos = random.randint(0, len(output)-1)
            crackle_len = random.randint(1, 50)
            end_pos = min(pos + crackle_len, len(output))
            output[pos:end_pos] += (np.random.rand(end_pos-pos) - 0.5) * 0.5 * intensity
    return np.clip(output, -1.0, 1.0) # Final clip

def generate_biometric_rhythm(duration_samples, vol, rhythm_type="heartbeat_distorted"):
    """Generates distorted biometric rhythms."""
    # ... (Similar to heartbeat in v2, but more options and distortion) ...
    noise_out = np.zeros(duration_samples, dtype=np.float32)
    if rhythm_type == "heartbeat_distorted":
        bpm = random.uniform(30, 150) * random.uniform(0.5, 1.5) # Wild BPM
        bps = bpm / 60.0
        if bps == 0: return noise_out # Avoid division by zero
        beat_interval_samples = int(SAMPLE_RATE / bps)
        if beat_interval_samples == 0: return noise_out

        thud_len = int(SAMPLE_RATE * random.uniform(0.05, 0.15))
        thud_freq = random.uniform(30, 100)
        if thud_len <=0 : return noise_out
        t_thud = np.linspace(0, thud_len * time_step, thud_len, endpoint=False)
        thud_env = np.exp(-np.linspace(0, random.uniform(5,15), thud_len))
        thud_sound = np.sin(2*np.pi*thud_freq*t_thud) * thud_env
        thud_sound = apply_distortion(thud_sound, (2.0, 5.0)) # Distort the thud

        current_sample = 0
        while current_sample < duration_samples:
            actual_thud_len = min(thud_len, duration_samples - current_sample)
            if actual_thud_len <=0 : break
            noise_out[current_sample : current_sample + actual_thud_len] += thud_sound[:actual_thud_len]
            current_sample += beat_interval_samples
    # Add "breathing_labored" etc.
    return apply_distortion(noise_out * vol, (1.5, 3.0)) # Distort final rhythm


# --- Melody Generation (More Abrasive & Extreme) ---
def generate_melody_fragment(duration_samples, vol, theme_data, intensity):
    """Generates a more abrasive melodic fragment with potential counter-melody and pitch bends."""
    # ... (from v2, but add counter-melody and pitch bends) ...
    if duration_samples <= 0: return np.array([], dtype=np.float32)

    scale_name = theme_data.get("melody_scale", "minor_pentatonic")
    scale_intervals = SCALES.get(scale_name, SCALES["minor_pentatonic"])
    base_freq = random.uniform(60, 660) # Wider base frequency
    notes_in_fragment = random.randint(2, int(6 + 10 * intensity)) # More notes when intense
    note_duration_samples = max(int(SAMPLE_RATE * 0.05), duration_samples // notes_in_fragment) # Min duration

    melody = np.zeros(duration_samples, dtype=np.float32)
    current_pos = 0

    # Main Melody
    for i in range(notes_in_fragment):
        if current_pos >= duration_samples: break
        scale_degree = random.choice(scale_intervals)
        octave_shift = random.choice([-1, 0, 0, 1, 1, 2]) # Wider octave jumps
        note_freq_start = base_freq * (2**((scale_degree + octave_shift * 12) / 12.0))

        # Extreme Pitch Bend
        pitch_bend_amount = random.uniform(-12, 12) * intensity # Semitones, scaled by intensity
        note_freq_end = note_freq_start * (2**(pitch_bend_amount / 12.0))
        note_freq_start = np.clip(note_freq_start, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)
        note_freq_end = np.clip(note_freq_end, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)

        actual_note_len = min(note_duration_samples, duration_samples - current_pos)
        if actual_note_len <= 0: break

        t_note = np.linspace(0, actual_note_len * time_step, actual_note_len, endpoint=False)
        # Create frequency sweep for pitch bend
        current_freqs = np.linspace(note_freq_start, note_freq_end, actual_note_len)
        # Phase accumulator for continuous frequency change
        phase = np.cumsum(2 * np.pi * current_freqs * time_step)
        note_tone = np.sin(phase)

        # Apply envelope to the pitch-bent tone
        attack_len = min(int(actual_note_len * 0.1), actual_note_len)
        decay_len = min(int(actual_note_len * 0.4), actual_note_len - attack_len)
        sustain_level = random.uniform(0.2, 0.8)
        envelope = np.ones(actual_note_len)
        if attack_len > 0: envelope[:attack_len] = np.linspace(0,1,attack_len)
        if decay_len > 0 and attack_len + decay_len <= actual_note_len :
            envelope[attack_len:attack_len+decay_len] = np.linspace(1,sustain_level,decay_len)
            envelope[attack_len+decay_len:] = sustain_level
        elif attack_len < actual_note_len:
            envelope[attack_len:] = sustain_level


        melody[current_pos : current_pos + actual_note_len] += note_tone * envelope

        current_pos += actual_note_len

    # Dissonant Counter-Melody (optional)
    if random.random() < theme_data.get("melody_counter_melody_chance", 0.3) * intensity:
        counter_melody = np.zeros(duration_samples, dtype=np.float32)
        current_pos_counter = 0
        # Use a highly dissonant interval from main melody notes or a different dissonant scale
        dissonant_intervals = [-1, 1, -6, 6, -11, 11] # m2, tritone, M7 etc.
        for i in range(notes_in_fragment // 2 + 1): # Fewer notes for counter
            if current_pos_counter >= duration_samples: break
            # Derive from main melody's scale or use a fixed dissonant one
            scale_degree_main = random.choice(scale_intervals)
            counter_degree = (scale_degree_main + random.choice(dissonant_intervals)) % 12
            octave_shift = random.choice([-1, 0, 1])
            note_freq = base_freq * 0.7 * (2**((counter_degree + octave_shift * 12) / 12.0)) # Slightly different base
            note_freq = np.clip(note_freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF)
            actual_note_len = min(note_duration_samples * random.randint(1,2), duration_samples - current_pos_counter)
            if actual_note_len <=0: break
            note_tone = generate_tone(note_freq, actual_note_len, 0.7, fm_chance=0.6, harmonic_chance=0.1) # More FM
            counter_melody[current_pos_counter : current_pos_counter + actual_note_len] += note_tone
            current_pos_counter += actual_note_len
        melody += counter_melody * 0.6 # Mix counter-melody quieter

    # Heavy Distortion & Abrasive Effects
    distortion_amount = theme_data.get("melody_distortion", 10.0) * (1.0 + intensity) # Scale distortion
    melody = apply_distortion(melody, intensity_range=(distortion_amount * 0.7, distortion_amount * 1.5))
    if random.random() < 0.6 * intensity:
         melody = apply_bitcrush(melody, bit_depth_range=(2, 6)) # Harsher bitcrush
    if random.random() < 0.3 * intensity:
         melody = apply_spectral_glitch(melody, intensity=intensity*0.8)
    if random.random() < 0.2 * intensity: # Granular shredding simulation (crude)
        grain_len = int(SAMPLE_RATE * random.uniform(0.005, 0.02))
        num_grains = int(len(melody) / grain_len) // 2
        shredded_melody = np.zeros_like(melody)
        for _ in range(num_grains):
            if len(melody) < grain_len : break
            start_idx = random.randint(0, len(melody) - grain_len)
            grain = melody[start_idx : start_idx + grain_len]
            place_idx = random.randint(0, len(shredded_melody) - grain_len)
            shredded_melody[place_idx : place_idx + grain_len] += grain * random.uniform(0.5,1.2)
        melody = shredded_melody

    return melody * vol


# --- Frame Generation (incorporating ARG, new effects) ---
def generate_frames_enhanced(theme_data, intensity_func, global_params):
    # ... (Setup from v2: DURATION, FPS, FRAME_COUNT, current_params, instability_chance, video writer etc.)
    # ... (Persistent text setup from v2, but use zalgo_text more often)
    DURATION = random.randint(MIN_DURATION, MAX_DURATION)
    FPS = random.randint(MIN_FPS, MAX_FPS)
    FRAME_COUNT = DURATION * FPS

    current_visual_params = global_params["visual"] # Use the visual sub-dictionary
    instability_chance = global_params["instability_chance"] * 1.5

    print(f"Generating {DURATION}s video ({FPS} FPS), Theme: {global_params['theme_name']}...")
    print(f"Visual Instability Chance: {instability_chance:.4f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_temp_path = f"video_temp_{int(time.time())}_{random.randint(100,999)}.mp4"
    video = cv2.VideoWriter(video_temp_path, fourcc, FPS, (WIDTH, HEIGHT))

    prev_frame = None
    frame_hold_counter = 0
    held_frame = None
    effect_history = []
    # For dynamic theme mutation
    current_theme_mutated = copy.deepcopy(theme_data)
    mutation_timer = 0
    mutation_interval = FPS * random.randint(5,15) # Mutate theme params every 5-15s

    # ARG element persistence/timing
    arg_qr_img = None
    arg_qr_display_frames = 0
    arg_text_code = None
    arg_text_display_frames = 0

    persistent_texts = [] # ... (Setup as in v2, ensure zalgo_text is used based on theme) ...
    num_persistent = random.randint(8, 20) # Even more text elements
    for _ in range(num_persistent):
        word = str(random.choice(current_theme_mutated["words"])) # Ensure string
        symbol = str(random.choice(current_theme_mutated["symbols"]))
        if random.random() < current_theme_mutated.get("zalgo_words_chance", 0.3):
             word = zalgo_text(word)
        # ... (encoding as in v2) ...
        persistent_texts.append({
            "text": word + symbol,
            "pos": (random.randint(-100, WIDTH), random.randint(-100, HEIGHT)), # Allow more off-screen
            "font_size": random.uniform(0.4, 5.0), # Wider
            "color": random.choice(random.choice(current_theme_mutated["colors"])),
            "lifetime": random.randint(int(FPS * 0.05), int(FPS * 10)), # Shorter min, longer max
            "frame_count": 0,
            "move_speed": (random.uniform(-12, 12), random.uniform(-10, 10)), # Faster
            "alpha": random.uniform(0.3, 1.0) # Add alpha for text
        })


    for i in range(FRAME_COUNT):
        current_time_sec = i / FPS
        current_intensity = intensity_func(current_time_sec)
        current_intensity = np.clip(current_intensity, 0.01, 4.0) # Even wider clamp

        # --- Dynamic Theme Mutation ---
        mutation_timer += 1
        if mutation_timer >= mutation_interval:
            mutation_timer = 0
            if random.random() < 0.3 * current_intensity: # Higher chance when intense
                # Mutate a color palette
                palette_idx = random.randrange(len(current_theme_mutated["colors"]))
                color_idx = random.randrange(len(current_theme_mutated["colors"][palette_idx]))
                current_theme_mutated["colors"][palette_idx][color_idx] = (
                    np.clip(current_theme_mutated["colors"][palette_idx][color_idx][0] + random.randint(-30,30), 0, 255),
                    np.clip(current_theme_mutated["colors"][palette_idx][color_idx][1] + random.randint(-30,30), 0, 255),
                    np.clip(current_theme_mutated["colors"][palette_idx][color_idx][2] + random.randint(-30,30), 0, 255),
                )
                print(f"Theme Mutation: Color palette {palette_idx} changed at {current_time_sec:.1f}s")
            # Could mutate other things: dominant noise type, ARG code probability etc.

        # --- Parameter Instability (Visual) ---
        # ... (Same logic as v2, but use current_visual_params) ...

        # --- Frame Stutter/Hold ---
        # ... (Same logic, maybe increase hold duration with intensity) ...
        if frame_hold_counter > 0:
            if held_frame is not None: video.write(held_frame); frame_hold_counter -=1; continue
            else: frame_hold_counter = 0
        if random.random() < (0.02 + 0.03 * current_intensity) and frame_hold_counter == 0:
            frame_hold_counter = random.randint(1, int(FPS * (0.5 + current_intensity * 0.5)))
            held_frame = None


        # --- Base Frame ---
        # ... (Same as v2: solid, gradient, noise base) ...
        base_roll = random.random()
        if base_roll < 0.2: # Solid Color - less frequent
             bg_color = random.choice(random.choice(current_theme_mutated["colors"]))
             frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
        elif base_roll < 0.4: # Gradient
             c1 = random.choice(random.choice(current_theme_mutated["colors"]))
             c2 = random.choice(random.choice(current_theme_mutated["colors"]))
             frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
             for k_row in range(HEIGHT): frame[k_row, :] = [int(c1[ch] * (1 - k_row/HEIGHT) + c2[ch] * (k_row/HEIGHT)) for ch in range(3)]
             if random.random() < 0.5: frame = cv2.rotate(frame, random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]))
        else: # Noise Base (more common)
             frame = np.random.randint(0, int(30 + 40 * current_intensity), (HEIGHT, WIDTH, 3), dtype=np.uint8) # Intensity affects noise brightness


        # --- Glitch Catalyst Event ---
        is_catalyst_event = False
        if current_intensity > (global_params.get("visual",{}).get("catalyst_threshold", 2.8)) and random.random() < 0.3: # High intensity peak
            is_catalyst_event = True
            print(f"GLITCH CATALYST at {current_time_sec:.1f}s! Intensity: {current_intensity:.2f}")
            num_effects_to_apply_catalyst = random.randint(5, 8) # Apply many effects
            catalyst_effect_duration = int(FPS * random.uniform(0.2, 0.7)) # Short burst duration
            # This needs to integrate into the main effect loop or have its own short loop

        # --- Apply Visual Effects ---
        # ... (Effect selection from v2, but use current_theme_mutated and current_visual_params) ...
        # ... (Ensure new effects like apply_sensor_overload are in theme's visual_effects list and global_params probabilities)
        available_effect_names = current_theme_mutated.get("visual_effects", [])
        effect_candidates = [globals()[name] for name in available_effect_names if name in globals() and callable(globals()[name])]
        num_effects_base = random.randint(1,3) if not is_catalyst_event else num_effects_to_apply_catalyst
        num_effects_intensity = int(current_intensity * 2.0) if not is_catalyst_event else 0 # Catalyst overrides
        num_effects_to_apply = min(len(effect_candidates), num_effects_base + num_effects_intensity)

        # ... (Effect application loop from v2, pass scaled ranges, handle prev_frame) ...
        # ... (Ensure new effects like apply_flickering_glitch_layer get called correctly)
        # For flickering layer, choose another random effect to apply within it:
        # if effect_func == apply_flickering_glitch_layer:
        #    inner_effect_func = random.choice([f for f in effect_candidates if f != apply_flickering_glitch_layer])
        #    frame = apply_flickering_glitch_layer(frame, current_intensity, inner_effect_func, prev_frame)


        # --- ARG Element Integration (Visual) ---
        frame_overlay_arg = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8) # RGBA for transparency

        # QR Code (Tier 1)
        if arg_qr_display_frames > 0:
            if arg_qr_img is not None:
                # Random position, slightly distorted/flickering
                qr_x = random.randint(0, WIDTH - ARG_QR_SIZE)
                qr_y = random.randint(0, HEIGHT - ARG_QR_SIZE)
                temp_qr = arg_qr_img.copy()
                if random.random() < 0.3 * current_intensity: # Flicker/Distort QR
                    temp_qr = apply_pixelation(temp_qr, (5,15))
                    temp_qr = cv2.addWeighted(temp_qr, random.uniform(0.5,1.0), np.zeros_like(temp_qr),0.5,0)

                # Simple overlay (no alpha blending here for speed, could improve with PIL draw)
                # This needs alpha blending to be effective with frame_overlay_arg
                # For now, direct overlay on frame before other text
                if qr_y + ARG_QR_SIZE <= HEIGHT and qr_x + ARG_QR_SIZE <= WIDTH:
                     frame[qr_y:qr_y+ARG_QR_SIZE, qr_x:qr_x+ARG_QR_SIZE] = cv2.addWeighted(
                         frame[qr_y:qr_y+ARG_QR_SIZE, qr_x:qr_x+ARG_QR_SIZE], 0.5,
                         temp_qr, 0.9 * (arg_qr_display_frames / (FPS*2.0) ), # Fade out
                         0)
            arg_qr_display_frames -=1
        elif random.random() < (0.005 + 0.01 * current_intensity) * global_params.get("arg_qr_prob_modifier", 1.0): # Chance to show QR
            qr_data = current_theme_mutated.get("arg_elements",{}).get("qr_data_prefix","QR_") + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            arg_qr_img = generate_qr_code_image(qr_data, ARG_QR_SIZE)
            arg_qr_display_frames = int(FPS * random.uniform(0.5, 2.0)) # Display for 0.5-2s
            print(f"ARG: Displaying QR Code: {qr_data} for {arg_qr_display_frames} frames")

        # Text Codes (Base64, Hex, Binary - Tier 1)
        if arg_text_display_frames > 0:
            if arg_text_code:
                # Display somewhere prominent, maybe with slight distortion
                text_size, _ = cv2.getTextSize(arg_text_code, FONT_FACE_TO_USE, ARG_TEXT_FONT_SCALE_BASE * (1.0 + current_intensity*0.2), 2)
                text_x = random.randint(10, WIDTH - text_size[0] - 10)
                text_y = random.randint(text_size[1] + 10, HEIGHT - 10)
                alpha_val = int(255 * (arg_text_display_frames / (FPS*1.5))) # Fade out
                # Use Pillow to draw on RGBA overlay for better text rendering & alpha
                # This is more complex, for now, simple OpenCV putText on main frame:
                cv2.putText(frame, arg_text_code, (text_x, text_y), FONT_FACE_TO_USE,
                            ARG_TEXT_FONT_SCALE_BASE * (1.0 + current_intensity*0.1),
                            (200,220,255), # Light color
                            random.randint(1,2), cv2.LINE_AA)
            arg_text_display_frames -=1
        elif random.random() < (0.01 + 0.02 * current_intensity) * global_params.get("arg_text_prob_modifier", 1.0):
            arg_text_code = generate_arg_text_code(current_theme_mutated, length=random.randint(16,48))
            arg_text_display_frames = int(FPS * random.uniform(0.3, 1.5))
            print(f"ARG: Displaying Text Code: {arg_text_code} for {arg_text_display_frames} frames")

        # Fake Error Codes / Cipher Keys (Tier 1 / Shard Protocol)
        if random.random() < (0.02 + 0.05 * current_intensity) * global_params.get("arg_protocol_prob_modifier", 1.0): # More frequent
            arg_elements = current_theme_mutated.get("arg_elements", {})
            error_list = arg_elements.get("fake_errors", [])
            key_list = arg_elements.get("cipher_keys", [])
            protocol_text = ""
            if error_list and random.random() < 0.6:
                protocol_text = random.choice(error_list)
            elif key_list:
                protocol_text = "KEY_FRAGMENT: " + random.choice(key_list)

            if protocol_text:
                # Display as a rapid, large flash
                font_scale = ARG_ERROR_CODE_FONT_SCALE * (1.0 + current_intensity * 0.5)
                text_size, _ = cv2.getTextSize(protocol_text, FONT_FACE_TO_USE, font_scale, 3)
                text_x = max(10, (WIDTH - text_size[0]) // 2)
                text_y = max(10, (HEIGHT + text_size[1]) // 2 + random.randint(-HEIGHT//4, HEIGHT//4)) # Random Y
                # Flash with high contrast color
                flash_bg_color = (0,0,0) if np.mean(frame) > 100 else (200,200,200)
                flash_text_color = (255,0,0) if np.mean(frame) > 100 else (255,50,50)
                # Create a temporary overlay for the flash
                flash_overlay = frame.copy()
                cv2.putText(flash_overlay, protocol_text, (text_x, text_y), FONT_FACE_TO_USE, font_scale, flash_text_color, random.randint(2,4), cv2.LINE_AA)
                # Blend this flash strongly for 1-2 frames (this is a single frame flash here)
                frame = cv2.addWeighted(frame, 0.3, flash_overlay, 0.7,0)
                print(f"ARG: Flashing Protocol Text: {protocol_text}")


        # --- Subliminal Flash (More Abrasive) ---
        # ... (v2 logic, but increase probability/intensity of effect) ...
        flash_probability = 0.20 * current_intensity # Higher base chance
        if random.random() < flash_probability:
            # ... (Flash logic as in v2, ensuring it's brief and impactful) ...
            pass


        # --- Persistent Text Rendering (More Chaotic) ---
        # ... (v2 logic, but use current_theme_mutated, add alpha, faster movement, more off-screen) ...
        # ... (Ensure text color contrasts with dynamic background) ...
        # This part needs careful alpha blending if using frame_overlay_arg


        # --- Store frame & Write ---
        prev_frame = frame.copy()
        if frame_hold_counter > 0 and held_frame is None: held_frame = frame.copy()
        video.write(frame)

        if (i + 1) % (FPS * 1) == 0 and frame_hold_counter == 0: # More frequent updates
            print(f"Video Frame {i+1}/{FRAME_COUNT} (Int: {current_intensity:.2f})...")

    video.release()
    return DURATION, FPS, video_temp_path


# --- Audio Generation (incorporating ARG, new effects) ---
def generate_audio_enhanced(video_duration, video_fps, theme_data, intensity_func, global_params):
    # ... (Setup from v2: DURATION, NUM_SAMPLES, etc.) ...
    # ... (Use current_theme_mutated and current_audio_params) ...
    # ... (Melody track generation using enhanced generate_melody_fragment) ...
    # ... (Incorporate biometric rhythms, broken speaker sim) ...

    # ARG Audio Elements (Placeholders)
    # if random.random() < 0.01 * current_intensity: # DTMF tones
    #     # Generate DTMF sequence for a short code/URL fragment
    #     pass
    # if random.random() < 0.01 * current_intensity: # Morse code
    #     # Generate Morse code clicks/beeps
    #     pass
    # if random.random() < 0.005 * current_intensity: # Reversed Speech
    #     # (Requires TTS and reversing - complex)
    #     pass
    # if random.random() < 0.005 * current_intensity: # Numbers Station
    #     # (Requires TTS - complex)
    #     pass
    # ... (Final mixing and WAV writing from v2) ...
    DURATION = video_duration
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    if NUM_SAMPLES <= 0: return None # Safety
    NUM_CHANNELS = 2
    samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32)

    current_audio_params = global_params["audio"]
    audio_instability_chance = global_params["instability_chance"] * 1.2

    print(f"Generating {DURATION:.1f}s audio, Theme: {global_params['theme_name']}...")

    min_freq, max_freq = theme_data["audio_freq_range"]
    noise_types = theme_data["audio_noise_types"]
    features = theme_data["audio_features"]
    current_theme_mutated = copy.deepcopy(theme_data) # Use a mutable copy for audio gen too

    last_event_end_sample = 0
    melody_track = np.zeros(NUM_SAMPLES, dtype=np.float32)
    biometric_track = np.zeros(NUM_SAMPLES, dtype=np.float32)

    # Generate base biometric track if theme supports it
    if "biometric_rhythm" in features: # Check generic feature
        biometric_type = "heartbeat_distorted" # Default
        # Could refine this by checking specific noise types in theme_data
        if "heartbeat" in noise_types or "breathing" in noise_types:
            biometric_type = random.choice([t for t in noise_types if "heart" in t or "breath" in t] or ["heartbeat_distorted"])
        biometric_track = generate_biometric_rhythm(NUM_SAMPLES, 0.3, rhythm_type=biometric_type) # Lower vol for base

    while last_event_end_sample < NUM_SAMPLES:
        current_sample_time = last_event_end_sample / SAMPLE_RATE
        current_intensity = intensity_func(current_sample_time)
        current_intensity = np.clip(current_intensity, 0.01, 4.0)

        # ... (Audio Parameter Instability logic from v2) ...

        event_max_duration = max(0.01, 1.5 / (current_intensity + 1.0)) # Even shorter events possible
        event_min_duration = 0.002 # Very short grains
        event_duration_sec = random.uniform(event_min_duration, event_max_duration)
        event_duration_samples = int(event_duration_sec * SAMPLE_RATE)
        event_duration_samples = min(max(1, event_duration_samples), NUM_SAMPLES - last_event_end_sample)
        if event_duration_samples <= 0: break

        event_start_sample = last_event_end_sample
        event_end_sample = event_start_sample + event_duration_samples
        event_vol = np.clip(abs(random.uniform(0.01,0.5) * (0.2 + current_intensity * 1.5)), 0.001, 1.0) # More dynamic range

        # Event Type Choice
        type_roll = random.random()
        silence_thresh = max(0.005, 0.1 * (2.0 - current_intensity)) # Less silence
        melody_thresh = silence_thresh + (0.1 + 0.1 * current_intensity) # Melody chance increases with intensity
        noise_thresh = melody_thresh + (0.6 + 0.25 * current_intensity) # Noise very likely

        segment_mono = np.zeros(event_duration_samples, dtype=np.float32)
        is_melody = False

        if type_roll < silence_thresh: event_type = "silence"
        elif type_roll < melody_thresh:
            event_type = "melody"; is_melody = True
            melody_vol_mod = 0.3 + 0.3 * current_intensity # Melody louder when intense
            segment_mono = generate_melody_fragment(event_duration_samples, event_vol * melody_vol_mod, current_theme_mutated, current_intensity)
        elif type_roll < noise_thresh:
            event_type = "noise"
            noise_choice = random.choice(noise_types)
            if current_intensity > 2.0 and random.random() < 0.8: # Prioritize extreme noises
                 harsh_noises = [n for n in noise_types if any(k in n for k in ["screech", "clip", "feedback", "artifact", "data_stream"])]
                 if harsh_noises: noise_choice = random.choice(harsh_noises)
            segment_mono = generate_noise(noise_choice, event_duration_samples, event_vol, features)
        else:
            event_type = "tone"
            # ... (Tone generation as in v2, but with more aggressive FM/Harmonic chances based on intensity) ...
            fm_chance = np.clip(0.3 + current_intensity * 0.4, 0.1, 0.95)
            harmonic_chance = np.clip(0.2 + current_intensity * 0.3, 0.1, 0.9)
            # ... (freq calculation) ...
            freq = random.uniform(min_freq, max_freq * (0.5 + current_intensity))
            segment_mono = generate_tone(np.clip(freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF), event_duration_samples, event_vol, fm_chance, harmonic_chance)


        if len(segment_mono) != event_duration_samples and event_duration_samples > 0: # Safety pad/truncate
            padded_segment = np.zeros(event_duration_samples, dtype=np.float32)
            len_to_copy = min(len(segment_mono), event_duration_samples)
            if len_to_copy > 0: padded_segment[:len_to_copy] = segment_mono[:len_to_copy]
            segment_mono = padded_segment

        if event_type != "silence":
            # Apply Audio Effects (More Abrasive)
            effect_chance_mod = 0.6 + current_intensity # Very high chance for effects

            if "distortion" in features and random.random() < 0.6 * effect_chance_mod:
                d_min, d_max = current_audio_params["distortion_intensity"]
                dist_range = (d_min * (0.7 + current_intensity*1.2), d_max * (1.0 + current_intensity*1.5))
                segment_mono = apply_distortion(segment_mono, intensity_range=np.clip(dist_range, 1.0, 40.0)) # Max distortion up

            if "bitcrush" in features and random.random() < 0.5 * effect_chance_mod:
                 b_min, b_max = current_audio_params["bitcrush_depth"]
                 target_bits = int(max(2, b_min + (b_max - b_min) * (2.0 - current_intensity)))
                 segment_mono = apply_bitcrush(segment_mono, bit_depth_range=(2, max(2, target_bits)))

            if "spectral_glitch" in features and random.random() < 0.4 * effect_chance_mod:
                 segment_mono = apply_spectral_glitch(segment_mono, intensity=current_intensity * 1.2)

            if "broken_speaker_sim" in features and random.random() < 0.2 * effect_chance_mod:
                 segment_mono = apply_broken_speaker_sim(segment_mono, current_intensity)

            # ... (Convolution Reverb from v2) ...

            # Panning
            pan_extremity = np.clip(0.5 + current_intensity * 0.5, 0.2, 1.0)
            pan = random.uniform(-pan_extremity, pan_extremity)
            gain_l, gain_r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))

            end_idx = min(event_end_sample, NUM_SAMPLES)
            length = end_idx - event_start_sample
            if length <=0: continue

            if is_melody:
                 melody_track[event_start_sample:end_idx] += segment_mono[:length]
            else:
                 samples[event_start_sample:end_idx, 0] += segment_mono[:length] * gain_l
                 samples[event_start_sample:end_idx, 1] += segment_mono[:length] * gain_r

        # Advance time, allowing extreme overlap
        overlap_factor = np.clip(0.3 + current_intensity * 0.5, 0.1, 0.95) # Up to 95% overlap
        advance_samples = int(event_duration_samples * (1.0 - overlap_factor))
        last_event_end_sample += max(1, advance_samples)


    # Mix tracks: Main samples + Melody + Biometric
    samples[:,0] += biometric_track * (0.2 + 0.3 * global_params.get("intensity_at_end",1.0)) # Biometric vol can scale with overall intensity
    samples[:,1] += biometric_track * (0.2 + 0.3 * global_params.get("intensity_at_end",1.0))

    max_melody_abs = np.max(np.abs(melody_track));
    if max_melody_abs > 1e-6: melody_track /= max_melody_abs
    melody_mix_level = 0.25 + 0.2 * global_params.get("intensity_at_end",1.0) # Melody louder if end is intense
    samples[:, 0] += melody_track * melody_mix_level
    samples[:, 1] += melody_track * melody_mix_level

    # Final Normalization
    max_abs_amplitude = np.max(np.abs(samples))
    if max_abs_amplitude > 1e-6: samples /= max_abs_amplitude
    else: print("Warning: Final audio mix is silent or near silent.")

    samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    audio_temp_path = f"audio_temp_{int(time.time())}_{random.randint(100,999)}.wav"
    # ... (WAV writing as in v2) ...
    try:
        with wave.open(audio_temp_path, 'wb') as wf:
            wf.setnchannels(NUM_CHANNELS); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples_int16.tobytes())
        return audio_temp_path
    except Exception as e: print(f"Error writing WAV: {e}"); return None


# --- Main Execution ---
if __name__ == "__main__":
    # ... (Setup from v2: initial_theme, global_params) ...
    # ... (global_params["visual"]["effect_probabilities"] should map function names (strings) to probs)
    # ... (Ensure new visual/audio effects are in global_params default probabilities)
    # ... (Call generate_intensity_profile, generate_frames_enhanced, generate_audio_enhanced)
    # ... (ffmpeg combination and cleanup) ...
    start_time_main = time.time()
    initial_theme_name = random.choice(list(THEMES.keys()))
    current_theme_data = copy.deepcopy(THEMES[initial_theme_name])
    print(f"--- Starting Shardmind Protocol Generation v3 ---")
    print(f"Initial Theme: {initial_theme_name}")

    global_params = {
        "theme_name": initial_theme_name,
        "visual": {
             "catalyst_threshold": 2.8, # Intensity level to trigger catalyst
             # ... (other visual params from v2) ...
             "effect_probabilities": {name: prob for name, prob in [
                ("apply_perlin_noise",0.7), ("apply_block_shift",0.6), ("apply_color_channel_shift",0.5),
                ("apply_warp",0.5), ("apply_pixelation",0.4), ("apply_scanlines",0.3),
                ("apply_solarize",0.2), ("apply_extreme_contrast",0.5), ("apply_feedback",0.6),
                ("apply_datamosh_sim",0.3), ("apply_ascii_sim",0.2), ("apply_sensor_overload",0.25),
                ("apply_data_bleed_sim",0.2), ("apply_flickering_glitch_layer",0.25),
                ("apply_crt_ghost_sim",0.15), ("apply_slit_scan_sim",0.1), ("apply_vector_field_sim",0.1),
                ("apply_figurative_noise_sim",0.1)
             ]},
             # Add default ranges for any new effect parameters here if they are not hardcoded
        },
        "audio": {
            "distortion_intensity": (1.0, 25.0), # Wider
            "bitcrush_depth": (2, 8), # Harsher
        },
        "instability_chance": 0.012, # Higher instability
        "style_break_chance": 0.0005, # Lowered for less frequent full breaks
        "arg_qr_prob_modifier": 1.0, # Modifiers for ARG element frequencies
        "arg_text_prob_modifier": 1.0,
        "arg_protocol_prob_modifier": 1.0,
        "intensity_at_end": 1.0 # Placeholder, will be updated
    }

    estimated_max_duration_sec = MAX_DURATION + 15
    estimated_max_samples = int(estimated_max_duration_sec * SAMPLE_RATE)
    intensity_function = generate_intensity_profile(estimated_max_samples, SAMPLE_RATE)
    # Get intensity at the typical end for audio mixing scaling
    global_params["intensity_at_end"] = np.clip(intensity_function(MAX_DURATION - 5), 0.1, 3.0)


    # --- Style Break / Dynamic Theme Evolution (Conceptual) ---
    # This would ideally be integrated deeper. For a simpler version:
    # The `current_theme_data` can be mutated slowly within `generate_frames_enhanced`
    # and `generate_audio_enhanced` or a wrapper function.

    actual_duration, actual_fps, video_temp_file = generate_frames_enhanced(current_theme_data, intensity_function, global_params)
    if not video_temp_file: print("Video gen failed!"); exit()

    audio_temp_file = generate_audio_enhanced(actual_duration, actual_fps, current_theme_data, intensity_function, global_params)
    if not audio_temp_file: print("Audio gen failed!"); exit()

    # --- FFMPEG Combination ---
    if video_temp_file and audio_temp_file:
        print("Combining with ffmpeg...")
        if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
        ffmpeg_command = [
            'ffmpeg', '-y', '-i', video_temp_file, '-i', audio_temp_file,
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '24', # Slower preset for better compression/quality
            '-pix_fmt', 'yuv420p', # For wider compatibility
            '-c:a', 'aac', '-b:a', '320k', # Higher audio bitrate
            '-shortest', OUTPUT_FILE
        ]
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            print(f"Output: '{OUTPUT_FILE}'")
        except Exception as e:
            print(f"ffmpeg error: {e}")
            if hasattr(e, 'stderr'): print(e.stderr)

    # --- Cleanup ---
    for f in [video_temp_file, audio_temp_file]:
        if f and os.path.exists(f): os.remove(f)

    print(f"--- Total Time: {time.time() - start_time_main:.2f}s ---")




