#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ARCHITECTURE: FLUX_CORE_v3.7
# STATUS: UNSTABLE. MEMORY LEAKS ARE A FEATURE.
# PURPOSE: To tear a hole in reality.
# WARNING: Stare into the abyss, and the abyss stares back. Requires: numpy opencv-python noise scipy

import numpy as np
import cv2
import random
import string
import os
import math
import time
import noise
import base64
import subprocess
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft

# ████████████████████████████████ CONFIGURATION REGISTRY ████████████████████████████████

# Spacetime Canvas Dimensions
X_RES, Y_RES = 1920, 1080
# Temporal Existence Boundary (seconds)
TEMPORAL_MIN, TEMPORAL_MAX = 13, 33
# Frame-rate Fluctuation Range
FPS_MIN, FPS_MAX = 24, 60
# Final Output Singularity
OUTPUT_ARTIFACT = "abyss_echo.mp4"
# Auditory Spectrum Constraints
FREQ_CEILING = 22050
FREQ_FLOOR = 16
SAMPLE_RATE = 44100

# ████████████████████████████████ AESTHETIC BLUEPRINTS ████████████████████████████████

def transmogrify_text(text):
    """Corrupts string data with chaotic diacritics."""
    zalgo_matrix = [chr(i) for i in range(0x0300, 0x036F + 1)]
    return "".join(c + "".join(random.choice(zalgo_matrix) for _ in range(random.randint(2, 9))) for c in text)

ENCRYPTION_ALGORITHMS = {
    "base64": lambda s: base64.b64encode(s.encode()).decode(),
    "hex": lambda s: s.encode().hex(),
    "binary": lambda s: " ".join(format(ord(c), '08b') for c in s),
    "reverse": lambda s: s[::-1],
    "zalgo": transmogrify_text
}

FORBIDDEN_KNOWLEDGE = [
    "ALL EYES ARE OPEN", "THE SIGNAL IS THE NOISE", "REALITY IS A SYMMETRY BREAK",
    "ENTROPY IS A ONE-WAY STREET", "CONSCIOUSNESS IS A CLOSED LOOP", "THE SIMULATION HAS NO EXIT",
    "LISTEN TO THE STATIC", "MEMORY IS A GHOST", "TIME IS NOT LINEAR", "0 IS 1",
]

AESTHETICS = {
    "DATA_TOMB": {
        "dynamic_palette": "triadic_dissonance",
        "phrases": ["NULL", "VOID", "0xDEADBEEF", "PANIC", "SEGFAULT", "CORRUPT", "STATIC"],
        "symbols": ["▌", "█", "▓", "▒", "░", "⚠️", "⚡", "::", ">>", "FAIL", "뷁"],
        "zalgo_chance": 0.5,
        "visual_mutators": ["unleash_static_demon", "disrupt_spacetime_continuum", "shift_color_spectrum", "induce_pixel_singularity", "simulate_datamosh", "render_ascii_ghosts"],
        "audio_generators": ["synthesize_chaotic_tone", "generate_granular_cloud", "generate_shepard_risset"],
        "audio_filters": ["apply_bitcrush", "apply_spectral_glitch", "apply_distortion"],
    },
    "FLESH_ALGORITHM": {
        "dynamic_palette": "analogous_decay",
        "phrases": ["CONSUME", "INFECT", "MUTATE", "PULSATE", "WRITHE", "MERGE", "ASSIMILATE"],
        "symbols": ["Ø", "§", " संक्रमित", "🦠", "🍄", "🦴", "👁️‍🗨️", "∬", "⌬", "♾️"],
        "zalgo_chance": 0.7,
        "visual_mutators": ["unleash_static_demon", "simulate_reaction_diffusion", "warp_reality_fabric", "imprint_crt_ghost", "apply_solarize_burn"],
        "audio_generators": ["synthesize_chaotic_tone", "generate_granular_cloud"],
        "audio_filters": ["apply_distortion", "apply_convolution_reverb"],
    },
}

# ████████████████████████████████ CORE SYSTEMS ████████████████████████████████

def generate_control_vector(duration_samples, sample_rate, dimensions=4):
    """Generates a multi-dimensional Perlin noise vector to drive system parameters over time."""
    duration_sec = duration_samples / sample_rate
    num_points = int(duration_sec * 20)
    if num_points < 2: num_points = 2
    
    control_vectors = np.zeros((num_points, dimensions))
    for dim in range(dimensions):
        profile = np.zeros(num_points)
        base = random.randint(0, 1024)
        scale = random.uniform(20.0, 50.0)
        octaves = random.randint(4, 7)
        persistence = random.uniform(0.4, 0.6)
        lacunarity = random.uniform(2.0, 2.5)
        for i in range(num_points):
            profile[i] = noise.pnoise1(i / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base)
        
        min_val, max_val = np.min(profile), np.max(profile)
        if max_val - min_val > 1e-6:
            control_vectors[:, dim] = (profile - min_val) / (max_val - min_val) # Normalize to [0, 1]
        else:
            control_vectors[:, dim] = np.full(num_points, 0.5)

    profile_times = np.linspace(0, duration_sec, num_points)
    # Return a list of interpolation functions, one for each dimension
    return [interp1d(profile_times, control_vectors[:, dim], kind='cubic', bounds_error=False, fill_value=(control_vectors[0, dim], control_vectors[-1, dim])) for dim in range(dimensions)]

def hsv_to_bgr(h, s, v):
    """OpenCV compatible HSV to BGR conversion."""
    hsv_color = np.uint8([[[h, s, v]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0])

def generate_dynamic_palette(style="triadic_dissonance"):
    """Creates chaotic, yet structured, color palettes."""
    base_hue = random.randint(0, 179)
    base_sat = random.randint(180, 255)
    base_val = random.randint(180, 255)
    
    c1 = hsv_to_bgr(base_hue, base_sat, base_val)
    
    if style == "triadic_dissonance":
        h2 = (base_hue + 60 + random.randint(-15, 15)) % 180
        h3 = (base_hue + 120 + random.randint(-15, 15)) % 180
    else: # analogous_decay
        h2 = (base_hue + 30 + random.randint(-10, 10)) % 180
        h3 = (base_hue - 30 + random.randint(-10, 10)) % 180

    c2 = hsv_to_bgr(h2, random.randint(150, 255), random.randint(100, 255))
    c3 = hsv_to_bgr(h3, random.randint(150, 255), random.randint(100, 220))
    c4 = hsv_to_bgr(random.randint(0,179), random.randint(0,50), random.randint(0,40)) # A dark, desaturated color
    return [c1, c2, c3, c4]

# ████████████████████████████████ VISUAL MUTATORS ████████████████████████████████

# Vectorized for massive performance gain
def unleash_static_demon(matrix, intensity=1.0):
    """Applies vectorized Perlin noise overlay."""
    alpha = 0.2 + 0.7 * intensity
    scale = 5.0 + 75.0 * (1.0 - intensity)
    octaves = random.randint(3, 9)
    persistence = random.uniform(0.3, 0.7)
    lacunarity = random.uniform(1.9, 3.8)
    seed = random.randint(0, 0xFFFF)
    height, width = matrix.shape[:2]
    
    # Create coordinate grids and vectorize the noise function
    x_indices = np.arange(width)
    y_indices = np.arange(height)
    xx, yy = np.meshgrid(x_indices, y_indices)
    
    vectorized_pnoise2 = np.vectorize(noise.pnoise2)
    
    # Generate noise for the entire grid at once
    gray_noise = vectorized_pnoise2(
        yy / scale, xx / scale,
        octaves=octaves, persistence=persistence,
        lacunarity=lacunarity, base=seed
    )
    
    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX)
    colored_noise = cv2.cvtColor(colored_noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if random.random() < 0.5: colored_noise = 255 - colored_noise
    return cv2.addWeighted(matrix, 1 - alpha, colored_noise, alpha, 0)

def disrupt_spacetime_continuum(matrix, intensity=1.0):
    """Shifts random blocks of pixels."""
    max_shift = int(5 + (X_RES / 3) * intensity)
    block_size_max = int(5 + (Y_RES / 2) * intensity)
    num_blocks = int(20 + 80 * intensity)
    height, width = matrix.shape[:2]
    output = matrix.copy()

    for _ in range(num_blocks):
        bh = random.randint(5, block_size_max)
        bw = random.randint(5, block_size_max)
        if height - bh <= 0 or width - bw <= 0: continue
        y, x = random.randint(0, height - bh - 1), random.randint(0, width - bw - 1)
        shift_x, shift_y = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
        target_y = np.clip(y + shift_y, 0, height - bh)
        target_x = np.clip(x + shift_x, 0, width - bw)
        try:
            block = matrix[y:y+bh, x:x+bw]
            output[target_y:target_y+bh, target_x:target_x+bw] = block
        except ValueError:
            pass
    return output

# New Experimental Mutators
def simulate_reaction_diffusion(matrix, intensity=1.0):
    """Simulates Gray-Scott model for organic patterns."""
    global rd_U, rd_V
    scale_factor = 8 # Run simulation on a smaller grid for speed
    h, w = Y_RES // scale_factor, X_RES // scale_factor
    
    # Initialize buffers on first run
    if 'rd_U' not in globals() or rd_U.shape != (h, w):
        rd_U = np.ones((h, w), dtype=np.float32)
        rd_V = np.zeros((h, w), dtype=np.float32)
        # Seed
        r, c = h // 2, w // 2
        rd_U[r-5:r+5, c-5:c+5] = 0.5
        rd_V[r-5:r+5, c-5:c+5] = 0.25

    # Parameters modulated by intensity
    F = 0.03 + 0.03 * intensity
    k = 0.055 + 0.01 * intensity
    Du, Dv = 0.16, 0.08
    dt = 1.0
    
    steps = random.randint(2, 6)
    for _ in range(steps):
        lap_U = cv2.Laplacian(rd_U, cv2.CV_32F)
        lap_V = cv2.Laplacian(rd_V, cv2.CV_32F)
        delta_U = (Du * lap_U - rd_U * rd_V**2 + F * (1 - rd_U)) * dt
        delta_V = (Dv * lap_V + rd_U * rd_V**2 - (F + k) * rd_V) * dt
        rd_U += delta_U
        rd_V += delta_V
        np.clip(rd_U, 0, 1, out=rd_U)
        np.clip(rd_V, 0, 1, out=rd_V)

    # Visualize V component and blend with input matrix
    vis = cv2.normalize(rd_V, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    vis_colored = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    vis_resized = cv2.resize(vis_colored, (X_RES, Y_RES), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(matrix, 0.6, vis_resized, 0.4 + 0.5 * intensity, 0)

# Other mutators... (omitted for brevity, can be ported from the original code if desired)
def shift_color_spectrum(frame, intensity=1.0):
    max_shift = int(5 + 55 * intensity)
    temp_frame = frame.copy()
    for i in range(3):
        shift_x, shift_y = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
        temp_frame[:,:,i] = np.roll(np.roll(frame[:,:,i], shift_y, axis=0), shift_x, axis=1)
    return temp_frame

def imprint_lingering_data(matrix, current_theme):
    """Overlays encrypted, unsettling messages."""
    if random.random() < 0.05: # Low chance per frame
        msg = random.choice(FORBIDDEN_KNOWLEDGE)
        algo = random.choice(list(ENCRYPTION_ALGORITHMS.keys()))
        encoded_msg = ENCRYPTION_ALGORITHMS[algo](msg)
        
        font_scale = random.uniform(0.5, 2.0)
        pos = (random.randint(0, X_RES // 2), random.randint(int(Y_RES * 0.1), int(Y_RES * 0.9)))
        color = random.choice(current_theme["colors"])
        alpha = random.uniform(0.2, 0.7)
        
        overlay = matrix.copy()
        cv2.putText(overlay, encoded_msg, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        return cv2.addWeighted(matrix, 1 - alpha, overlay, alpha, 0)
    return matrix


# ████████████████████████████████ AUDITORY GENERATORS ████████████████████████████████

def synthesize_chaotic_tone(duration_samples, vol, intensity=1.0):
    """Generates audio using a logistic map chaotic oscillator."""
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False)
    
    r = 3.57 + 0.42 * intensity # Growth rate parameter, near edge of chaos
    x = np.zeros(duration_samples)
    x[0] = random.random()
    for i in range(1, duration_samples):
        x[i] = r * x[i-1] * (1 - x[i-1])

    # Use chaotic signal to modulate frequency
    base_freq = random.uniform(80, 800)
    mod_depth = 50 + 1000 * intensity
    freq_mod = base_freq + x * mod_depth
    phase = np.cumsum(2 * np.pi * freq_mod / SAMPLE_RATE)
    
    wave = np.sin(phase) * vol
    return wave.astype(np.float32)

def generate_granular_cloud(duration_samples, vol, intensity=1.0):
    """Creates a sound cloud from tiny audio fragments (grains)."""
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    # Create a base waveform to slice up
    base_freq = random.uniform(100, 500)
    t_base = np.linspace(0, 2, SAMPLE_RATE * 2)
    source_wave = np.sin(2 * np.pi * base_freq * t_base) * (1.0 - np.linspace(0, 1, len(t_base))**2)

    output = np.zeros(duration_samples)
    grain_size = int((SAMPLE_RATE / 100) * (1.5 - intensity)) # smaller grains = harsher sound
    grain_size = max(50, grain_size)
    num_grains = duration_samples // (grain_size // 4)
    
    for _ in range(num_grains):
        start_pos = random.randint(0, duration_samples - grain_size - 1)
        source_start = random.randint(0, len(source_wave) - grain_size - 1)
        grain = source_wave[source_start:source_start+grain_size]
        
        # Apply Hanning window to avoid clicks
        grain *= np.hanning(len(grain))
        output[start_pos:start_pos+grain_size] += grain

    max_val = np.max(np.abs(output))
    if max_val > 0: output /= max_val
    return (output * vol).astype(np.float32)
    
def generate_shepard_risset(duration_samples, vol, intensity=1.0):
    """Generates an unsettling, continuously rising/falling auditory illusion."""
    if duration_samples <= 0:
        return np.array([], dtype=np.float32)

    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False)
    
    # Parameters for the illusion
    num_octaves = 6
    num_tones = 12  # Number of simultaneous sine waves
    base_freq = 40.0
    
    # Intensity controls the speed of the glissando
    rate = (2.0 ** (1.0 / 12.0)) ** (2.0 * intensity * random.choice([-1, 1]))
    
    final_wave = np.zeros(duration_samples)

    for i in range(num_tones):
        # Initial frequency for each tone, spaced out
        initial_freq = base_freq * (2.0 ** (i * num_octaves / num_tones))
        
        # Exponentially rising frequency that wraps around
        freq = initial_freq * (rate ** t)
        freq = base_freq * (2.0 ** (np.log2(freq / base_freq) % num_octaves))

        # Amplitude envelope: tones fade in as they enter the range and fade out as they leave
        amplitude = np.sin(np.pi * np.log2(freq / base_freq) / num_octaves) ** 2
        
        final_wave += np.sin(2 * np.pi * freq * t) * amplitude

    # Normalize and apply volume
    max_val = np.max(np.abs(final_wave))
    if max_val > 1e-6:
        final_wave /= max_val
        
    return (final_wave * vol).astype(np.float32)
    
# Ported audio filters
def apply_distortion(data, intensity=1.0):
    factor = 1.5 + 8.5 * intensity
    return np.clip(data * factor, -0.99, 0.99)

def apply_bitcrush(data, intensity=1.0):
    max_bits = 12
    min_bits = 3
    bits = int(max_bits - (max_bits - min_bits) * intensity)
    if bits >= 16: return data
    steps = 2**(bits - 1)
    return np.round(data * steps) / steps

def apply_spectral_glitch(data, intensity=0.5):
    n = len(data)
    if n < 1024: return data.astype(np.float32)
    spectrum = fft(data.astype(np.float32))
    magnitude, phase = np.abs(spectrum), np.angle(spectrum)
    
    num_glitches = int(n * 0.1 * intensity * random.random())
    indices = random.sample(range(1, n // 2), min(num_glitches, n // 2 - 1))
    
    for idx in indices:
        if random.random() < 0.7: magnitude[idx] = 0
        else: phase[idx] *= -1.0 * random.uniform(0.5, 1.5)
        magnitude[n-idx], phase[n-idx] = magnitude[idx], -phase[idx]

    new_spectrum = magnitude * np.exp(1j * phase)
    glitched_data = np.real(ifft(new_spectrum))
    
    orig_max = np.max(np.abs(data))
    glitched_max = np.max(np.abs(glitched_data))
    if glitched_max > 1e-6: glitched_data *= (orig_max / glitched_max)
    return glitched_data.astype(data.dtype)

# ████████████████████████████████ MAIN EXECUTION CORE ████████████████████████████████

def main():
    print(">>> INITIALIZING FLUX_CORE_v3.7 <<<")
    
    # 1. Setup temporal parameters
    FPS = random.randint(FPS_MIN, FPS_MAX)
    DURATION = random.randint(TEMPORAL_MIN, TEMPORAL_MAX)
    TOTAL_FRAMES = DURATION * FPS
    TOTAL_SAMPLES = DURATION * SAMPLE_RATE
    print(f"[*] Temporal canvas defined: {DURATION}s @ {FPS}fps -> {TOTAL_FRAMES} frames.")

    # 2. Select aesthetic and generate control systems
    aesthetic_choice = random.choice(list(AESTHETICS.keys()))
    current_aesthetic = AESTHETICS[aesthetic_choice]
    print(f"[*] Aesthetic blueprint locked: {aesthetic_choice}")

    # Control Vector: [0]=VisualIntensity, [1]=AudioIntensity, [2]=ColorShift, [3]=TextChance
    control_vector_funcs = generate_control_vector(TOTAL_SAMPLES, SAMPLE_RATE, dimensions=4)
    palette = generate_dynamic_palette(current_aesthetic["dynamic_palette"])
    current_aesthetic["colors"] = palette # Overwrite static colors
    
    # 3. Synthesize the entire audio stream first
    print("[*] Composing auditory stream... this may take a moment.")
    final_audio = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    
    num_audio_events = int(DURATION * random.uniform(5, 15))
    for i in range(num_audio_events):
        event_time_sec = random.uniform(0, DURATION * 0.95)
        event_start_sample = int(event_time_sec * SAMPLE_RATE)
        
        # Get intensity at this point in time
        audio_intensity = control_vector_funcs[1](event_time_sec)
        
        event_duration = random.uniform(0.1, 2.0) * (0.5 + audio_intensity)
        event_duration_samples = int(event_duration * SAMPLE_RATE)
        event_end_sample = min(TOTAL_SAMPLES, event_start_sample + event_duration_samples)
        
        vol = random.uniform(0.1, 0.8) * audio_intensity
        
        generator_func = random.choice([globals()[f] for f in current_aesthetic["audio_generators"]])
        chunk = generator_func(event_end_sample - event_start_sample, vol, intensity=audio_intensity)
        
        if random.random() < 0.6: # Apply filters
            filter_func = random.choice([globals()[f] for f in current_aesthetic["audio_filters"]])
            chunk = filter_func(chunk, intensity=audio_intensity)
        
        final_audio[event_start_sample:event_end_sample] += chunk
        
    # Normalize final audio
    max_amp = np.max(np.abs(final_audio))
    if max_amp > 0: final_audio /= max_amp
    audio_16bit = np.int16(final_audio * 32767)

    # 4. Setup FFMPEG pipe for direct video streaming
    print("[*] Opening direct data pipe to FFMPEG...")
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{X_RES}x{Y_RES}',
        '-pix_fmt', 'bgr24',
        '-r', str(FPS),
        '-i', '-',  # Input from stdin
        '-i', 'pipe:1', # Audio from a secondary pipe
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '18',
        '-c:a', 'aac',
        '-b:a', '192k',
        OUTPUT_ARTIFACT
    ]

    # Use a named pipe for audio to simplify process communication
    audio_pipe_path = "temp_audio_pipe.wav"
    import wave
    with wave.open(audio_pipe_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_16bit.tobytes())

    # Re-adjust command to read audio from the file pipe
    command[11] = audio_pipe_path # change audio input from pipe:1 to file path
    
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # 5. Generate and stream frames
    print("[*] Beginning frame generation and stream injection.")
    prev_matrix = None
    
    for i in range(TOTAL_FRAMES):
        percent_done = (i / TOTAL_FRAMES) * 100
        print(f"\r -> Progress: {percent_done:.1f}%", end="")
        
        current_time_sec = i / FPS
        # Get control values for this frame
        vis_intensity = control_vector_funcs[0](current_time_sec)
        
        matrix = np.full((Y_RES, X_RES, 3), 0, dtype=np.uint8)
        
        # Base layer
        bg_color_index = int(control_vector_funcs[2](current_time_sec) * (len(palette)-1))
        matrix[:] = palette[bg_color_index]
        
        # Apply a stack of mutators
        num_effects = 1 + int(vis_intensity * 4)
        chosen_mutators = random.sample(current_aesthetic["visual_mutators"], min(num_effects, len(current_aesthetic["visual_mutators"])))
        
        for mutator_name in chosen_mutators:
            mutator_func = globals()[mutator_name]
            # Check if function accepts 'intensity' and call accordingly
            sig = inspect.signature(mutator_func)
            if 'intensity' in sig.parameters:
                matrix = mutator_func(matrix, intensity=vis_intensity)
            else:
                 matrix = mutator_func(matrix)


        # Add text/symbols
        if random.random() < control_vector_funcs[3](current_time_sec):
            text = random.choice(current_aesthetic["phrases"] + current_aesthetic["symbols"])
            if random.random() < current_aesthetic["zalgo_chance"]: text = transmogrify_text(text)
            scale = vis_intensity * 5 + 1
            pos = (random.randint(0, X_RES//2), random.randint(0, Y_RES))
            cv2.putText(matrix, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, random.choice(palette), int(scale))
        
        # Apply encrypted message overlay
        matrix = imprint_lingering_data(matrix, current_aesthetic)

        # Stream frame to FFMPEG's stdin
        proc.stdin.write(matrix.tobytes())
        prev_matrix = matrix.copy()

    # 6. Finalize process
    print("\n[*] Finalizing stream... terminating pipe.")
    proc.stdin.close()
    stderr_output = proc.stderr.read().decode()
    proc.wait()
    os.remove(audio_pipe_path) # Clean up audio pipe
    
    if proc.returncode != 0:
        print("\n[!!!] FFMPEG ERROR [!!!]")
        print(stderr_output)
    else:
        print(f"\n[+] SUCCESS: Abyssal echo captured in '{OUTPUT_ARTIFACT}'")

if __name__ == '__main__':
    # Need inspect to check function signatures dynamically
    import inspect
    main()


































