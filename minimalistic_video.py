#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FIXED VERSION of FLUX_CORE_v3.7 (practical & robust streaming to ffmpeg)
# - Key fixes: correct ffmpeg audio input handling, safer subprocess management,
#   more robust I/O / error handling, minor bugfixes & small performance tweaks.

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
import inspect
import shutil
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft

# ---------------------- CONFIGURATION ----------------------
X_RES, Y_RES = 1920, 1080
TEMPORAL_MIN, TEMPORAL_MAX = 13, 33
FPS_MIN, FPS_MAX = 24, 60
OUTPUT_ARTIFACT = "abyss_echo.mp4"
FREQ_CEILING = 22050
FREQ_FLOOR = 16
SAMPLE_RATE = 44100

# ---------------------- AESTHETICS & HELPERS ----------------------

def transmogrify_text(text):
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
        "visual_mutators": ["unleash_static_demon", "disrupt_spacetime_continuum", "shift_color_spectrum", "simulate_reaction_diffusion", "render_ascii_ghosts"],
        "audio_generators": ["synthesize_chaotic_tone", "generate_granular_cloud", "generate_shepard_risset"],
        "audio_filters": ["apply_bitcrush", "apply_spectral_glitch", "apply_distortion"],
    },
    "FLESH_ALGORITHM": {
        "dynamic_palette": "analogous_decay",
        "phrases": ["CONSUME", "INFECT", "MUTATE", "PULSATE", "WRITHE", "MERGE", "ASSIMILATE"],
        "symbols": ["Ø", "§", " संक्रमित", "🦠", "🍄", "🦴", "👁️‍🗨️", "∬", "⌬", "♾️"],
        "zalgo_chance": 0.7,
        "visual_mutators": ["unleash_static_demon", "simulate_reaction_diffusion", "disrupt_spacetime_continuum", "shift_color_spectrum", "imprint_crt_ghost"],
        "audio_generators": ["synthesize_chaotic_tone", "generate_granular_cloud"],
        "audio_filters": ["apply_distortion", "apply_spectral_glitch"],
    },
}

# ---------------------- CONTROL / PALETTE ----------------------

def generate_control_vector(duration_samples, sample_rate, dimensions=4):
    duration_sec = duration_samples / sample_rate
    num_points = int(duration_sec * 20)
    if num_points < 2:
        num_points = 2

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
            control_vectors[:, dim] = (profile - min_val) / (max_val - min_val)
        else:
            control_vectors[:, dim] = np.full(num_points, 0.5)

    profile_times = np.linspace(0, duration_sec, num_points)
    return [interp1d(profile_times, control_vectors[:, dim], kind='cubic', bounds_error=False, fill_value=(control_vectors[0, dim], control_vectors[-1, dim])) for dim in range(dimensions)]


def hsv_to_bgr(h, s, v):
    hsv_color = np.uint8([[[h, s, v]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0])


def generate_dynamic_palette(style="triadic_dissonance"):
    base_hue = random.randint(0, 179)
    base_sat = random.randint(180, 255)
    base_val = random.randint(180, 255)
    c1 = hsv_to_bgr(base_hue, base_sat, base_val)
    if style == "triadic_dissonance":
        h2 = (base_hue + 60 + random.randint(-15, 15)) % 180
        h3 = (base_hue + 120 + random.randint(-15, 15)) % 180
    else:
        h2 = (base_hue + 30 + random.randint(-10, 10)) % 180
        h3 = (base_hue - 30 + random.randint(-10, 10)) % 180
    c2 = hsv_to_bgr(h2, random.randint(150, 255), random.randint(100, 255))
    c3 = hsv_to_bgr(h3, random.randint(150, 255), random.randint(100, 220))
    c4 = hsv_to_bgr(random.randint(0,179), random.randint(0,50), random.randint(0,40))
    return [c1, c2, c3, c4]

# ---------------------- VISUAL MUTATORS ----------------------

def unleash_static_demon(matrix, intensity=1.0):
    alpha = 0.2 + 0.7 * float(np.clip(intensity, 0, 1))
    scale = 5.0 + 75.0 * (1.0 - float(np.clip(intensity, 0, 1)))
    octaves = random.randint(3, 6)
    persistence = random.uniform(0.3, 0.7)
    lacunarity = random.uniform(1.9, 3.8)
    seed = random.randint(0, 0xFFFF)
    height, width = matrix.shape[:2]

    # create a reasonably sized grid to avoid extreme slowdowns
    yy, xx = np.meshgrid(np.linspace(0, height, height), np.linspace(0, width, width), indexing='ij')
    vectorized_pnoise2 = np.vectorize(lambda a,b: noise.pnoise2(a/scale, b/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed))
    gray_noise = vectorized_pnoise2(yy, xx)
    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored_noise = cv2.cvtColor(colored_noise, cv2.COLOR_GRAY2BGR)
    if random.random() < 0.5:
        colored_noise = 255 - colored_noise
    return cv2.addWeighted(matrix, 1 - alpha, colored_noise, alpha, 0)


def disrupt_spacetime_continuum(matrix, intensity=1.0):
    max_shift = int(5 + (X_RES / 3) * float(np.clip(intensity, 0, 1)))
    block_size_max = int(5 + (Y_RES / 2) * float(np.clip(intensity, 0, 1)))
    num_blocks = max(1, int(20 + 80 * float(np.clip(intensity, 0, 1))))
    height, width = matrix.shape[:2]
    output = matrix.copy()
    for _ in range(num_blocks):
        bh = random.randint(5, max(6, block_size_max))
        bw = random.randint(5, max(6, block_size_max))
        if height - bh <= 0 or width - bw <= 0:
            continue
        y, x = random.randint(0, height - bh - 1), random.randint(0, width - bw - 1)
        shift_x, shift_y = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
        target_y = np.clip(y + shift_y, 0, height - bh)
        target_x = np.clip(x + shift_x, 0, width - bw)
        try:
            block = matrix[y:y+bh, x:x+bw]
            output[target_y:target_y+bh, target_x:target_x+bw] = block
        except Exception:
            pass
    return output


def simulate_reaction_diffusion(matrix, intensity=1.0):
    global rd_U, rd_V
    scale_factor = 8
    h, w = Y_RES // scale_factor, X_RES // scale_factor
    if 'rd_U' not in globals() or rd_U.shape != (h, w):
        rd_U = np.ones((h, w), dtype=np.float32)
        rd_V = np.zeros((h, w), dtype=np.float32)
        r, c = h // 2, w // 2
        rd_U[max(0,r-5):r+5, max(0,c-5):c+5] = 0.5
        rd_V[max(0,r-5):r+5, max(0,c-5):c+5] = 0.25
    F = 0.03 + 0.03 * float(np.clip(intensity, 0, 1))
    k = 0.055 + 0.01 * float(np.clip(intensity, 0, 1))
    Du, Dv = 0.16, 0.08
    dt = 1.0
    steps = random.randint(1, 5)
    for _ in range(steps):
        lap_U = cv2.Laplacian(rd_U, cv2.CV_32F)
        lap_V = cv2.Laplacian(rd_V, cv2.CV_32F)
        delta_U = (Du * lap_U - rd_U * rd_V**2 + F * (1 - rd_U)) * dt
        delta_V = (Dv * lap_V + rd_U * rd_V**2 - (F + k) * rd_V) * dt
        rd_U += delta_U
        rd_V += delta_V
        np.clip(rd_U, 0, 1, out=rd_U)
        np.clip(rd_V, 0, 1, out=rd_V)
    vis = cv2.normalize(rd_V, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    vis_colored = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    vis_resized = cv2.resize(vis_colored, (X_RES, Y_RES), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(matrix, 0.6, vis_resized, 0.4 + 0.5 * float(np.clip(intensity,0,1)), 0)


def shift_color_spectrum(frame, intensity=1.0):
    max_shift = int(5 + 55 * float(np.clip(intensity,0,1)))
    temp_frame = frame.copy()
    for i in range(3):
        shift_x, shift_y = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
        temp_frame[:,:,i] = np.roll(np.roll(frame[:,:,i], shift_y, axis=0), shift_x, axis=1)
    return temp_frame


def imprint_lingering_data(matrix, current_theme):
    if random.random() < 0.05:
        msg = random.choice(FORBIDDEN_KNOWLEDGE)
        algo = random.choice(list(ENCRYPTION_ALGORITHMS.keys()))
        encoded_msg = ENCRYPTION_ALGORITHMS[algo](msg)
        font_scale = random.uniform(0.5, 2.0)
        pos = (random.randint(0, X_RES // 2), random.randint(int(Y_RES * 0.1), int(Y_RES * 0.9)))
        color = random.choice(current_theme.get("colors", [(255,255,255)]))
        alpha = random.uniform(0.2, 0.7)
        overlay = matrix.copy()
        cv2.putText(overlay, encoded_msg, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        return cv2.addWeighted(matrix, 1 - alpha, overlay, alpha, 0)
    return matrix

# ---------------------- AUDIO GENERATORS & FILTERS ----------------------

def synthesize_chaotic_tone(duration_samples, vol, intensity=1.0):
    if duration_samples <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False)
    r = 3.57 + 0.42 * float(np.clip(intensity,0,1))
    x = np.zeros(duration_samples)
    x[0] = random.random()
    for i in range(1, duration_samples):
        x[i] = r * x[i-1] * (1 - x[i-1])
    base_freq = random.uniform(80, 800)
    mod_depth = 50 + 1000 * float(np.clip(intensity,0,1))
    freq_mod = base_freq + x * mod_depth
    phase = np.cumsum(2 * np.pi * freq_mod / SAMPLE_RATE)
    wave = np.sin(phase) * vol
    return wave.astype(np.float32)


def generate_granular_cloud(duration_samples, vol, intensity=1.0):
    if duration_samples <= 0:
        return np.array([], dtype=np.float32)
    base_freq = random.uniform(100, 500)
    t_base = np.linspace(0, 2, int(SAMPLE_RATE * 2))
    source_wave = np.sin(2 * np.pi * base_freq * t_base) * (1.0 - np.linspace(0, 1, len(t_base))**2)
    output = np.zeros(duration_samples)
    grain_size = int((SAMPLE_RATE / 100) * max(0.4, (1.5 - float(np.clip(intensity,0,1)))))
    grain_size = max(50, grain_size)
    num_grains = max(1, duration_samples // max(1, (grain_size // 4)))
    for _ in range(num_grains):
        if duration_samples - grain_size - 1 <= 0:
            break
        start_pos = random.randint(0, duration_samples - grain_size - 1)
        source_start = random.randint(0, len(source_wave) - grain_size - 1)
        grain = source_wave[source_start:source_start+grain_size]
        grain *= np.hanning(len(grain))
        output[start_pos:start_pos+grain_size] += grain
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output /= max_val
    return (output * vol).astype(np.float32)


def generate_shepard_risset(duration_samples, vol, intensity=1.0):
    if duration_samples <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False)
    num_octaves = 6
    num_tones = 12
    base_freq = 40.0
    rate = (2.0 ** (1.0 / 12.0)) ** (2.0 * float(np.clip(intensity,0,1)) * random.choice([-1, 1]))
    final_wave = np.zeros(duration_samples)
    for i in range(num_tones):
        initial_freq = base_freq * (2.0 ** (i * num_octaves / num_tones))
        freq = initial_freq * (rate ** t)
        freq = base_freq * (2.0 ** (np.log2(freq / base_freq) % num_octaves))
        amplitude = np.sin(np.pi * np.log2(freq / base_freq) / num_octaves) ** 2
        final_wave += np.sin(2 * np.pi * freq * t) * amplitude
    max_val = np.max(np.abs(final_wave))
    if max_val > 1e-6:
        final_wave /= max_val
    return (final_wave * vol).astype(np.float32)


def apply_distortion(data, intensity=1.0):
    factor = 1.5 + 8.5 * float(np.clip(intensity,0,1))
    return np.clip(data * factor, -0.99, 0.99)


def apply_bitcrush(data, intensity=1.0):
    max_bits = 12
    min_bits = 3
    bits = int(max_bits - (max_bits - min_bits) * float(np.clip(intensity,0,1)))
    if bits >= 16:
        return data
    steps = 2**max(1, (bits - 1))
    return np.round(data * steps) / steps


def apply_spectral_glitch(data, intensity=0.5):
    n = len(data)
    if n < 1024:
        return data.astype(np.float32)
    spectrum = fft(data.astype(np.float32))
    magnitude, phase = np.abs(spectrum), np.angle(spectrum)
    num_glitches = int(n * 0.05 * float(np.clip(intensity,0,1)) * random.random())
    indices = random.sample(range(1, n // 2), min(num_glitches, max(1, n // 2 - 1)))
    for idx in indices:
        if random.random() < 0.7:
            magnitude[idx] = 0
        else:
            phase[idx] *= -1.0 * random.uniform(0.5, 1.5)
        magnitude[n-idx], phase[n-idx] = magnitude[idx], -phase[idx]
    new_spectrum = magnitude * np.exp(1j * phase)
    glitched_data = np.real(ifft(new_spectrum))
    orig_max = np.max(np.abs(data))
    glitched_max = np.max(np.abs(glitched_data))
    if glitched_max > 1e-6 and orig_max > 1e-6:
        glitched_data *= (orig_max / glitched_max)
    return glitched_data.astype(np.float32)

# ---------------------- MAIN ----------------------

def main():
    print(">>> INITIALIZING FLUX_CORE_v3.7 (FIXED) <<<")

    FPS = random.randint(FPS_MIN, FPS_MAX)
    DURATION = random.randint(TEMPORAL_MIN, TEMPORAL_MAX)
    TOTAL_FRAMES = int(DURATION * FPS)
    TOTAL_SAMPLES = int(DURATION * SAMPLE_RATE)
    print(f"[*] Temporal canvas: {DURATION}s @ {FPS}fps -> {TOTAL_FRAMES} frames.")

    aesthetic_choice = random.choice(list(AESTHETICS.keys()))
    current_aesthetic = AESTHETICS[aesthetic_choice]
    print(f"[*] Aesthetic: {aesthetic_choice}")

    control_vector_funcs = generate_control_vector(TOTAL_SAMPLES, SAMPLE_RATE, dimensions=4)
    palette = generate_dynamic_palette(current_aesthetic["dynamic_palette"])
    current_aesthetic["colors"] = palette

    print("[*] Composing audio stream...")
    final_audio = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    num_audio_events = max(1, int(DURATION * random.uniform(5, 15)))
    for i in range(num_audio_events):
        event_time_sec = random.uniform(0, max(0.0, DURATION * 0.95))
        event_start_sample = int(event_time_sec * SAMPLE_RATE)
        audio_intensity = float(control_vector_funcs[1](event_time_sec))
        event_duration = random.uniform(0.1, 2.0) * (0.5 + audio_intensity)
        event_duration_samples = int(event_duration * SAMPLE_RATE)
        event_end_sample = min(TOTAL_SAMPLES, event_start_sample + event_duration_samples)
        vol = random.uniform(0.1, 0.8) * audio_intensity
        generator_func = random.choice([globals()[f] for f in current_aesthetic["audio_generators"]])
        chunk = generator_func(max(0, event_end_sample - event_start_sample), vol, intensity=audio_intensity)
        if random.random() < 0.6 and len(chunk) > 0:
            filter_func = random.choice([globals()[f] for f in current_aesthetic["audio_filters"]])
            chunk = filter_func(chunk, intensity=audio_intensity)
        final_audio[event_start_sample:event_end_sample] += chunk[:event_end_sample-event_start_sample]

    max_amp = np.max(np.abs(final_audio))
    if max_amp > 0:
        final_audio /= max_amp
    audio_16bit = np.int16(np.clip(final_audio, -1.0, 1.0) * 32767)

    # Check ffmpeg availability
    if shutil.which('ffmpeg') is None:
        print("[ERROR] ffmpeg not found in PATH. Install ffmpeg and retry.")
        return

    # Write audio to a temporary wav file
    audio_file = "temp_audio.wav"
    import wave
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_16bit.tobytes())

    # Build ffmpeg command: video from stdin (-i -), audio from file (second -i)
    command = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{X_RES}x{Y_RES}', '-pix_fmt', 'bgr24', '-r', str(FPS),
        '-i', '-',  # video stdin
        '-i', audio_file,  # audio file
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '18',
        '-c:a', 'aac', '-b:a', '192k',
        OUTPUT_ARTIFACT
    ]

    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"[ERROR] Failed to launch ffmpeg: {e}")
        os.remove(audio_file)
        return

    print("[*] Streaming frames to ffmpeg...")
    try:
        for i in range(TOTAL_FRAMES):
            percent_done = (i / max(1, TOTAL_FRAMES)) * 100
            print(f"\r -> Progress: {percent_done:.1f}%", end='', flush=True)
            current_time_sec = i / FPS
            vis_intensity = float(control_vector_funcs[0](current_time_sec))
            matrix = np.full((Y_RES, X_RES, 3), 0, dtype=np.uint8)
            bg_color_index = int(np.clip(control_vector_funcs[2](current_time_sec), 0, 0.9999) * (len(palette)-1))
            matrix[:] = palette[bg_color_index]
            num_effects = 1 + int(vis_intensity * 4)
            chosen_mutators = random.sample(current_aesthetic["visual_mutators"], min(num_effects, len(current_aesthetic["visual_mutators"])))
            for mutator_name in chosen_mutators:
                mutator_func = globals().get(mutator_name)
                if mutator_func is None:
                    continue
                try:
                    sig = inspect.signature(mutator_func)
                    if 'intensity' in sig.parameters:
                        matrix = mutator_func(matrix, intensity=vis_intensity)
                    else:
                        matrix = mutator_func(matrix)
                except Exception:
                    pass
            if random.random() < float(control_vector_funcs[3](current_time_sec)):
                text = random.choice(current_aesthetic.get("phrases", []) + current_aesthetic.get("symbols", []))
                if random.random() < current_aesthetic.get("zalgo_chance", 0):
                    text = transmogrify_text(text)
                scale = max(0.5, vis_intensity * 5 + 0.5)
                pos = (random.randint(0, X_RES//2), random.randint(int(Y_RES*0.1), Y_RES-20))
                cv2.putText(matrix, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, random.choice(palette), int(max(1, scale*2)), cv2.LINE_AA)
            matrix = imprint_lingering_data(matrix, current_aesthetic)
            # write frame
            try:
                proc.stdin.write(matrix.tobytes())
            except BrokenPipeError:
                # ffmpeg died; break out
                break
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

    stderr_output = proc.stderr.read().decode() if proc.stderr is not None else ""
    proc.wait()
    # cleanup
    try:
        if os.path.exists(audio_file):
            os.remove(audio_file)
    except Exception:
        pass

    if proc.returncode != 0:
        print("\n[!!!] FFMPEG ERROR [!!!]")
        print(stderr_output)
    else:
        print(f"\n[+] SUCCESS: '{OUTPUT_ARTIFACT}' created")

if __name__ == '__main__':
    main()
