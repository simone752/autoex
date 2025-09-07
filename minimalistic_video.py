#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ARCHITECTURE: FLUX_CORE_v3.7 (SAFE MODE)
# STATUS: STABILIZED FOR CI ENVIRONMENTS
# PURPOSE: To tear a hole in reality without segfaults.
#
# NOTE: This safe mode lowers resolution, limits mutators, and validates frame buffers

import numpy as np
import cv2
import random
import os
import time
import noise
import base64
import subprocess
import inspect
import wave
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft

# █████████████████ CONFIGURATION █████████████████
X_RES, Y_RES = 640, 360   # Reduced resolution for stability
TEMPORAL_MIN, TEMPORAL_MAX = 10, 15
FPS_MIN, FPS_MAX = 20, 30
OUTPUT_ARTIFACT = "abyss_echo.mp4"
SAMPLE_RATE = 22050       # Lower sample rate for lighter audio

# █████████████████ AESTHETICS █████████████████
AESTHETICS = {
    "SAFE": {
        "dynamic_palette": "analogous_decay",
        "phrases": ["NULL", "VOID", "STATIC"],
        "symbols": ["▌", "█", "▓", "▒", "░"],
        "zalgo_chance": 0.3,
        # Mutators: only lightweight ones kept
        "visual_mutators": ["shift_color_spectrum"],
        "audio_generators": ["synthesize_chaotic_tone"],
        "audio_filters": ["apply_distortion"]
    }
}

# █████████████████ UTILS █████████████████
def hsv_to_bgr(h, s, v):
    hsv_color = np.uint8([[[h, s, v]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0])

def generate_dynamic_palette(style="analogous_decay"):
    base_hue = random.randint(0, 179)
    base_sat, base_val = 200, 200
    c1 = hsv_to_bgr(base_hue, base_sat, base_val)
    c2 = hsv_to_bgr((base_hue+30)%180, 200, 180)
    c3 = hsv_to_bgr((base_hue-30)%180, 200, 180)
    c4 = (0,0,0)
    return [c1, c2, c3, c4]

# █████████████████ MUTATORS █████████████████
def shift_color_spectrum(frame, intensity=1.0):
    max_shift = int(5 + 20 * intensity)
    temp_frame = frame.copy()
    for i in range(3):
        shift_x, shift_y = random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)
        temp_frame[:,:,i] = np.roll(np.roll(frame[:,:,i], shift_y, axis=0), shift_x, axis=1)
    return temp_frame

# █████████████████ AUDIO █████████████████
def synthesize_chaotic_tone(duration_samples, vol, intensity=1.0):
    if duration_samples <= 0: return np.array([], dtype=np.float32)
    t = np.linspace(0, duration_samples / SAMPLE_RATE, duration_samples, endpoint=False)
    base_freq = random.uniform(80, 400)
    wave = np.sin(2*np.pi*base_freq*t) * vol
    return wave.astype(np.float32)

def apply_distortion(data, intensity=1.0):
    factor = 1.5 + 2.0 * intensity
    return np.clip(data * factor, -0.99, 0.99)

# █████████████████ CONTROL VECTOR █████████████████
def generate_control_vector(duration_samples, sample_rate, dimensions=4):
    duration_sec = duration_samples / sample_rate
    times = np.linspace(0, duration_sec, max(2,int(duration_sec*5)))
    values = np.random.rand(len(times), dimensions)
    return [interp1d(times, values[:,d], kind='linear', bounds_error=False, fill_value=(0.5,0.5)) for d in range(dimensions)]

# █████████████████ MAIN █████████████████
def main():
    print(">>> INITIALIZING FLUX_CORE_v3.7 (SAFE MODE) <<<")

    FPS = random.randint(FPS_MIN, FPS_MAX)
    DURATION = random.randint(TEMPORAL_MIN, TEMPORAL_MAX)
    TOTAL_FRAMES = DURATION * FPS
    TOTAL_SAMPLES = DURATION * SAMPLE_RATE
    print(f"[*] Temporal canvas: {DURATION}s @ {FPS}fps -> {TOTAL_FRAMES} frames.")

    aesthetic_choice = "SAFE"
    current_aesthetic = AESTHETICS[aesthetic_choice]
    palette = generate_dynamic_palette(current_aesthetic["dynamic_palette"])
    current_aesthetic["colors"] = palette

    # Audio generation
    print("[*] Composing audio stream...")
    final_audio = np.zeros(TOTAL_SAMPLES, dtype=np.float32)
    chunk = synthesize_chaotic_tone(TOTAL_SAMPLES, vol=0.5, intensity=0.7)
    final_audio += apply_distortion(chunk, 0.5)

    max_amp = np.max(np.abs(final_audio))
    if max_amp > 0: final_audio /= max_amp
    audio_16bit = np.int16(final_audio * 32767)

    audio_path = "temp_audio.wav"
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_16bit.tobytes())

    command = [
        'ffmpeg','-y',
        '-f','rawvideo','-vcodec','rawvideo',
        '-s',f'{X_RES}x{Y_RES}','-pix_fmt','bgr24',
        '-r',str(FPS),'-i','-',
        '-i',audio_path,
        '-c:v','libx264','-pix_fmt','yuv420p','-preset','fast','-crf','23',
        '-c:a','aac','-b:a','128k',
        OUTPUT_ARTIFACT
    ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    print("[*] Streaming frames to ffmpeg...")
    for i in range(TOTAL_FRAMES):
        percent_done = (i/TOTAL_FRAMES)*100
        print(f"\r -> Progress: {percent_done:.1f}%", end="")

        frame = np.full((Y_RES, X_RES, 3), 0, dtype=np.uint8)
        bg_color = palette[int(random.random()*(len(palette)-1))]
        frame[:] = bg_color

        for mutator_name in current_aesthetic["visual_mutators"]:
            mutator_func = globals()[mutator_name]
            frame = mutator_func(frame, intensity=0.5)

        # Ensure correct format before writing
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != (Y_RES, X_RES, 3):
            raise ValueError(f"Frame shape mismatch: {frame.shape}")
        proc.stdin.write(frame.tobytes())
        proc.stdin.flush()

    print("\n[*] Finalizing...")
    proc.stdin.close()
    stderr_output = proc.stderr.read().decode()
    proc.wait()
    os.remove(audio_path)

    if proc.returncode != 0:
        print("[!!!] FFMPEG ERROR [!!!]")
        print(stderr_output)
    else:
        print(f"[+] SUCCESS: Video written to '{OUTPUT_ARTIFACT}'")

if __name__ == '__main__':
    main()

