#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MINIMALISTIC_VIDEO — ULTRA CHAOS edition

import numpy as np
import cv2
import random
import time
import inspect
import subprocess
import wave as wave_module

# CONFIGURATION
X_RES, Y_RES = 960, 540   # increase resolution for more detail
FPS = 25
DURATION = 12   # seconds
TOTAL_FRAMES = FPS * DURATION
SAMPLE_RATE = 44100

OUTPUT_VIDEO = "final_extreme_video.mp4"
OUTPUT_AUDIO = "ultra_cryptic_audio.wav"

# Cryptic glyphs & phrases
GLYPHS = [
    "VOID", "0xDEAD", "⚡", "⧫", "∅", "ECHO", "████", "Σ", "∞", "GLITCH"
]

# ——— AUDIO GENERATORS & FILTERS ———

def chaotic_oscillator(n, vol=0.5):
    """Generate chaotic logistic map–driven tone."""
    if n <= 0:
        return np.array([], dtype=np.float32)
    t = np.linspace(0, n / SAMPLE_RATE, n, endpoint=False)
    x = np.zeros(n)
    x[0] = random.random()
    r = 3.8 + 0.2*random.random()
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    base_freq = random.uniform(100, 1000)
    mod = x * random.uniform(400, 2000)
    phase = np.cumsum(2 * np.pi * (base_freq + mod) / SAMPLE_RATE)
    sig = np.sin(phase)
    return (sig * vol).astype(np.float32)

def glitch_filter(data):
    """Randomly zero out frequency bins or invert phases."""
    from scipy.fft import fft, ifft
    n = len(data)
    if n < 256:
        return data
    sp = fft(data)
    mag = np.abs(sp)
    ph = np.angle(sp)
    # introduce random glitches in spectrum
    for _ in range(random.randint(1, 5)):
        idx = random.randint(10, n//2 - 2)
        mag[idx] = 0
        mag[n - idx] = 0
    new = mag * np.exp(1j * ph)
    out = np.real(ifft(new))
    # re-normalize
    mx = np.max(np.abs(out))
    if mx > 1e-6:
        out = out * (np.max(np.abs(data)) / mx)
    return out.astype(np.float32)

def compose_audio():
    total = np.zeros(DURATION * SAMPLE_RATE, dtype=np.float32)
    # overlay multiple chaotic events
    for _ in range(random.randint(4, 8)):
        start = random.randint(0, len(total) - 1)
        length = random.randint(SAMPLE_RATE//2, SAMPLE_RATE * 2)
        chunk = chaotic_oscillator(length, vol=random.uniform(0.2, 0.8))
        if random.random() < 0.7:
            chunk = glitch_filter(chunk)
        end = min(len(total), start + length)
        total[start:end] += chunk[: end - start]
    # normalize
    mx = np.max(np.abs(total))
    if mx > 0:
        total /= mx
    return np.int16(total * 32767)

# ——— VISUAL MUTATORS ———

def glitch_shift(frame):
    """Cut random horizontal slices and shift them."""
    h, w = frame.shape[:2]
    out = frame.copy()
    for _ in range(random.randint(3, 8)):
        y = random.randint(0, h-1)
        height = random.randint(1, max(1, h//10))
        shift = random.randint(-w//4, w//4)
        slice_ = frame[y:y+height, :]
        out[y:y+height, :] = np.roll(slice_, shift, axis=1)
    return out

def color_pulse(frame, tnorm):
    """Modulate frame colors with time-based sin pulses."""
    factor = 0.5 + 0.5 * np.sin(2 * np.pi * tnorm * random.uniform(0.5, 1.5))
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def overlay_glyph(frame):
    """Overlay cryptic glyphs with random transforms."""
    if random.random() < 0.5:
        text = random.choice(GLYPHS)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = random.uniform(0.5, 2.0)
        thickness = random.randint(1, 3)
        w, h = cv2.getTextSize(text, font, scale, thickness)[0]
        x = random.randint(0, frame.shape[1] - w - 1)
        y = random.randint(h, frame.shape[0] - 1)
        color = tuple(int(c) for c in np.random.randint(0, 256, 3))
        matrix = cv2.getRotationMatrix2D((x + w/2, y - h/2), random.uniform(-45, 45), 1.0)
        tmp = frame.copy()
        cv2.putText(tmp, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        warped = cv2.warpAffine(tmp, matrix, (frame.shape[1], frame.shape[0]))
        alpha = random.uniform(0.2, 0.7)
        frame = cv2.addWeighted(frame, 1 - alpha, warped, alpha, 0)
    return frame

# ——— MAIN & FFmpeg Streaming ———

def stream_video(audio_wav_path, audio_bytes, audio_npy):
    # write audio WAV
    with wave_module.open(audio_wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{X_RES}x{Y_RES}',
        '-r', str(FPS), '-i', '-',
        '-i', audio_wav_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
        '-c:a', 'aac', '-b:a', '192k',
        OUTPUT_VIDEO
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for i in range(TOTAL_FRAMES):
        tnorm = i / TOTAL_FRAMES
        frame = np.zeros((Y_RES, X_RES, 3), dtype=np.uint8)

        # base noise rectangles
        for _ in range(random.randint(5, 12)):
            x1, y1 = random.randint(0, X_RES), random.randint(0, Y_RES)
            x2, y2 = random.randint(0, X_RES), random.randint(0, Y_RES)
            c = tuple(int(c) for c in np.random.randint(0, 256, 3))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, thickness=-1)

        # apply mutators in random order
        for mut in [glitch_shift, color_pulse, overlay_glyph]:
            frame = mut(frame, tnorm) if mut.__code__.co_argcount == 2 else mut(frame)

        # ensure contiguous + correct dtype
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != (Y_RES, X_RES, 3):
            # fallback if shape mismatches
            frame = cv2.resize(frame, (X_RES, Y_RES))
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    stderr = proc.stderr.read().decode()
    proc.wait()
    if proc.returncode != 0:
        print("FFmpeg error:", stderr)

def main():
    print(">>> ULTRA CHAOS minimalistic_video starting <<<")
    audio_data = compose_audio()
    audio_bytes = audio_data.tobytes()
    stream_video(OUTPUT_AUDIO, audio_bytes, audio_data)
    print("Wrote video to", OUTPUT_VIDEO)

if __name__ == '__main__':
    main()




































