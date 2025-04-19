import numpy as np
import cv2
import pygame # Still used for mixer init check if needed, though audio gen is numpy based
import random
import string
import wave
import os
import math
import time # For potential time-based effects
import noise # pip install noise

# --- Configuration ---
WIDTH, HEIGHT = 640, 480
MIN_DURATION, MAX_DURATION = 15, 35 # Increased duration range
MIN_FPS, MAX_FPS = 4, 10 # Lower FPS range for jerkier feel
OUTPUT_FILE = "extreme_video.mp4"

# --- Thematic Elements ---
THEMES = {
    "Digital Decay": {
        "colors": [
            [(0, 0, 0), (0, 255, 0), (50, 50, 50)], # Glitch Green
            [(20, 0, 0), (255, 255, 255), (0, 100, 100)], # Corrupted White
            [(10, 10, 30), (0, 0, 255), (100, 100, 0)], # Blue Screen Glare
        ],
        "words": ["ERROR", "CORRUPT", "FRAGMENT", "NULL", "VOID", "DELETE", "SEGFAULT", "BUFFER", "OVERFLOW", "STATIC", "LOST", "SIGNAL", "404"],
        "symbols": ["▌", "█", "░", "▒", "▓", "0xDEADBEEF", "NaN", "...", "//", "%", "&", "*", "!", "?", "01101100011011110111001101110100"], # 'lost' in binary
        "audio_freq_range": (50, 1500),
        "audio_noise_types": ["white", "glitch", "static"],
        "audio_features": ["stutter", "bitcrush", "clicks", "panning"]
    },
    "Organic Corruption": {
        "colors": [
            [(40, 10, 10), (200, 50, 50), (100, 150, 80)], # Fleshy Tones
            [(20, 30, 10), (50, 100, 30), (180, 180, 150)], # Mold/Decay
            [(0, 0, 0), (180, 0, 30), (255, 100, 100)], # Blood/Viscera
        ],
        "words": ["GROW", "DECAY", "CONSUME", "INFECT", "MUTATE", "FLESH", "BONE", "ROT", "SPORE", "INSIDE", "BREATHE", "PULSATE", "WRITHE"],
        "symbols": ["Ø", "§", " संक्रमित", "تلوث", "🦠", "🍄", "🦴", "👁️‍🗨️", "〰", "~~", "...", "呼吸"], # Infected (Hindi), Contamination (Arabic), Microbe, Mushroom, Bone, Eye in speech bubble, Wave, Breathe (Chinese)
        "audio_freq_range": (40, 800),
        "audio_noise_types": ["brown", "squelch", "heartbeat", "breathing"],
        "audio_features": ["wet_sounds", "slow_lfo", "dissonance", "low_rumble"]
    },
    "Cosmic Horror": {
        "colors": [
            [(0, 0, 10), (100, 0, 150), (200, 200, 255)], # Deep Space Purple
            [(10, 0, 0), (0, 50, 50), (255, 150, 0)], # Nebula Orange/Teal
            [(0, 0, 0), (255, 255, 255), (10, 10, 10)], # Stark Void
        ],
        "words": ["BEYOND", "VOID", "STARS", "MADNESS", "UNKNOWN", "ELDRITCH", "ENTITY", "SILENCE", "WATCHING", "BELOW", "ABYSS", "UNFOLD", "SEE"],
        "symbols": ["★", "☆", "✶", "✡", "♮", "♄", "♃", "☉", "☿", "♁", "∝", "∫", "∇", "∞", "⋮"], # Stars, Planets, Occult/Math symbols, Infinity, Ellipsis
        "audio_freq_range": (20, 20000), # Wide range
        "audio_noise_types": ["pink", "deep_drone", "radio_interference", "void_silence"],
        "audio_features": ["shepard_tone", "extreme_pitch", "reverb", "panning", "random_bursts"]
    },
    "Subconscious Echoes": {
         "colors": [
            [(100, 100, 120), (200, 200, 220), (50, 50, 70)], # Washed out / Dreamlike
            [(20, 20, 20), (255, 100, 0), (0, 100, 255)], # Contrasting memories
            [(150, 150, 150), (80, 80, 80), (200, 200, 200)], # Monochrome fog
        ],
        "words": ["REMEMBER", "FORGET", "DREAM", "LOOP", "ECHO", "MIRROR", "SHADOW", "WHO", "WHY", "LOST", "AGAIN", "TRAPPED", "FACE"],
        "symbols": ["?", "¿", " திரும்பத்திரும்ப", "繰り返す", "≡", "∽", "◌", "()", "...", "≠", "||", "⚰️"], # Tamil 'repeatedly', Japanese 'repeat', equivalence, similarity, dotted circle, coffin
        "audio_freq_range": (100, 1200),
        "audio_noise_types": ["whispers", "filtered_noise", "reversed_sounds", "tape_hiss"],
        "audio_features": ["delay", "reverb", "subtle_melody", "panning", "fading"]
    }
}

# --- Font Configuration (Optional) ---
# Download some .ttf fonts and place them in the same directory or specify path
# Ensure the font supports the symbols you use, or it might render squares.
FONT_PATHS = [
    "DejaVuSans.ttf", # Standard fallback (often available on Linux)
    "Arial.ttf",      # Standard fallback (often available on Windows)
    # Add paths to more abstract or unsettling fonts if you have them
    "WEIRD_FONT.ttf", # Example
]
AVAILABLE_FONTS = [f for f in FONT_PATHS if os.path.exists(f)]
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX # Fallback if no TTF fonts work

# Initialize pygame mixer (optional, mainly for sound init check/potential future use)
# os.environ["SDL_AUDIODRIVER"] = "dummy" # Use if running headless
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2) # Use 2 channels for stereo
except pygame.error as e:
    print(f"Pygame mixer init failed (might be normal in some envs): {e}")
    # Continue without pygame mixer features if any were planned

# --- Enhanced Glitch Functions ---

def apply_perlin_noise(frame, alpha=0.2, scale=10.0, octaves=4, persistence=0.5, lacunarity=2.0):
    """Applies Perlin noise overlay."""
    height, width = frame.shape[:2]
    gray_noise = np.zeros((height, width))

    # Generate Perlin noise - adjust scale for detail level
    scale = random.uniform(5.0, 50.0) # Randomize scale
    octaves = random.randint(2, 6)
    persistence = random.uniform(0.3, 0.7)
    lacunarity = random.uniform(1.5, 2.5)
    seed = random.randint(0, 100) # Different noise pattern per frame/run

    for i in range(height):
        for j in range(width):
            gray_noise[i][j] = noise.pnoise2(i/scale,
                                             j/scale,
                                             octaves=octaves,
                                             persistence=persistence,
                                             lacunarity=lacunarity,
                                             base=seed)

    # Normalize noise to 0-255 and make it color
    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX)
    colored_noise = cv2.cvtColor(colored_noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Randomly invert noise color sometimes
    if random.random() < 0.2:
         colored_noise = 255 - colored_noise

    # Blend with frame
    return cv2.addWeighted(frame, 1 - alpha, colored_noise, alpha, 0)

def apply_block_shift(frame, max_shift=30, block_size=50):
    """Shifts random blocks of the image horizontally or vertically."""
    height, width = frame.shape[:2]
    temp_frame = frame.copy()
    num_blocks = random.randint(5, 20)

    for _ in range(num_blocks):
        bh = random.randint(10, block_size)
        bw = random.randint(10, block_size)
        y = random.randint(0, height - bh - 1)
        x = random.randint(0, width - bw - 1)

        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)

        target_y = np.clip(y + shift_y, 0, height - bh)
        target_x = np.clip(x + shift_x, 0, width - bw)

        # Swap blocks - can cause interesting overlaps
        block = frame[y:y+bh, x:x+bw].copy()
        temp_frame[target_y:target_y+bh, target_x:target_x+bw] = block

    return temp_frame

def apply_color_channel_shift(frame, max_shift=10):
    """Shifts R, G, B channels independently."""
    temp_frame = frame.copy()
    for i in range(3): # Iterate through B, G, R channels
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        channel = frame[:,:,i]
        shifted_channel = np.roll(channel, shift_y, axis=0)
        shifted_channel = np.roll(shifted_channel, shift_x, axis=1)
        temp_frame[:,:,i] = shifted_channel
    return temp_frame

def apply_warp(frame):
    """Applies a simple wave warp distortion."""
    rows, cols = frame.shape[:2]
    img_output = np.zeros(frame.shape, dtype=frame.dtype)
    amplitude = random.uniform(5, 20)
    frequency = random.uniform(0.01, 0.05)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(amplitude * math.sin(2 * math.pi * i * frequency))
            offset_y = int(amplitude * math.cos(2 * math.pi * j * frequency)) # Can add y-offset too

            src_x = min(max(j + offset_x, 0), cols - 1)
            src_y = min(max(i + offset_y, 0), rows - 1) # Use y-offset here

            img_output[i, j] = frame[src_y, src_x]
    return img_output

def apply_feedback(frame, prev_frame, alpha=0.1):
    """Blends the current frame with a slightly modified previous frame."""
    if prev_frame is None:
        return frame

    # Modify previous frame slightly (e.g., slight scale/rotation/shift)
    # Simple blend for now:
    modified_prev = prev_frame # Could add cv2.warpAffine here for rotation/scaling
    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)


# --- Enhanced Frame Generation ---
def generate_frames_enhanced(theme_data):
    """Generates complex, abstract, unsettling video frames."""
    DURATION = random.randint(MIN_DURATION, MAX_DURATION)
    FPS = random.randint(MIN_FPS, MAX_FPS)
    FRAME_COUNT = DURATION * FPS

    print(f"Generating {DURATION}-second video ({FPS} FPS) with theme: {current_theme_name}...")
    print(f"Theme elements: {theme_data}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))

    prev_frame = None # For feedback loop

    # Global text elements that persist/evolve
    persistent_texts = []
    for _ in range(random.randint(1, 4)): # Fewer, more impactful persistent texts
         persistent_texts.append({
             "text": random.choice(theme_data["words"]) + random.choice(theme_data["symbols"]),
             "pos": (random.randint(50, WIDTH - 150), random.randint(50, HEIGHT - 50)),
             "font_size": random.uniform(0.5, 2.5),
             "color": tuple(random.randint(0, 255) for _ in range(3)),
             "angle": random.uniform(-45, 45),
             "alpha": random.uniform(0.5, 1.0), # Transparency
             "lifetime": random.randint(int(FPS * 0.5), int(FPS * 5)), # How long it stays
             "frame_count": 0,
             "move_speed": (random.uniform(-2, 2), random.uniform(-1, 1)) # Movement per frame
         })


    for i in range(FRAME_COUNT):
        # Start with a base color or noise
        if random.random() < 0.3:
            bg_color = random.choice(random.choice(theme_data["colors"]))
            frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
        else:
            # Start with black for noise/feedback base
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # --- Apply Layers of Effects ---
        effect_stack = [
            apply_perlin_noise,
            apply_block_shift,
            apply_color_channel_shift,
            apply_warp,
            # Add more effects here
        ]
        random.shuffle(effect_stack) # Apply in random order

        # Apply a random number of effects
        num_effects_to_apply = random.randint(1, len(effect_stack))
        for effect_func in effect_stack[:num_effects_to_apply]:
             if random.random() < 0.6: # Probability of applying each chosen effect
                 try:
                    if effect_func == apply_perlin_noise:
                        frame = effect_func(frame, alpha=random.uniform(0.1, 0.6))
                    elif effect_func == apply_block_shift:
                        frame = effect_func(frame, max_shift=random.randint(10, 60), block_size=random.randint(20, 100))
                    elif effect_func == apply_color_channel_shift:
                        frame = effect_func(frame, max_shift=random.randint(5, 25))
                    elif effect_func == apply_warp:
                         frame = effect_func(frame)
                    # Add elif for other effects
                 except Exception as e:
                     print(f"Error applying {effect_func.__name__}: {e}") # Catch errors in effects

        # Feedback loop
        if random.random() < 0.25:
             frame = apply_feedback(frame, prev_frame, alpha=random.uniform(0.05, 0.2))

        # --- Subliminal Flash ---
        if random.random() < 0.05: # Low probability
             flash_type = random.choice(["invert", "text", "symbol", "color"])
             if flash_type == "invert":
                 frame = 255 - frame
             elif flash_type == "color":
                 flash_col = random.choice(random.choice(theme_data["colors"]))
                 frame = np.full((HEIGHT, WIDTH, 3), flash_col, dtype=np.uint8)
             else: # text or symbol
                 # Draw large, centered text/symbol
                 sub_text = random.choice(theme_data["words"]) if flash_type == "text" else random.choice(theme_data["symbols"])
                 font_scale = random.uniform(3.0, 6.0)
                 text_size, _ = cv2.getTextSize(sub_text, DEFAULT_FONT, font_scale, 3)
                 text_x = (WIDTH - text_size[0]) // 2
                 text_y = (HEIGHT + text_size[1]) // 2
                 flash_col = tuple(random.randint(0, 255) for _ in range(3))
                 cv2.putText(frame, sub_text, (text_x, text_y), DEFAULT_FONT, font_scale, flash_col, random.randint(3, 6), cv2.LINE_AA)
             # Optionally overlay with heavy noise for just this frame
             noise_overlay = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
             frame = cv2.addWeighted(frame, 0.5, noise_overlay, 0.5, 0)


        # --- Persistent and Fleeting Text ---
        current_persistent_texts = []
        for text_info in persistent_texts:
            # Update position and lifetime
            px, py = text_info["pos"]
            mx, my = text_info["move_speed"]
            text_info["pos"] = (int(px + mx), int(py + my))
            # Bounce off edges roughly
            if not (0 < text_info["pos"][0] < WIDTH - 50): text_info["move_speed"] = (-mx + random.uniform(-0.5, 0.5), my + random.uniform(-0.5, 0.5))
            if not (0 < text_info["pos"][1] < HEIGHT - 20): text_info["move_speed"] = (mx + random.uniform(-0.5, 0.5), -my + random.uniform(-0.5, 0.5))
            text_info["pos"] = (np.clip(text_info["pos"][0], 0, WIDTH - 50), np.clip(text_info["pos"][1], 0, HEIGHT - 20))


            text_info["frame_count"] += 1

            if text_info["frame_count"] < text_info["lifetime"]:
                # Draw the text (potentially add rotation here if needed)
                # For simplicity, using default font here. Rotation needs warpAffine.
                cv2.putText(frame, text_info["text"], text_info["pos"], DEFAULT_FONT,
                            text_info["font_size"], text_info["color"], random.randint(1, 3), cv2.LINE_AA)
                current_persistent_texts.append(text_info) # Keep it for next frame
            else:
                 # Chance to respawn it somewhere else
                 if random.random() < 0.3:
                     text_info["pos"] = (random.randint(50, WIDTH - 150), random.randint(50, HEIGHT - 50))
                     text_info["frame_count"] = 0
                     text_info["lifetime"] = random.randint(int(FPS * 0.5), int(FPS * 5))
                     current_persistent_texts.append(text_info)


        persistent_texts = current_persistent_texts

        # Add some fleeting text too
        if random.random() > 0.6:
            fleet_text = random.choice(theme_data["words"]) + " " + random.choice(theme_data["symbols"])
            font_size = random.uniform(0.4, 1.5)
            position = (random.randint(10, WIDTH - 100), random.randint(20, HEIGHT - 20))
            text_color = tuple(random.randint(0, 255) for _ in range(3))
            thickness = random.randint(1,2)
            cv2.putText(frame, fleet_text, position, DEFAULT_FONT, font_size, text_color, thickness, cv2.LINE_AA)


        # --- Final Touches ---
        # Random brightness/contrast adjustment
        if random.random() < 0.1:
            frame = cv2.convertScaleAbs(frame, alpha=random.uniform(0.7, 1.3), beta=random.randint(-20, 20))

        # Store frame for feedback loop
        prev_frame = frame.copy()

        video.write(frame)

        # Progress indicator
        if (i + 1) % FPS == 0:
            print(f"Generated { (i + 1) // FPS } seconds of video...")


    video.release()
    print("Video generation complete.")

# --- Enhanced Audio Generation ---

def generate_audio_enhanced(theme_data):
    """Generates complex, dynamic, and unsettling stereo audio."""
    DURATION = random.randint(MIN_DURATION, MAX_DURATION) # Use same duration logic as video
    SAMPLE_RATE = 44100
    NUM_SAMPLES = SAMPLE_RATE * DURATION
    NUM_CHANNELS = 2 # Stereo

    samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32) # Use float for processing

    # --- Audio Elements based on Theme ---
    min_freq, max_freq = theme_data["audio_freq_range"]
    noise_types = theme_data["audio_noise_types"]
    features = theme_data["audio_features"]

    current_time = 0.0
    time_step = 1.0 / SAMPLE_RATE

    # --- Generators for different sound types ---

    # Basic tone generator (can be dissonant)
    def generate_tone(freq, duration_samples, vol):
        t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
        # Add harmonics or simple FM for texture
        if random.random() < 0.3: # Simple FM
             mod_freq = freq * random.uniform(0.5, 2.0)
             mod_depth = vol * random.uniform(0.5, 2.0)
             wave = vol * np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))
        else: # Add harmonic
             wave = vol * (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t) * random.random())
        # Apply envelope (quick attack, decay)
        envelope = np.exp(-np.linspace(0, 5, duration_samples))
        return wave * envelope

    # Noise generator
    def generate_noise(noise_type, duration_samples, vol):
        if noise_type == "white":
            return vol * (2 * np.random.random(duration_samples) - 1)
        elif noise_type == "pink": # Approximation
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            # Ensure enough samples for filter initialization if using scipy.lfilter
            # Simple approach: generate longer white noise and hope for the best with slicing
            wn = vol * (2 * np.random.random(duration_samples + len(a)) - 1)
            # Simulate filtering - VERY basic approximation without SciPy
            pink = wn.copy()
            for n in range(1, len(pink)):
                 pink[n] = 0.99 * pink[n-1] + wn[n] * 0.05 # Simple low-pass like characteristic
            return pink[:duration_samples] * 2.0 # Boost amplitude approximation

        elif noise_type == "brown": # Approximation (integration of white noise)
             wn = vol * (2 * np.random.random(duration_samples) - 1)
             return np.cumsum(wn) / (SAMPLE_RATE / 100) # Normalize roughly
        elif noise_type == "glitch" or "clicks" in features:
            noise = np.zeros(duration_samples)
            num_clicks = random.randint(5, 50)
            for _ in range(num_clicks):
                pos = random.randint(0, duration_samples - 2)
                noise[pos] = vol * random.uniform(-1, 1) * 2 # Sharp click
            return noise
        elif noise_type == "static":
             wn = vol * (2 * np.random.random(duration_samples) - 1)
             # Add some crackle - modulate amplitude randomly
             crackle = 1.0 + 0.5 * (2* np.random.random(duration_samples) - 1)
             return wn * crackle
        # Add more specific noises here if needed (heartbeat, squelch simulation etc.)
        else: # Default to white
            return vol * (2 * np.random.random(duration_samples) - 1)


    # --- Generate Audio Event by Event ---
    last_event_end = 0
    while last_event_end < NUM_SAMPLES:
        event_start = last_event_end + random.randint(0, int(SAMPLE_RATE * 0.1)) # Small gap sometimes
        event_start = min(event_start, NUM_SAMPLES -1)

        event_type = random.choice(["tone", "noise", "silence"] + (["complex"] if "dissonance" in features else []))
        event_duration_samples = random.randint(int(SAMPLE_RATE * 0.05), int(SAMPLE_RATE * random.uniform(0.5, 3.0))) # Variable duration
        event_end = min(event_start + event_duration_samples, NUM_SAMPLES)
        actual_duration = event_end - event_start

        if actual_duration <= 0: break

        volume = random.uniform(0.1, 0.8) # Overall volume range (float)

        channel_data = np.zeros((actual_duration, NUM_CHANNELS), dtype=np.float32)

        if event_type == "silence":
            pass # Already zeros

        elif event_type == "tone":
            freq = random.uniform(min_freq, max_freq)
            if "extreme_pitch" in features and random.random() < 0.2:
                 freq = random.choice([random.uniform(10, 50), random.uniform(8000, 16000)]) # Very low or high
            tone_data = generate_tone(freq, actual_duration, volume)
            channel_data[:, 0] += tone_data # Add to left channel initially
            channel_data[:, 1] += tone_data # Add to right channel initially

        elif event_type == "noise":
            chosen_noise = random.choice(noise_types)
            noise_data = generate_noise(chosen_noise, actual_duration, volume)
            channel_data[:, 0] += noise_data
            channel_data[:, 1] += noise_data

        elif event_type == "complex" and "dissonance" in features:
             # Layer two dissonant tones
             freq1 = random.uniform(min_freq, max_freq / 2)
             # Create dissonance (e.g., minor second or tritone related)
             freq2 = freq1 * random.choice([1.059, 1.414, 0.943]) # Approx semitone, tritone ratios
             tone1 = generate_tone(freq1, actual_duration, volume * 0.6)
             tone2 = generate_tone(freq2, actual_duration, volume * 0.6)
             channel_data[:, 0] += tone1
             channel_data[:, 1] += tone2 # Put second tone maybe more in right?

        # --- Apply Features to the Event ---
        # Panning
        if "panning" in features and random.random() < 0.7:
            pan_pos = random.uniform(-1.0, 1.0) # -1 = Full Left, 0 = Center, 1 = Full Right
            gain_L = math.sqrt(0.5 * (1.0 - pan_pos))
            gain_R = math.sqrt(0.5 * (1.0 + pan_pos))
            event_channel_data_copy = channel_data.copy() # Avoid modifying original if needed elsewhere
            channel_data[:, 0] = event_channel_data_copy[:, 0] * gain_L + event_channel_data_copy[:, 1] * gain_L # Mix both sources based on pan
            channel_data[:, 1] = event_channel_data_copy[:, 0] * gain_R + event_channel_data_copy[:, 1] * gain_R
        else: # Default slightly random pan if not explicitly panned
             pan_pos = random.uniform(-0.2, 0.2)
             gain_L = math.sqrt(0.5 * (1.0 - pan_pos))
             gain_R = math.sqrt(0.5 * (1.0 + pan_pos))
             event_channel_data_copy = channel_data.copy()
             channel_data[:, 0] = event_channel_data_copy[:, 0] * gain_L
             channel_data[:, 1] = event_channel_data_copy[:, 1] * gain_R


        # Stutter / Granular sim
        if "stutter" in features and random.random() < 0.2:
             chunk_size = random.randint(int(SAMPLE_RATE * 0.01), int(SAMPLE_RATE * 0.05))
             repeats = random.randint(2, 5)
             if actual_duration > chunk_size * repeats:
                 start_chunk = random.randint(0, actual_duration - (chunk_size * repeats) -1)
                 chunk = channel_data[start_chunk:start_chunk+chunk_size, :].copy()
                 for r in range(repeats):
                      st = start_chunk + r * chunk_size
                      en = st + chunk_size
                      if en <= actual_duration:
                          channel_data[st:en, :] = chunk

        # Pitch Bend Simulation (simple linear slide)
        # This needs more sophisticated generation, integrated into tone/noise ideally.
        # Skipping for now to avoid overcomplication without a proper synth structure.

        # Add the processed event data to the main buffer
        samples[event_start:event_end, :] += channel_data

        last_event_end = event_end


    # --- Final Output Processing ---
    # Normalize audio to prevent clipping
    max_abs_val = np.max(np.abs(samples))
    if max_abs_val > 0:
        samples /= max_abs_val # Normalize to [-1, 1] range
        samples *= 0.8 # Reduce amplitude slightly to avoid clipping after conversion

    # Convert to 16-bit integer for WAV file
    samples_int16 = (samples * 32767).astype(np.int16)

    # --- Save Audio ---
    print("Generating WAV file...")
    with wave.open("audio_temp.wav", "w") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(2) # 16 bits = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples_int16.tobytes())

    print("Audio generation complete.")


# --- Combine Video and Audio ---
def combine_video_audio():
    """Combines video and audio using ffmpeg."""
    print("Combining video and audio...")
    # -strict experimental or -strict -2 might be needed for aac codec depending on ffmpeg version
    command = f'ffmpeg -y -i video_temp.mp4 -i audio_temp.wav -c:v copy -c:a aac -shortest -strict -2 "{OUTPUT_FILE}"'
    try:
        os.system(command)
        print(f"Final video saved as {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error during ffmpeg combination: {e}")
        print("Please ensure ffmpeg is installed and in your system's PATH.")

    # Clean up temporary files
    try:
        if os.path.exists("video_temp.mp4"):
            os.remove("video_temp.mp4")
        if os.path.exists("audio_temp.wav"):
            os.remove("audio_temp.wav")
    except OSError as e:
         print(f"Error removing temp files: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Select a random theme
    current_theme_name = random.choice(list(THEMES.keys()))
    current_theme_data = THEMES[current_theme_name]

    generate_frames_enhanced(current_theme_data)
    generate_audio_enhanced(current_theme_data)
    combine_video_audio()
