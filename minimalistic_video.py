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

# --- Configuration ---
WIDTH, HEIGHT = 640, 480
MIN_DURATION, MAX_DURATION = 20, 45 # Even wider duration range
MIN_FPS, MAX_FPS = 2, 8 # EXTREMELY low FPS range for intense jerkiness
OUTPUT_FILE = "extreme_video.mp4"
HIGH_FREQ_CUTOFF = 20000 # Target high frequency limit (adjust based on hearing/equipment)
LOW_FREQ_CUTOFF = 25     # Target low frequency limit

# --- Thematic Elements ---
# Added more extreme symbols, audio types, and features
THEMES = {
    "Digital Decay EXTREME": {
        "colors": [
            [(0, 0, 0), (0, 255, 0), (255, 0, 0)], # Glitch Red/Green/Black
            [(255, 255, 255), (0, 0, 0), (100, 100, 100)], # Stark Contrast
            [(0, 0, 50), (0, 0, 255), (255, 255, 0)], # Corrupted Blue/Yellow
        ],
        # Added encoded/glitched words
        "words": ["ERROR", "CORRUPT", "FRAGMENT", "NULL", "VOID", "DELETE", "SEGFAULT", "BUFFER", "OVERFLOW", "STATIC", "LOST", "SIGNAL", "404", "0xDEADBEEF", "Ŗ̸͈̪E̴̮͊S̷̩͂E̸̡ͮT̷͓ͨ", "U̸͖͒N̴̩ͥR̷͉ͮȆ̷̺Ȃ̶͜D̶̩̎A̸͚ͣB̷̻ͥL̴̯͘E̸̹͐", base64.b64encode(b"FEAR").decode(), binascii.hexlify(b"PAIN").decode()],
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
        "words": ["GROW", "DECAY", "CONSUME", "INFECT", "MUTATE", "FLESH", "BONE", "ROT", "SPORE", "INSIDE", "BREATHE", "PULSATE", "WRITHE", "SWELL", "BURST", "T̸̜̔H̸̻̋R̶̻͐O̵̺͒Ḇ̴͊", "內部", "곪다"], # Internal(Chinese), Fester(Korean)
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
        "words": ["BEYOND", "VOID", "STARS", "MADNESS", "UNKNOWN", "ELDRITCH", "ENTITY", "SILENCE", "WATCHING", "BELOW", "ABYSS", "UNFOLD", "SEE", "NOTHING", "Ȉ̸̮Ț̴͘_̸̤͋I̶̻͋S̴̩̕_̴͕̽H̷̙̿E̸̻̅Ŕ̴͔E̵̥͌", "無", "深淵"], # Nothingness(Chinese), Abyss(Japanese)
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
        "words": ["REMEMBER", "FORGET", "DREAM", "LOOP", "ECHO", "MIRROR", "SHADOW", "WHO", "WHY", "LOST", "AGAIN", "TRAPPED", "FACE", "NEVER", "W̴̼͗Ḫ̴̏Á̸̤T̷̮̚_̸͎̂W̸̫͐Ȁ̷̙S̶̻̐", "잊다", "什么"], # Forget(Korean), What(Chinese)
        "symbols": ["?", "¿", " திரும்பத்திரும்ப", "繰り返す", "≡", "∽", "◌", "()", "...", "≠", "||", "⚰️", "🎭", "🗝️", "", "௸", "؟"], # Added mask, key, more obscure symbols
        "audio_freq_range": (80, 14000),
        "audio_noise_types": ["whispers", "filtered_noise", "reversed_sounds", "tape_hiss", "glitch", "digital_artifact"],
        "audio_features": ["delay", "reverb", "fading_melody", "extreme_panning", "stutter", "bitcrush", "distortion"]
    }
}


# --- Font Configuration (Optional) ---
# Make sure you have fonts that support a wide range of characters, including those in the themes.
# Noto Fonts (by Google) are often good for broad Unicode support.
FONT_PATHS = [
    # Linux/MacOS common paths (adjust if needed)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Arial.ttf",
     # Windows common paths (adjust if needed)
    "C:/Windows/Fonts/arial.ttf",
    # Add paths to specific unsettling or broad Unicode fonts if you have them
    "NotoSans-Regular.ttf", # Example: Download Noto Sans
    "WEIRD_FONT.ttf",
]
AVAILABLE_FONTS = [f for f in FONT_PATHS if os.path.exists(f)]
DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX # Fallback if no TTF fonts work
SELECTED_TTF_FONT = random.choice(AVAILABLE_FONTS) if AVAILABLE_FONTS else None
print(f"Available TTF fonts found: {AVAILABLE_FONTS}")
print(f"Selected TTF font for some elements: {SELECTED_TTF_FONT}")


# Initialize pygame mixer
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024) # Smaller buffer?
except pygame.error as e:
    print(f"Pygame mixer init failed (might be normal in some envs): {e}")

# --- EXTREME Glitch Functions ---

def apply_perlin_noise(frame, alpha_range=(0.1, 0.8), scale_range=(3.0, 60.0), oct_range=(2, 8)):
    """Applies more variable Perlin noise overlay."""
    alpha = random.uniform(*alpha_range)
    scale = random.uniform(*scale_range)
    octaves = random.randint(*oct_range)
    persistence = random.uniform(0.3, 0.8) # Increased max persistence
    lacunarity = random.uniform(1.5, 3.0) # Increased max lacunarity
    seed = random.randint(0, 1000) # Wider seed range

    height, width = frame.shape[:2]
    gray_noise = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            gray_noise[i][j] = noise.pnoise2(i/scale, j/scale,
                                             octaves=octaves,
                                             persistence=persistence,
                                             lacunarity=lacunarity,
                                             base=seed)

    colored_noise = cv2.normalize(gray_noise, None, 0, 255, cv2.NORM_MINMAX)
    colored_noise = cv2.cvtColor(colored_noise.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if random.random() < 0.3: # Higher chance of inversion
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
        if height - bh -1 <= 0 or width - bw -1 <=0: continue
        y = random.randint(0, height - bh - 1)
        x = random.randint(0, width - bw - 1)

        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)

        # Ensure target indices are valid before slicing
        target_y_start = np.clip(y + shift_y, 0, height - bh)
        target_x_start = np.clip(x + shift_x, 0, width - bw)

        block = frame[y:y+bh, x:x+bw].copy()
        # Place the block - potential for overlap/overwriting is part of the effect
        temp_frame[target_y_start:target_y_start+bh, target_x_start:target_x_start+bw] = block


    return temp_frame

def apply_color_channel_shift(frame, max_shift_range=(10, 40)):
    """Shifts R, G, B channels more independently."""
    max_shift = random.randint(*max_shift_range)
    temp_frame = frame.copy()
    for i in range(3): # Iterate through B, G, R channels
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        channel = frame[:,:,i]
        shifted_channel = np.roll(channel, shift_y, axis=0)
        shifted_channel = np.roll(shifted_channel, shift_x, axis=1)
        temp_frame[:,:,i] = shifted_channel
    return temp_frame

def apply_warp(frame, amplitude_range=(10, 40), freq_range=(0.005, 0.08)):
    """Applies a more intense wave warp distortion."""
    rows, cols = frame.shape[:2]
    img_output = np.zeros(frame.shape, dtype=frame.dtype)
    amplitude_x = random.uniform(*amplitude_range)
    frequency_x = random.uniform(*freq_range)
    amplitude_y = random.uniform(*amplitude_range) * random.uniform(0.5, 1.5) # Independent Y amplitude
    frequency_y = random.uniform(*freq_range) * random.uniform(0.5, 1.5) # Independent Y frequency
    phase_x = random.uniform(0, 2 * math.pi)
    phase_y = random.uniform(0, 2 * math.pi)

    for i in range(rows):
        for j in range(cols):
            offset_x = int(amplitude_x * math.sin(2 * math.pi * i * frequency_x + phase_x))
            offset_y = int(amplitude_y * math.cos(2 * math.pi * j * frequency_y + phase_y)) # Use cos for variation

            src_x = min(max(j + offset_x, 0), cols - 1)
            src_y = min(max(i + offset_y, 0), rows - 1)

            img_output[i, j] = frame[src_y, src_x]
    return img_output

def apply_feedback(frame, prev_frame, alpha_range=(0.05, 0.3)):
    """Blends the current frame with a potentially modified previous frame."""
    if prev_frame is None:
        return frame
    alpha = random.uniform(*alpha_range)
    modified_prev = prev_frame # Could add random rotation/scaling here if desired
    # Example: slight random zoom/shift
    if random.random() < 0.1:
         rows, cols = modified_prev.shape[:2]
         M = cv2.getRotationMatrix2D((cols/2,rows/2), random.uniform(-2,2), random.uniform(0.98, 1.02))
         modified_prev = cv2.warpAffine(modified_prev, M, (cols, rows))

    return cv2.addWeighted(frame, 1 - alpha, modified_prev, alpha, 0)

def apply_pixelation(frame, block_size_range=(8, 48)):
    """Applies a pixelation effect."""
    height, width = frame.shape[:2]
    block_size = random.randint(*block_size_range)
    # Ensure block size is not zero
    block_size = max(2, block_size)
    
    # Resize down to pixelated size
    temp = cv2.resize(frame, (width // block_size, height // block_size), interpolation=cv2.INTER_NEAREST)
    # Resize back to original size
    pixelated_frame = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_frame

def apply_scanlines(frame, intensity_range=(0.5, 0.9), thickness_range=(1, 3), color_variation=30):
    """Applies more prominent scanlines."""
    intensity = random.uniform(*intensity_range)
    thickness = random.randint(*thickness_range)
    temp_frame = frame.copy()
    height, width = frame.shape[:2]
    base_color = random.randint(0, 50) # Dark lines

    for y in range(0, height, thickness * 2): # Skip lines
         line_color_val = base_color + random.randint(-color_variation, color_variation)
         line_color = (np.clip(line_color_val, 0, 255),)*3
         cv2.line(temp_frame, (0, y), (width, y), line_color, thickness)

    return cv2.addWeighted(frame, 1.0, temp_frame, intensity, -int(255 * intensity * 0.5)) # Blend darker

def apply_solarize(frame, threshold_range=(80, 180)):
    """Inverts pixels above a random threshold."""
    threshold = random.randint(*threshold_range)
    ret, mask = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
    solarized_frame = frame.copy()
    solarized_frame[mask == 255] = 255 - solarized_frame[mask == 255]
    return solarized_frame

def apply_extreme_contrast(frame, alpha_range=(1.5, 3.5), beta_range=(-60, 60)):
    """Applies very high contrast adjustments."""
    alpha = random.uniform(*alpha_range) # Contrast control
    beta = random.randint(*beta_range)    # Brightness control
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


# --- EXTREME Frame Generation ---
def generate_frames_enhanced(theme_data):
    DURATION = random.randint(MIN_DURATION, MAX_DURATION)
    FPS = random.randint(MIN_FPS, MAX_FPS)
    FRAME_COUNT = DURATION * FPS

    print(f"Generating EXTREME {DURATION}-second video ({FPS} FPS) with theme: {current_theme_name}...")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("video_temp.mp4", fourcc, FPS, (WIDTH, HEIGHT))

    prev_frame = None
    frame_hold_counter = 0
    held_frame = None

    # Persistent texts (can be more numerous and erratic)
    persistent_texts = []
    for _ in range(random.randint(3, 8)): # More persistent elements
        word = random.choice(theme_data["words"])
        symbol = random.choice(theme_data["symbols"])
        # Randomly encode some words
        if random.random() < 0.15: word = base64.b64encode(word.encode()).decode()
        elif random.random() < 0.10: word = binascii.hexlify(word.encode()).decode()

        persistent_texts.append({
            "text": word + symbol,
            "pos": (random.randint(10, WIDTH - 100), random.randint(20, HEIGHT - 20)),
            "font_size": random.uniform(0.4, 3.0), # Wider size range
            "color": tuple(random.randint(0, 255) for _ in range(3)),
            "angle": random.uniform(-90, 90), # Wider angle
            "alpha": random.uniform(0.4, 1.0),
            "lifetime": random.randint(int(FPS * 0.2), int(FPS * 6)), # More variance
            "frame_count": 0,
            "move_speed": (random.uniform(-4, 4), random.uniform(-3, 3)) # Faster movement
        })

    for i in range(FRAME_COUNT):

        # Frame stutter/hold simulation
        if frame_hold_counter > 0:
            if held_frame is not None:
                video.write(held_frame)
                frame_hold_counter -= 1
                # Progress indicator needs frame index i
                if (i + 1) % (FPS * 5) == 0: # Update less often
                    print(f"Generated { (i + 1) / FPS :.1f} seconds of video...")
                continue # Skip generating a new frame
            else: # Reset if held_frame is somehow None
                frame_hold_counter = 0


        if random.random() < 0.02: # Chance to hold a frame
            frame_hold_counter = random.randint(1, int(FPS * 0.5)) # Hold for up to half a second


        # Base frame
        if random.random() < 0.4:
            bg_color = random.choice(random.choice(theme_data["colors"]))
            frame = np.full((HEIGHT, WIDTH, 3), bg_color, dtype=np.uint8)
        else:
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # --- Apply Layers of EXTREME Effects ---
        effect_stack = [
            (apply_perlin_noise, 0.7),
            (apply_block_shift, 0.6),
            (apply_color_channel_shift, 0.6),
            (apply_warp, 0.5),
            (apply_pixelation, 0.4),
            (apply_scanlines, 0.3),
            (apply_solarize, 0.2),
            (apply_extreme_contrast, 0.3),
            # Add more effects here with their probability
        ]
        random.shuffle(effect_stack)

        num_effects_to_apply = random.randint(2, len(effect_stack)) # Apply more effects generally

        for effect_func, probability in effect_stack[:num_effects_to_apply]:
            if random.random() < probability: # Use individual probabilities
                try:
                    # Call functions without specific args here if they fetch ranges internally
                    if effect_func == apply_feedback: # Feedback needs prev_frame
                        frame = apply_feedback(frame, prev_frame, alpha_range=(0.1, 0.4)) # More intense feedback
                    else:
                        frame = effect_func(frame)
                except Exception as e:
                    print(f"Error applying {effect_func.__name__}: {e}")


        # --- Subliminal Flash --- MORE LIKELY
        if random.random() < 0.12: # Increased probability
            flash_type = random.choice(["invert", "text", "symbol", "color", "noise", "contrast"])
            overlay = frame.copy() # Work on a copy for the flash
            if flash_type == "invert":
                overlay = 255 - frame
            elif flash_type == "color":
                flash_col = random.choice(random.choice(theme_data["colors"]))
                overlay = np.full((HEIGHT, WIDTH, 3), flash_col, dtype=np.uint8)
            elif flash_type == "noise":
                noise_overlay = np.random.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
                overlay = cv2.addWeighted(frame, 0.3, noise_overlay, 0.7, 0)
            elif flash_type == "contrast":
                 overlay = apply_extreme_contrast(frame, alpha_range=(3.0, 6.0), beta_range=(-80, 80))
            else: # text or symbol
                sub_text = random.choice(theme_data["words"]) if flash_type == "text" else random.choice(theme_data["symbols"])
                font_scale = random.uniform(4.0, 8.0) # Larger flash text
                thickness = random.randint(4, 8)
                # Try to use TTF font if available for better symbol support
                try:
                     # Using simple fallback for size calc - TTF needs different handling
                     text_size, _ = cv2.getTextSize(sub_text, DEFAULT_FONT, font_scale, thickness)
                     text_x = max(0, (WIDTH - text_size[0]) // 2)
                     text_y = max(0, (HEIGHT + text_size[1]) // 2)
                     flash_col = tuple(random.randint(0, 255) for _ in range(3))
                     # Add outline/shadow for visibility
                     cv2.putText(overlay, sub_text, (text_x+3, text_y+3), DEFAULT_FONT, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
                     cv2.putText(overlay, sub_text, (text_x, text_y), DEFAULT_FONT, font_scale, flash_col, thickness, cv2.LINE_AA)
                except Exception as e:
                     print(f"Error rendering flash text '{sub_text}': {e}") # Catch potential rendering errors

            # Blend the flash frame briefly
            frame = cv2.addWeighted(frame, 0.2, overlay, 0.8, 0)


        # --- Persistent and Fleeting Text --- (More chaotic)
        current_persistent_texts = []
        frame_overlay = frame.copy() # Draw text on an overlay to manage alpha
        for text_info in persistent_texts:
            # Update position and lifetime
            px, py = text_info["pos"]
            mx, my = text_info["move_speed"]
            text_info["pos"] = (int(px + mx), int(py + my))

            # More erratic bounce/wrap
            if not (0 < text_info["pos"][0] < WIDTH - 50):
                text_info["move_speed"] = (-mx * random.uniform(0.8, 1.2), my + random.uniform(-1, 1))
                if random.random() < 0.1: text_info["pos"] = (random.randint(10, WIDTH-100), text_info["pos"][1]) # Random reposition
            if not (0 < text_info["pos"][1] < HEIGHT - 20):
                text_info["move_speed"] = (mx + random.uniform(-1, 1), -my * random.uniform(0.8, 1.2))
                if random.random() < 0.1: text_info["pos"] = (text_info["pos"][0], random.randint(20, HEIGHT-20))

            text_info["pos"] = (np.clip(text_info["pos"][0], 0, WIDTH - 50), np.clip(text_info["pos"][1], 0, HEIGHT - 20))
            text_info["frame_count"] += 1

            if text_info["frame_count"] < text_info["lifetime"]:
                try:
                    # Attempt drawing with rotation (more complex)
                    # For simplicity, stick to non-rotated or use TTF handling if implemented
                    font_face = DEFAULT_FONT # Could try SELECTED_TTF_FONT but needs font loading/handling
                    cv2.putText(frame_overlay, text_info["text"], text_info["pos"], font_face,
                                text_info["font_size"], text_info["color"], random.randint(1, 4), cv2.LINE_AA)
                    current_persistent_texts.append(text_info)
                except Exception as e:
                    #print(f"Error drawing persistent text: {e}") # Avoid spamming logs
                    current_persistent_texts.append(text_info) # Keep it anyway
            else:
                if random.random() < 0.5: # Higher chance to respawn
                    text_info["pos"] = (random.randint(10, WIDTH - 100), random.randint(20, HEIGHT - 20))
                    text_info["frame_count"] = 0
                    text_info["lifetime"] = random.randint(int(FPS * 0.2), int(FPS * 6))
                    text_info["text"] = random.choice(theme_data["words"]) + random.choice(theme_data["symbols"]) # Change text on respawn
                    current_persistent_texts.append(text_info)

        persistent_texts = current_persistent_texts

        # Add more fleeting text
        if random.random() > 0.3: # More likely
             num_fleet = random.randint(1, 4) # Multiple fleeting texts
             for _ in range(num_fleet):
                fleet_text = random.choice(theme_data["words"]) + " " + random.choice(theme_data["symbols"])
                if random.random() < 0.1: fleet_text = base64.b64encode(fleet_text.encode()).decode() # Encode some fleeting text
                font_size = random.uniform(0.3, 1.8)
                position = (random.randint(5, WIDTH - 80), random.randint(10, HEIGHT - 10))
                text_color = tuple(random.randint(0, 255) for _ in range(3))
                thickness = random.randint(1, 3)
                try:
                     cv2.putText(frame_overlay, fleet_text, position, DEFAULT_FONT, font_size, text_color, thickness, cv2.LINE_AA)
                except Exception as e:
                     pass # Ignore errors for fleeting text

        # Blend text overlay with alpha (simple fixed alpha for now)
        frame = cv2.addWeighted(frame, 0.7, frame_overlay, 0.9, 0) # Adjust weights for visibility


        # Store frame for feedback loop BEFORE potential hold
        prev_frame = frame.copy()

        # If this frame is designated to be held, store it
        if frame_hold_counter > 0 and held_frame is None:
             held_frame = frame.copy()


        video.write(frame)

        # Progress indicator
        if (i + 1) % (FPS * 5) == 0: # Update less often
            print(f"Generated { (i + 1) / FPS :.1f} seconds of video...")


    video.release()
    print("Video generation complete.")
    return DURATION # Return duration for audio sync

# --- EXTREME Audio Generation ---

def apply_distortion(data, intensity_range=(1.5, 5.0)):
    """Applies clipping distortion."""
    intensity = random.uniform(*intensity_range)
    # Simple hard clipping
    return np.clip(data * intensity, -0.95, 0.95) # Clip slightly below 1 to leave headroom maybe?

def apply_bitcrush(data, bit_depth_range=(4, 12)):
    """Simulates bitcrushing."""
    bits = random.randint(*bit_depth_range)
    if bits >= 16: return data # No effect if high bit depth
    max_val = 2**(bits -1) # Signed integer range
    # Quantize: scale, round, descale
    quantized = np.round(data * max_val) / max_val
    return quantized

# Enhanced tone/noise generators for extremes
def generate_tone(freq, duration_samples, vol, allow_fm=True):
    t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
    # More aggressive FM/harmonics
    if allow_fm and random.random() < 0.5: # More FM
        mod_freq = freq * random.uniform(0.1, 5.0) # Wilder FM freq ratio
        mod_depth = vol * random.uniform(1.0, 8.0) # Wilder FM depth
        wave = vol * np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))
    else: # More harmonics
        harmonic_count = random.randint(1, 4)
        wave = np.sin(2 * np.pi * freq * t)
        for h in range(2, harmonic_count + 2):
            wave += random.uniform(0.1, 0.5) * np.sin(2 * np.pi * freq * h * t + random.uniform(0, np.pi)) # Add phase shift
        wave = wave / (1 + harmonic_count * 0.3) # Rough normalization
        wave *= vol

    # Sharper envelope
    attack_len = min(int(SAMPLE_RATE * 0.005), duration_samples) # Very short attack
    decay_len = min(int(SAMPLE_RATE * random.uniform(0.1, 0.8)), duration_samples - attack_len)
    sustain_level = random.uniform(0.1, 0.7)

    envelope = np.ones(duration_samples)
    if attack_len > 0:
        envelope[:attack_len] = np.linspace(0, 1, attack_len)
    if decay_len > 0:
         envelope[attack_len:attack_len+decay_len] = np.linspace(1, sustain_level, decay_len)
         envelope[attack_len+decay_len:] = sustain_level * np.exp(-np.linspace(0, 5, duration_samples - attack_len - decay_len)) # Exponential tail
    elif duration_samples > attack_len: # Handle short sounds with no decay phase
         envelope[attack_len:] = np.exp(-np.linspace(0, 5, duration_samples - attack_len))


    return wave * envelope


def generate_noise(noise_type, duration_samples, vol):
    if noise_type == "white":
        return vol * (2 * np.random.random(duration_samples) - 1)
    elif noise_type == "pink":
        # Simple approx (more realistic needs filtering)
        wn = 2 * np.random.random(duration_samples) - 1
        pink = np.cumsum(wn) # Integrate white
        # High-pass filter to remove DC offset and excessive low end
        alpha = 0.99
        pink_filtered = np.zeros_like(pink)
        pink_filtered[0] = pink[0]
        for n in range(1, duration_samples):
            pink_filtered[n] = alpha * pink_filtered[n-1] + pink[n] - pink[n-1]
        pink_norm = pink_filtered / (np.max(np.abs(pink_filtered)) + 1e-6) # Normalize
        return vol * pink_norm

    elif noise_type == "brown":
         wn = vol * (2 * np.random.random(duration_samples) - 1)
         brown = np.cumsum(wn)
         return (brown / (np.max(np.abs(brown)) + 1e-6)) * vol # Normalize roughly

    elif noise_type == "glitch" or noise_type == "clicks" or noise_type == "wet_clicks":
        noise = np.zeros(duration_samples)
        num_clicks = random.randint(10, 150) # More clicks
        click_type = random.choice(["sharp", "burst", "resonant"])
        for _ in range(num_clicks):
            pos = random.randint(0, duration_samples - 50) # Ensure space for burst/resonance
            amp = vol * random.uniform(0.5, 1.5) # Louder clicks possible
            if click_type == "sharp":
                 noise[pos] = amp * random.choice([-1, 1])
                 noise[pos+1] = -noise[pos] * 0.5 # Simple decay
            elif click_type == "burst":
                 burst_len = random.randint(2, 10)
                 noise[pos:pos+burst_len] = amp * (2 * np.random.random(burst_len) - 1)
            elif click_type == "resonant" and duration_samples > pos + 40: # Resonant click (short decaying sine)
                 click_freq = random.uniform(500, 8000)
                 click_env = np.exp(-np.linspace(0, 15, 40)) # Fast decay
                 click_wave = amp * np.sin(2*np.pi*click_freq*np.arange(40)/SAMPLE_RATE) * click_env
                 noise[pos:pos+40] += click_wave

        return noise
    elif noise_type == "static":
        wn = vol * (2 * np.random.random(duration_samples) - 1)
        crackle = 1.0 + random.uniform(0.5, 2.5) * (2* np.random.random(duration_samples) - 1) # More intense crackle
        # Add some filtering for flavor
        alpha = random.uniform(0.1, 0.9)
        filtered_static = np.zeros_like(wn)
        for n in range(1,duration_samples):
            filtered_static[n] = alpha * filtered_static[n-1] + (1-alpha) * wn[n]
        return filtered_static * crackle * vol

    elif noise_type == "digital_artifact": # Simulate harsh digital noise
        noise = vol * (2 * np.random.random(duration_samples) - 1)
        # Randomly zero out sections or repeat small chunks
        if random.random() < 0.5: # Zeroing
            num_zeros = random.randint(5, 20)
            for _ in range(num_zeros):
                z_start = random.randint(0, duration_samples - 100)
                z_len = random.randint(10, 100)
                noise[z_start:z_start+z_len] = 0
        else: # Repeating chunks
            num_repeats = random.randint(3, 10)
            chunk_len = random.randint(5, 50)
            if duration_samples > chunk_len * num_repeats + 10:
                 r_start = random.randint(0, duration_samples - (chunk_len * num_repeats) - 5)
                 chunk = noise[r_start:r_start+chunk_len].copy()
                 for r_idx in range(num_repeats):
                     noise[r_start+(r_idx*chunk_len) : r_start+((r_idx+1)*chunk_len)] = chunk
        # Add some bitcrushing effect
        noise = apply_bitcrush(noise, (3, 8))
        return noise

    elif noise_type == "screech": # High-frequency chaos
        freq = random.uniform(4000, HIGH_FREQ_CUTOFF - 1000)
        mod_freq = random.uniform(50, 500) # Fast modulation
        mod_depth = vol * random.uniform(5, 20) # Deep modulation
        t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
        wave = vol * np.sin(2 * np.pi * freq * t + mod_depth * np.sin(2 * np.pi * mod_freq * t))
        # Add noise component
        wave += vol * 0.3 * (2*np.random.random(duration_samples)-1)
        return wave

    elif noise_type == "sub_bass": # Very low frequencies
         freq = random.uniform(LOW_FREQ_CUTOFF, 60)
         # Use square or sawtooth for more harmonics/rumble
         t = np.linspace(0, duration_samples * time_step, duration_samples, endpoint=False)
         if random.random() < 0.5: # Square wave approx
             wave = vol * np.sign(np.sin(2 * np.pi * freq * t))
         else: # Sawtooth wave approx
             wave = vol * (2 * (t * freq - np.floor(0.5 + t * freq)))
         # Add slight LFO modulation for throb if requested
         if "sub_bass_throb" in features:
              lfo_freq = random.uniform(0.1, 2)
              wave *= (0.7 + 0.3 * np.sin(2*np.pi* lfo_freq * t))
         return wave

    else: # Default to white
        return vol * (2 * np.random.random(duration_samples) - 1)


def generate_audio_enhanced(video_duration, theme_data):
    """Generates EXTREME, dynamic, and unsettling stereo audio."""
    DURATION = video_duration # Sync with video
    global SAMPLE_RATE, time_step # Make global for helpers
    SAMPLE_RATE = 44100
    time_step = 1.0 / SAMPLE_RATE
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
    NUM_CHANNELS = 2 # Stereo

    samples = np.zeros((NUM_SAMPLES, NUM_CHANNELS), dtype=np.float32)

    # --- Audio Elements based on Theme ---
    min_freq, max_freq = theme_data["audio_freq_range"]
    global noise_types, features # Make global for generate_noise
    noise_types = theme_data["audio_noise_types"]
    features = theme_data["audio_features"]


    # --- Generate Audio Event by Event --- More Chaotically
    last_event_end = 0
    while last_event_end < NUM_SAMPLES:
        # Overlapping is possible and encouraged
        event_start = last_event_end - random.randint(0, int(SAMPLE_RATE * 0.05)) # Allow overlap
        event_start = max(0, event_start)

        # Wilder duration range, allow very short bursts
        event_duration_samples = random.randint(int(SAMPLE_RATE * 0.01), int(SAMPLE_RATE * random.uniform(0.3, 5.0)))
        event_end = min(event_start + event_duration_samples, NUM_SAMPLES)
        actual_duration = event_end - event_start

        if actual_duration <= 5: # Need at least a few samples
            last_event_end = event_end # Move marker even if skipping tiny event
            continue


        event_type = random.choice(["tone", "noise", "silence"] * 2 + ["complex", "extreme_tone"] + noise_types) # More chances for noise/extremes
        volume = random.uniform(0.05, 1.0) # Wider volume range (will be normalized later)

        channel_data = np.zeros((actual_duration, NUM_CHANNELS), dtype=np.float32)

        # --- Generate Base Sound ---
        if event_type == "silence":
            pass # Zeros
        elif event_type == "tone":
            freq = random.uniform(min_freq, max_freq)
            tone_data = generate_tone(freq, actual_duration, volume)
            channel_data[:, 0] += tone_data # Add to left/right equally before panning
            channel_data[:, 1] += tone_data
        elif event_type == "extreme_tone" and "extreme_pitch" in features:
             # Force very high or very low
             if random.random() < 0.5:
                 freq = random.uniform(LOW_FREQ_CUTOFF, min_freq * 0.8) # Ultra low
             else:
                 freq = random.uniform(max_freq * 1.2, HIGH_FREQ_CUTOFF) # Ultra high
             freq = np.clip(freq, LOW_FREQ_CUTOFF, HIGH_FREQ_CUTOFF) # Ensure within bounds
             tone_data = generate_tone(freq, actual_duration, volume, allow_fm=False) # Often cleaner without FM for extremes
             channel_data[:, 0] += tone_data
             channel_data[:, 1] += tone_data

        elif event_type == "noise" or event_type in noise_types: # Handle specific noise types
            chosen_noise = event_type if event_type in noise_types else random.choice(noise_types)
            noise_data = generate_noise(chosen_noise, actual_duration, volume)
            channel_data[:, 0] += noise_data
            channel_data[:, 1] += noise_data

        elif event_type == "complex" and "dissonance" in features:
            freq1 = random.uniform(min_freq, max_freq * 0.6)
            # More dissonant intervals (semitones, tritones, major sevenths etc.)
            freq2 = freq1 * random.choice([1.05946, 1.4983, 0.94387, 1.8877, 1.1224])
            freq2 = np.clip(freq2, min_freq, max_freq) # Keep within range
            tone1 = generate_tone(freq1, actual_duration, volume * 0.7)
            tone2 = generate_tone(freq2, actual_duration, volume * 0.7)
            # Place dissonant tones in different channels initially
            channel_data[:, 0] += tone1
            channel_data[:, 1] += tone2


        # --- Apply Features to the Event --- More Aggressively
        processed_event_data = channel_data # Start with generated data

        # Panning (Wider and More Frequent)
        if ("panning" in features or "extreme_panning" in features) and random.random() < 0.85:
            if "extreme_panning" in features:
                 pan_pos = random.choice([-1.0, 1.0, random.uniform(-0.9, 0.9)]) # Force hard L/R sometimes
                 # Add rapid LFO panning for extreme effect
                 if random.random() < 0.2:
                      pan_lfo_freq = random.uniform(2, 15) # Fast LFO
                      t_pan = np.linspace(0, actual_duration*time_step, actual_duration, endpoint=False)
                      pan_signal = 0.95 * np.sin(2*np.pi*pan_lfo_freq*t_pan) # Vary pan over time
                 else:
                      pan_signal = pan_pos # Static pan for this event
            else:
                 pan_signal = random.uniform(-0.8, 0.8)

            # Apply panning using constant power law (sqrt)
            gain_L = np.sqrt(0.5 * (1.0 - pan_signal))
            gain_R = np.sqrt(0.5 * (1.0 + pan_signal))

            # Need to handle array vs scalar gain - broadcast if needed
            if isinstance(gain_L, np.ndarray):
                 gain_L = gain_L[:, np.newaxis] # Add channel dimension
                 gain_R = gain_R[:, np.newaxis]

            # Mix sources based on pan (assuming mono source was duplicated)
            source_mono = (processed_event_data[:, 0] + processed_event_data[:, 1]) / 2 # Mix L/R before panning
            processed_event_data[:, 0] = source_mono * gain_L.flatten() # Apply gain (flatten if LFO)
            processed_event_data[:, 1] = source_mono * gain_R.flatten()


        # Stutter / Granular sim (More likely, smaller chunks)
        if "stutter" in features and random.random() < 0.35:
            chunk_size = random.randint(int(SAMPLE_RATE * 0.005), int(SAMPLE_RATE * 0.03)) # Smaller chunks
            repeats = random.randint(3, 8) # More repeats
            if actual_duration > chunk_size * repeats + 10:
                start_chunk = random.randint(0, actual_duration - (chunk_size * repeats) - 5)
                chunk = processed_event_data[start_chunk:start_chunk+chunk_size, :].copy()
                # Apply fades to chunk edges to reduce clicks
                fade_len = min(chunk_size // 4, 50) # Short fade
                if fade_len > 0:
                    fade_in = np.linspace(0,1,fade_len)
                    fade_out = np.linspace(1,0,fade_len)
                    chunk[:fade_len, :] *= fade_in[:, np.newaxis]
                    chunk[-fade_len:, :] *= fade_out[:, np.newaxis]

                for r in range(repeats):
                    st = start_chunk + r * chunk_size
                    en = st + chunk_size
                    if en <= actual_duration:
                        processed_event_data[st:en, :] = chunk * random.uniform(0.7, 1.0) # Vary volume slightly


        # Bitcrushing
        if "bitcrush" in features and random.random() < 0.3:
             processed_event_data = apply_bitcrush(processed_event_data, (3, 10)) # Lower bit depths

        # Distortion/Clipping
        if ("distortion" in features or "clipping_sim" in features) and random.random() < 0.4:
             processed_event_data = apply_distortion(processed_event_data, (2.0, 8.0)) # Higher intensity


        # Add the processed event data to the main buffer, allowing overlaps
        # Ensure indices are within bounds
        end_idx = min(event_start + processed_event_data.shape[0], NUM_SAMPLES)
        len_to_add = end_idx - event_start
        if len_to_add > 0 :
             samples[event_start:end_idx, :] += processed_event_data[:len_to_add, :]


        # Move the 'marker' for the next sound's potential start time
        # Make it less predictable - sometimes big gaps, sometimes immediate overlaps
        last_event_end = event_start + int(actual_duration * random.uniform(0.1, 1.2))
        last_event_end = min(last_event_end, NUM_SAMPLES)


    # --- Final Output Processing ---
    print("Normalizing and saving audio...")
    # Normalize audio aggressively to use full dynamic range
    max_abs_val = np.max(np.abs(samples))
    if max_abs_val > 0:
        # Normalize to -0.98 to 0.98 to prevent potential clipping from float->int conversion issues
        samples = samples / max_abs_val * 0.98
    else:
        print("Warning: Generated audio is silent.")


    # Convert to 16-bit integer for WAV file
    samples_int16 = (samples * 32767).astype(np.int16)

    # --- Save Audio ---
    try:
        with wave.open("audio_temp.wav", "w") as wf:
            wf.setnchannels(NUM_CHANNELS)
            wf.setsampwidth(2) # 16 bits = 2 bytes
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples_int16.tobytes())
        print("Temporary WAV audio saved.")
    except Exception as e:
        print(f"Error saving WAV file: {e}")


# --- Combine Video and Audio ---
def combine_video_audio(output_filename):
    """Combines video and audio using ffmpeg."""
    print("Combining video and audio using ffmpeg...")
    # Use -loglevel error to hide verbose ffmpeg output unless there's an error
    # -shortest ensures output duration matches the shorter input (video or audio)
    # -y overwrites output file without asking
    # -c:a aac is a common audio codec for mp4
    # -strict -2 or -strict experimental might be needed for AAC on some ffmpeg versions
    command = (f'ffmpeg -y -loglevel error -i video_temp.mp4 -i audio_temp.wav '
               f'-c:v copy -c:a aac -b:a 192k -shortest -strict experimental "{output_filename}"')
    print(f"Executing: {command}") # Show the command being run

    status = os.system(command)

    if status == 0:
        print(f"Final video saved successfully as {output_filename}")
    else:
        print("="*20 + " FFMPEG ERROR " + "="*20)
        print(f"ffmpeg command failed with status {status}.")
        print("Please ensure ffmpeg is installed and in your system's PATH.")
        print("Check if the temporary files 'video_temp.mp4' and 'audio_temp.wav' were created.")
        print("Command run was:")
        print(command)
        print("="*50)


    # Clean up temporary files
    for temp_file in ["video_temp.mp4", "audio_temp.wav"]:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                #print(f"Removed temporary file: {temp_file}")
        except OSError as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    # Select a random theme
    current_theme_name = random.choice(list(THEMES.keys()))
    current_theme_data = THEMES[current_theme_name]

    # Ensure consistency in duration by generating video first and passing duration to audio
    try:
        video_duration = generate_frames_enhanced(current_theme_data)
        if video_duration > 0:
             generate_audio_enhanced(video_duration, current_theme_data)
             combine_video_audio(OUTPUT_FILE)
        else:
             print("Error: Video generation resulted in zero duration.")

    except Exception as e:
        print("\n" + "="*20 + " UNEXPECTED ERROR " + "="*20)
        import traceback
        traceback.print_exc()
        print("="*58)
        print("An error occurred during generation. Check the traceback above.")

    finally:
         # Attempt cleanup again just in case
         for temp_file in ["video_temp.mp4", "audio_temp.wav"]:
             if os.path.exists(temp_file):
                 try: os.remove(temp_file)
                 except: pass
         if pygame.mixer.get_init():
             pygame.mixer.quit()
