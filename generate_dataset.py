"""
Traffic Sign Dataset Generator
Generates clean and augmented synthetic traffic sign images for KNN training and testing.
Uses high-quality scalable vector-style fonts and robust augmentations.
"""

import os
import random
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

CLASSES = {
    "stop": {"name": "Stop Sign", "color": "#DC2626"},
    "speed_30": {"name": "Speed Limit (30 km/h)", "color": "#F97316"},
    "speed_50": {"name": "Speed Limit (50 km/h)", "color": "#F59E0B"},
    "speed_80": {"name": "Speed Limit (80 km/h)", "color": "#EAB308"},
    "yield": {"name": "Yield / Give Way", "color": "#84CC16"},
    "no_entry": {"name": "No Entry", "color": "#EF4444"},
    "turn_right": {"name": "Turn Right Ahead", "color": "#3B82F6"},
    "turn_left": {"name": "Turn Left Ahead", "color": "#06B6D4"},
    "ahead_only": {"name": "Ahead Only", "color": "#6366F1"},
    "pedestrian": {"name": "Pedestrian Crossing", "color": "#8B5CF6"}
}

# Locate system bold fonts
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
]
VALID_FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]
if not VALID_FONTS:
    # Fallback to any system TTF font
    all_ttfs = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    VALID_FONTS = all_ttfs[:3] if all_ttfs else [None]

def draw_base_sign(cls_key, size=64, font_path=None, bg_color=(240, 240, 240)):
    if font_path is None and VALID_FONTS and VALID_FONTS[0]:
        font_path = VALID_FONTS[0]
        
    pad = int(size * 0.06)
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    if cls_key == "stop":
        w = size - 2 * pad
        c = w / 3.0
        points = [
            (pad + c, pad), (pad + 2 * c, pad),
            (size - pad, pad + c), (size - pad, pad + 2 * c),
            (pad + 2 * c, size - pad), (pad + c, size - pad),
            (pad, pad + 2 * c), (pad, pad + c)
        ]
        draw.polygon(points, fill="#DC2626", outline="#FFFFFF", width=max(2, int(size * 0.04)))
        if font_path:
            f = ImageFont.truetype(font_path, int(size * 0.24))
            draw.text((size // 2, size // 2), "STOP", fill="#FFFFFF", font=f, anchor="mm")
        else:
            draw.text((size // 2, size // 2), "STOP", fill="#FFFFFF", anchor="mm")

    elif cls_key in ["speed_30", "speed_50", "speed_80"]:
        border_w = max(4, int(size * 0.11))
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#FFFFFF", outline="#DC2626", width=border_w)
        num_str = cls_key.split("_")[1]
        if font_path:
            f = ImageFont.truetype(font_path, int(size * 0.40))
            draw.text((size // 2, size // 2 - int(size * 0.02)), num_str, fill="#000000", font=f, anchor="mm")
        else:
            draw.text((size // 2, size // 2), num_str, fill="#000000", anchor="mm")

    elif cls_key == "yield":
        points = [(pad, pad), (size - pad, pad), (size // 2, size - pad)]
        border_w = max(4, int(size * 0.12))
        draw.polygon(points, fill="#FFFFFF", outline="#DC2626", width=border_w)
        if font_path:
            f = ImageFont.truetype(font_path, int(size * 0.16))
            draw.text((size // 2, int(size * 0.38)), "YIELD", fill="#DC2626", font=f, anchor="mm")
        else:
            draw.text((size // 2, int(size * 0.38)), "YIELD", fill="#DC2626", anchor="mm")

    elif cls_key == "no_entry":
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#DC2626", outline="#FFFFFF", width=max(2, int(size * 0.03)))
        bar_h = max(4, int(size * 0.18))
        draw.rectangle([pad + int(size * 0.12), size // 2 - bar_h // 2, size - pad - int(size * 0.12), size // 2 + bar_h // 2], fill="#FFFFFF")

    elif cls_key == "turn_right":
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#2563EB", outline="#FFFFFF", width=max(2, int(size * 0.04)))
        cx, cy = size // 2, size // 2
        draw.line([(pad + int(size * 0.2), cy), (size - pad - int(size * 0.25), cy)], fill="#FFFFFF", width=max(3, int(size * 0.09)))
        arrow_head = [
            (size - pad - int(size * 0.15), cy),
            (size - pad - int(size * 0.35), cy - int(size * 0.16)),
            (size - pad - int(size * 0.35), cy + int(size * 0.16))
        ]
        draw.polygon(arrow_head, fill="#FFFFFF")

    elif cls_key == "turn_left":
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#2563EB", outline="#FFFFFF", width=max(2, int(size * 0.04)))
        cx, cy = size // 2, size // 2
        draw.line([(size - pad - int(size * 0.2), cy), (pad + int(size * 0.25), cy)], fill="#FFFFFF", width=max(3, int(size * 0.09)))
        arrow_head = [
            (pad + int(size * 0.15), cy),
            (pad + int(size * 0.35), cy - int(size * 0.16)),
            (pad + int(size * 0.35), cy + int(size * 0.16))
        ]
        draw.polygon(arrow_head, fill="#FFFFFF")

    elif cls_key == "ahead_only":
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#2563EB", outline="#FFFFFF", width=max(2, int(size * 0.04)))
        cx, cy = size // 2, size // 2
        draw.line([(cx, size - pad - int(size * 0.2)), (cx, pad + int(size * 0.25))], fill="#FFFFFF", width=max(3, int(size * 0.09)))
        arrow_head = [
            (cx, pad + int(size * 0.15)),
            (cx - int(size * 0.16), pad + int(size * 0.35)),
            (cx + int(size * 0.16), pad + int(size * 0.35))
        ]
        draw.polygon(arrow_head, fill="#FFFFFF")

    elif cls_key == "pedestrian":
        draw.rectangle([pad, pad, size - pad, size - pad], fill="#2563EB", outline="#FFFFFF", width=max(2, int(size * 0.04)))
        cx = size // 2
        draw.ellipse([cx - int(size*0.06), pad + int(size*0.12), cx + int(size*0.06), pad + int(size*0.24)], fill="#FFFFFF")
        draw.line([(cx, pad + int(size*0.26)), (cx, pad + int(size*0.52))], fill="#FFFFFF", width=max(2, int(size*0.07)))
        draw.line([(cx, pad + int(size*0.52)), (cx - int(size*0.14), size - pad - int(size*0.14))], fill="#FFFFFF", width=max(2, int(size*0.06)))
        draw.line([(cx, pad + int(size*0.52)), (cx + int(size*0.14), size - pad - int(size*0.14))], fill="#FFFFFF", width=max(2, int(size*0.06)))
        draw.line([(cx - int(size*0.14), pad + int(size*0.42)), (cx + int(size*0.14), pad + int(size*0.34))], fill="#FFFFFF", width=max(2, int(size*0.06)))

    return img

def augment_image(base_img):
    img = base_img.copy()

    # 1. Random rotation (-12 to 12 deg)
    angle = random.uniform(-12, 12)
    bg_color = (random.randint(225, 245), random.randint(225, 245), random.randint(225, 245))
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=bg_color)

    # 2. Random Brightness & Contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.85, 1.2))

    # 3. Random subtle blur
    if random.random() > 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.5)))

    # 4. Subtle sensor noise
    arr = np.array(img, dtype=float)
    noise = np.random.normal(0, 2.5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)

def generate_full_dataset(train_samples_per_class=40):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "data", "train")
    test_dir = os.path.join(base_dir, "data", "test_samples")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Generating Traffic Sign Dataset with clear typography and multi-font diversity...")
    for cls_key in CLASSES:
        cls_folder = os.path.join(train_dir, cls_key)
        os.makedirs(cls_folder, exist_ok=True)

        # 1. Clean test sample
        default_font = VALID_FONTS[0] if VALID_FONTS else None
        clean_img = draw_base_sign(cls_key, size=64, font_path=default_font)
        clean_img.save(os.path.join(test_dir, f"{cls_key}_sample.png"))

        # 2. Augmented training set
        for i in range(train_samples_per_class):
            chosen_font = random.choice(VALID_FONTS) if VALID_FONTS else None
            base = draw_base_sign(cls_key, size=64, font_path=chosen_font)
            aug = augment_image(base)
            aug.save(os.path.join(cls_folder, f"{cls_key}_{i:03d}.png"))

    total_train = len(CLASSES) * train_samples_per_class
    print(f"Dataset generated successfully! ({total_train} training images in data/train/, {len(CLASSES)} test samples in data/test_samples/)")

if __name__ == "__main__":
    generate_full_dataset()
