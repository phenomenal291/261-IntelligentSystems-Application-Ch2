"""
Traffic Sign Dataset Generator
Generates clean and augmented synthetic traffic sign images for KNN training and testing.
Includes 15 distinct international / GTSRB traffic sign categories with multi-font and background augmentations.
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
    "speed_100": {"name": "Speed Limit (100 km/h)", "color": "#D97706"},
    "yield": {"name": "Yield / Give Way", "color": "#84CC16"},
    "no_entry": {"name": "No Entry", "color": "#EF4444"},
    "no_overtaking": {"name": "No Overtaking", "color": "#B91C1C"},
    "turn_right": {"name": "Turn Right Ahead", "color": "#3B82F6"},
    "turn_left": {"name": "Turn Left Ahead", "color": "#06B6D4"},
    "ahead_only": {"name": "Ahead Only", "color": "#6366F1"},
    "roundabout": {"name": "Roundabout Mandatory", "color": "#2563EB"},
    "traffic_signals": {"name": "Traffic Light Ahead", "color": "#10B981"},
    "road_work": {"name": "Road Work Ahead", "color": "#EA580C"},
    "pedestrian": {"name": "Pedestrian Crossing", "color": "#8B5CF6"}
}

# Locate system fonts
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
]
VALID_FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]
if not VALID_FONTS:
    all_ttfs = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    VALID_FONTS = all_ttfs[:5] if all_ttfs else [None]

BACKGROUND_COLORS = [
    (240, 240, 240),  # Neutral light grey
    (225, 230, 235),  # Sky grey
    (210, 215, 210),  # Muted greenery tint
    (190, 195, 200),  # Overcast cement
    (230, 225, 220),  # Warm asphalt dust
    (245, 245, 248),  # Clean studio white
    (200, 205, 210),  # Urban concrete
    (215, 220, 215)   # Natural roadside
]

def draw_base_sign(cls_key, size=64, font_path=None, bg_color=(240, 240, 240)):
    if font_path is None and VALID_FONTS and VALID_FONTS[0]:
        font_path = VALID_FONTS[0]
        
    pad = int(size * 0.06)
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # 1. Stop
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

    # 2. Speed Limits (30, 50, 80)
    elif cls_key in ["speed_30", "speed_50", "speed_80"]:
        border_w = max(4, int(size * 0.11))
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#FFFFFF", outline="#DC2626", width=border_w)
        num_str = cls_key.split("_")[1]
        if font_path:
            f = ImageFont.truetype(font_path, int(size * 0.40))
            draw.text((size // 2, size // 2 - int(size * 0.02)), num_str, fill="#000000", font=f, anchor="mm")
        else:
            draw.text((size // 2, size // 2), num_str, fill="#000000", anchor="mm")

    # 3. Speed Limit 100
    elif cls_key == "speed_100":
        border_w = max(4, int(size * 0.11))
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#FFFFFF", outline="#DC2626", width=border_w)
        if font_path:
            f = ImageFont.truetype(font_path, int(size * 0.31))
            draw.text((size // 2, size // 2 - int(size * 0.02)), "100", fill="#000000", font=f, anchor="mm")
        else:
            draw.text((size // 2, size // 2), "100", fill="#000000", anchor="mm")

    # 4. Yield
    elif cls_key == "yield":
        points = [(pad, pad), (size - pad, pad), (size // 2, size - pad)]
        border_w = max(4, int(size * 0.12))
        draw.polygon(points, fill="#FFFFFF", outline="#DC2626", width=border_w)
        if font_path:
            f = ImageFont.truetype(font_path, int(size * 0.16))
            draw.text((size // 2, int(size * 0.38)), "YIELD", fill="#DC2626", font=f, anchor="mm")
        else:
            draw.text((size // 2, int(size * 0.38)), "YIELD", fill="#DC2626", anchor="mm")

    # 5. No Entry
    elif cls_key == "no_entry":
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#DC2626", outline="#FFFFFF", width=max(2, int(size * 0.03)))
        bar_h = max(4, int(size * 0.18))
        draw.rectangle([pad + int(size * 0.12), size // 2 - bar_h // 2, size - pad - int(size * 0.12), size // 2 + bar_h // 2], fill="#FFFFFF")

    # 6. No Overtaking
    elif cls_key == "no_overtaking":
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#FFFFFF", outline="#DC2626", width=max(4, int(size * 0.11)))
        cy = size // 2
        # Left black car
        draw.rounded_rectangle([int(size*0.23), cy - int(size*0.08), int(size*0.45), cy + int(size*0.12)], radius=2, fill="#000000")
        draw.ellipse([int(size*0.27), cy - int(size*0.14), int(size*0.41), cy - int(size*0.06)], fill="#000000")
        # Right red car
        draw.rounded_rectangle([int(size*0.55), cy - int(size*0.08), int(size*0.77), cy + int(size*0.12)], radius=2, fill="#DC2626")
        draw.ellipse([int(size*0.59), cy - int(size*0.14), int(size*0.73), cy - int(size*0.06)], fill="#DC2626")

    # 7. Turn Right
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

    # 8. Turn Left
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

    # 9. Ahead Only
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

    # 10. Roundabout
    elif cls_key == "roundabout":
        draw.ellipse([pad, pad, size - pad, size - pad], fill="#2563EB", outline="#FFFFFF", width=max(2, int(size * 0.04)))
        draw.arc([int(size*0.25), int(size*0.25), int(size*0.75), int(size*0.75)], start=30, end=330, fill="#FFFFFF", width=max(2, int(size*0.07)))
        draw.polygon([(int(size*0.5), int(size*0.25)-int(size*0.06)), (int(size*0.5)+int(size*0.08), int(size*0.25)), (int(size*0.5), int(size*0.25)+int(size*0.06))], fill="#FFFFFF")

    # 11. Traffic Signals Ahead
    elif cls_key == "traffic_signals":
        pts = [(pad, pad), (size - pad, pad), (size // 2, size - pad)]
        draw.polygon(pts, fill="#FFFFFF", outline="#DC2626", width=max(3, int(size * 0.11)))
        cx = size // 2
        box_w, box_h = int(size * 0.16), int(size * 0.40)
        top_y = int(size * 0.20)
        draw.rectangle([cx - box_w//2, top_y, cx + box_w//2, top_y + box_h], fill="#1E293B")
        r_light = int(size * 0.04)
        draw.ellipse([cx - r_light, top_y + int(box_h*0.2) - r_light, cx + r_light, top_y + int(box_h*0.2) + r_light], fill="#EF4444")
        draw.ellipse([cx - r_light, top_y + int(box_h*0.5) - r_light, cx + r_light, top_y + int(box_h*0.5) + r_light], fill="#F59E0B")
        draw.ellipse([cx - r_light, top_y + int(box_h*0.8) - r_light, cx + r_light, top_y + int(box_h*0.8) + r_light], fill="#10B981")

    # 12. Road Work Ahead
    elif cls_key == "road_work":
        pts = [(pad, pad), (size - pad, pad), (size // 2, size - pad)]
        draw.polygon(pts, fill="#FFFFFF", outline="#DC2626", width=max(3, int(size * 0.11)))
        cx = size // 2
        draw.ellipse([cx - int(size*0.05), int(size*0.22), cx + int(size*0.05), int(size*0.32)], fill="#000000")
        draw.line([(cx, int(size*0.33)), (cx - int(size*0.08), int(size*0.52))], fill="#000000", width=max(2, int(size*0.06)))
        draw.line([(cx - int(size*0.08), int(size*0.52)), (cx - int(size*0.14), int(size*0.68))], fill="#000000", width=max(2, int(size*0.05)))
        draw.line([(cx - int(size*0.08), int(size*0.52)), (cx + int(size*0.04), int(size*0.68))], fill="#000000", width=max(2, int(size*0.05)))
        draw.line([(cx - int(size*0.02), int(size*0.42)), (cx + int(size*0.16), int(size*0.65))], fill="#000000", width=max(2, int(size*0.04)))
        draw.polygon([(cx + int(size*0.14), int(size*0.63)), (cx + int(size*0.22), int(size*0.68)), (cx + int(size*0.16), int(size*0.72))], fill="#000000")

    # 13. Pedestrian Crossing
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

    # 1. Random rotation (-15 to 15 deg)
    angle = random.uniform(-15, 15)
    bg_color = random.choice(BACKGROUND_COLORS)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=bg_color)

    # 2. Random Scale / Resizing jitter (0.85x to 1.15x)
    if random.random() > 0.35:
        w, h = img.size
        scale = random.uniform(0.85, 1.15)
        new_w, new_h = max(32, int(w * scale)), max(32, int(h * scale))
        img_scaled = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (w, h), color=bg_color)
        off_x = (w - new_w) // 2
        off_y = (h - new_h) // 2
        if off_x >= 0 and off_y >= 0:
            canvas.paste(img_scaled, (off_x, off_y))
        else:
            canvas = img_scaled.crop((-off_x, -off_y, -off_x + w, -off_y + h))
        img = canvas

    # 3. Random Brightness & Contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.72, 1.28))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.78, 1.28))

    # 4. Color temperature / saturation shift
    if random.random() > 0.4:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    # 5. Random subtle blur
    if random.random() > 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    # 6. Subtle sensor Gaussian noise
    arr = np.array(img, dtype=float)
    noise = np.random.normal(0, random.uniform(1.5, 4.0), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)

def generate_full_dataset(train_samples_per_class=90):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "data", "train")
    test_dir = os.path.join(base_dir, "data", "test_samples")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"Generating expanded Traffic Sign Dataset ({len(CLASSES)} classes, {train_samples_per_class} samples/class)...")
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
            chosen_bg = random.choice(BACKGROUND_COLORS)
            base = draw_base_sign(cls_key, size=64, font_path=chosen_font, bg_color=chosen_bg)
            aug = augment_image(base)
            aug.save(os.path.join(cls_folder, f"{cls_key}_{i:03d}.png"))

    total_train = len(CLASSES) * train_samples_per_class
    print(f"Dataset generated successfully! ({total_train} training images across {len(CLASSES)} classes in data/train/)")

if __name__ == "__main__":
    generate_full_dataset(train_samples_per_class=90)
