"""
Traffic Sign Dataset Generator
Generates clean and augmented synthetic traffic sign images for KNN training and testing.
Classes follow standard international / GTSRB categories.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

CLASSES = {
    "stop": {"shape": "octagon", "bg": "#DC2626", "text": "STOP", "fg": "#FFFFFF", "border": "#FFFFFF", "name": "Stop Sign"},
    "speed_30": {"shape": "circle", "bg": "#FFFFFF", "text": "30", "fg": "#000000", "border": "#DC2626", "name": "Speed Limit 30 km/h"},
    "speed_50": {"shape": "circle", "bg": "#FFFFFF", "text": "50", "fg": "#000000", "border": "#DC2626", "name": "Speed Limit 50 km/h"},
    "speed_80": {"shape": "circle", "bg": "#FFFFFF", "text": "80", "fg": "#000000", "border": "#DC2626", "name": "Speed Limit 80 km/h"},
    "yield": {"shape": "inv_triangle", "bg": "#FFFFFF", "text": "YIELD", "fg": "#DC2626", "border": "#DC2626", "name": "Yield / Give Way"},
    "no_entry": {"shape": "no_entry", "bg": "#DC2626", "text": "-", "fg": "#FFFFFF", "border": "#FFFFFF", "name": "No Entry"},
    "turn_right": {"shape": "arrow_right", "bg": "#2563EB", "text": "->", "fg": "#FFFFFF", "border": "#FFFFFF", "name": "Turn Right Ahead"},
    "turn_left": {"shape": "arrow_left", "bg": "#2563EB", "text": "<-", "fg": "#FFFFFF", "border": "#FFFFFF", "name": "Turn Left Ahead"},
    "ahead_only": {"shape": "arrow_up", "bg": "#2563EB", "text": "^", "fg": "#FFFFFF", "border": "#FFFFFF", "name": "Ahead Only"},
    "pedestrian": {"shape": "pedestrian", "bg": "#2563EB", "text": "PED", "fg": "#FFFFFF", "border": "#FFFFFF", "name": "Pedestrian Crossing"}
}

def draw_base_sign(cls_key, size=64):
    img = Image.new("RGB", (size, size), color=(235, 235, 235))
    draw = ImageDraw.Draw(img)
    cfg = CLASSES[cls_key]
    pad = 4

    if cfg["shape"] == "octagon":
        # Draw 8-sided polygon
        w = size - 2 * pad
        c = w / 3.0
        points = [
            (pad + c, pad), (pad + 2 * c, pad),
            (size - pad, pad + c), (size - pad, pad + 2 * c),
            (pad + 2 * c, size - pad), (pad + c, size - pad),
            (pad, pad + 2 * c), (pad, pad + c)
        ]
        draw.polygon(points, fill=cfg["bg"], outline=cfg["border"], width=3)
        # Add text
        draw.text((size // 2, size // 2), cfg["text"], fill=cfg["fg"], anchor="mm")

    elif cfg["shape"] == "circle":
        draw.ellipse([pad, pad, size - pad, size - pad], fill=cfg["bg"], outline=cfg["border"], width=6)
        draw.text((size // 2, size // 2), cfg["text"], fill=cfg["fg"], anchor="mm")

    elif cfg["shape"] == "inv_triangle":
        points = [(pad, pad), (size - pad, pad), (size // 2, size - pad)]
        draw.polygon(points, fill=cfg["bg"], outline=cfg["border"], width=6)
        draw.text((size // 2, size // 2 - 2), cfg["text"], fill=cfg["fg"], anchor="mm")

    elif cfg["shape"] == "no_entry":
        draw.ellipse([pad, pad, size - pad, size - pad], fill=cfg["bg"], outline=cfg["border"], width=3)
        bar_h = 10
        draw.rectangle([pad + 8, size // 2 - bar_h // 2, size - pad - 8, size // 2 + bar_h // 2], fill=cfg["fg"])

    elif cfg["shape"] == "arrow_right":
        draw.ellipse([pad, pad, size - pad, size - pad], fill=cfg["bg"], outline=cfg["border"], width=3)
        # Draw right arrow
        draw.line([(pad + 12, size // 2), (size - pad - 14, size // 2)], fill=cfg["fg"], width=5)
        draw.polygon([(size - pad - 10, size // 2), (size - pad - 20, size // 2 - 8), (size - pad - 20, size // 2 + 8)], fill=cfg["fg"])

    elif cfg["shape"] == "arrow_left":
        draw.ellipse([pad, pad, size - pad, size - pad], fill=cfg["bg"], outline=cfg["border"], width=3)
        # Draw left arrow
        draw.line([(pad + 14, size // 2), (size - pad - 12, size // 2)], fill=cfg["fg"], width=5)
        draw.polygon([(pad + 10, size // 2), (pad + 20, size // 2 - 8), (pad + 20, size // 2 + 8)], fill=cfg["fg"])

    elif cfg["shape"] == "arrow_up":
        draw.ellipse([pad, pad, size - pad, size - pad], fill=cfg["bg"], outline=cfg["border"], width=3)
        # Draw upward arrow
        draw.line([(size // 2, size - pad - 12), (size // 2, pad + 14)], fill=cfg["fg"], width=5)
        draw.polygon([(size // 2, pad + 10), (size // 2 - 8, pad + 20), (size // 2 + 8, pad + 20)], fill=cfg["fg"])

    elif cfg["shape"] == "pedestrian":
        # Draw blue square with walking stick figure
        draw.rectangle([pad, pad, size - pad, size - pad], fill=cfg["bg"], outline=cfg["border"], width=3)
        # Head
        draw.ellipse([size // 2 - 3, pad + 10, size // 2 + 3, pad + 16], fill=cfg["fg"])
        # Body
        draw.line([(size // 2, pad + 17), (size // 2, pad + 32)], fill=cfg["fg"], width=4)
        # Legs
        draw.line([(size // 2, pad + 32), (size // 2 - 8, size - pad - 10)], fill=cfg["fg"], width=3)
        draw.line([(size // 2, pad + 32), (size // 2 + 8, size - pad - 10)], fill=cfg["fg"], width=3)
        # Arms
        draw.line([(size // 2 - 8, pad + 24), (size // 2 + 8, pad + 20)], fill=cfg["fg"], width=3)

    return img

def augment_image(base_img):
    img = base_img.copy()

    # Random rotation (-12 to 12 degrees)
    angle = random.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(235, 235, 235))

    # Random Brightness & Contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.75, 1.25))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random slight blur or noise
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

    # Convert to array to add subtle Gaussian noise
    arr = np.array(img, dtype=float)
    noise = np.random.normal(0, 3.5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)

def generate_full_dataset(train_samples_per_class=35):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "data", "train")
    test_dir = os.path.join(base_dir, "data", "test_samples")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Generating Traffic Sign Dataset...")
    for cls_key in CLASSES:
        cls_folder = os.path.join(train_dir, cls_key)
        os.makedirs(cls_folder, exist_ok=True)

        base_img = draw_base_sign(cls_key, size=64)

        # Save clean test sample
        test_sample_path = os.path.join(test_dir, f"{cls_key}_sample.png")
        base_img.save(test_sample_path)

        # Generate augmented training samples
        for i in range(train_samples_per_class):
            aug = augment_image(base_img)
            aug.save(os.path.join(cls_folder, f"{cls_key}_{i:03d}.png"))

    print(f"Dataset generated successfully! ({len(CLASSES) * train_samples_per_class} training images in data/train/, {len(CLASSES)} test samples in data/test_samples/)")

if __name__ == "__main__":
    generate_full_dataset()
