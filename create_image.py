import os
from PIL import Image
import random

def find_png_images(folder):
    """Find and return all .png files from a folder."""
    png_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files

def compose_images(image_paths, background_size=(800, 600)):
    """Compose images onto a background, keeping transparency."""
    bg = Image.new('RGBA', background_size, (255, 255, 255, 255))

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGBA')

        datas = img.getdata()
        new_data = []
        for item in datas:
            if item[:3] == (255, 255, 255):
                new_data.append((255, 255, 255, 0))  # Make white transparent
            else:
                new_data.append(item)
        img.putdata(new_data)

        # Generate random position
        x = random.randint(0, background_size[0] - img.width)
        y = random.randint(0, background_size[1] - img.height)

        # Paste onto background with transparency
        bg.paste(img, (x, y), img)

    return bg.convert("RGB")

def get_first_image_from_subfolders(base_folder):
    """Get the first image from each subfolder (circle, square, star, triangle)."""
    shapes = ["circle", "square", "star", "triangle"]
    selected_images = []

    for shape in shapes:
        shape_folder = os.path.join(base_folder, shape)
        if os.path.exists(shape_folder):
            # Find all .png images in the subfolder
            png_images = find_png_images(shape_folder)
            if png_images:
                selected_images.append(png_images[0])  # Add the first image
            else:
                print(f"No .png files found in {shape_folder}")
        else:
            print(f"Missing directory for shape: {shape}")

    return selected_images

# ---- SETTINGS ----
input_folder = "./four_shapes/2/shapes/"  # Folder containing 'circle', 'square', 'star', 'triangle' subfolders
output_file = "output_image.jpg"
background_size = (800, 600)  # Adjust if needed

# ---- EXECUTION ----
selected_images = get_first_image_from_subfolders(input_folder)

if selected_images:
    final_image = compose_images(selected_images, background_size)
    final_image.save(output_file)
    print(f"Saved composed image with {len(selected_images)} shapes to {output_file}")
else:
    print("No images selected to compose.")
