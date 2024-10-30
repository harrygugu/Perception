import os
from PIL import Image

# Paths to input and output folders
input_folder = "data\images"
output_folder = "data\images_subsample"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of image files in input folder, sorted to maintain sequence
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Subsample images every 5th and rename
for i, filename in enumerate(image_files[::5]):
    # Load image
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path)

    # Save with new name in output folder
    new_name = f"{i}.png"
    img.save(os.path.join(output_folder, new_name))

print("Subsampling complete!")
