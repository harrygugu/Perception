import os
from PIL import Image

# Paths to input and output folders
input_folder = "data/exploration_data_maze_2/images"
output_folder = "data/exploration_data_maze_2/images_subsample"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Subsample and process every 5th image
for i in range(len(os.listdir(input_folder))):
    print(f"\rLoading image {i}/{len(os.listdir(input_folder))}", end="")
    image_path = os.path.join(input_folder, f"image_{i}.png")

    # Check if the image exists
    if os.path.exists(image_path):
        # Only process every 5th image
        if i % 5 == 0:
            try:
                # Open the image
                with Image.open(image_path) as img:
                    # Save the image with a new name
                    new_name = f"{i // 5}.png"
                    img.save(os.path.join(output_folder, new_name))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    else:
        print(f"Image {image_path} does not exist.")
        break

print("Subsampling complete!")
