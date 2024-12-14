import os
import json

# Define the folder paths
image_folder = "data/exploration_data_maze_1/images_filtered/"  # Replace with the actual path
json_file_path = "data/exploration_data_maze_1/image_actions.json"
new_json_file_path = "data/exploration_data_maze_1/image_filtered.json"

# Load the JSON file
with open(json_file_path, "r") as file:
    actions = json.load(file)

# Initialize new JSON content
new_actions = {}

# Keep track of the new file names
new_index = 0

# Iterate through the actions and process images
for key, value in actions.items():
    image_name = f"{value['image']}.png"
    image_path = os.path.join(image_folder, image_name)

    if value["action"] != "Action.IDLE":
        # New image name
        new_image_name = f"image_{new_index}.png"
        new_image_path = os.path.join(image_folder, new_image_name)

        # Rename the image file
        if os.path.exists(image_path):
            os.rename(image_path, new_image_path)

        # Update the new actions dictionary
        new_actions[new_index] = {
            "image": f"image_{new_index}",
            "action": value["action"]
        }
        new_index += 1
    else:
        # Remove the image file corresponding to Action.IDLE
        if os.path.exists(image_path):
            os.remove(image_path)

# Write the updated actions to the JSON file
with open(new_json_file_path, "w") as file:
    json.dump(new_actions, file, indent=4)

print("Processing complete. Updated JSON saved as action_filtered.json.")
