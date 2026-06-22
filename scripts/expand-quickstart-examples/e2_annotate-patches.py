from mlx_mast3r import DUNE
from PIL import Image, ImageDraw
import numpy as np
import os

# Load encoder
encoder = DUNE.from_pretrained("base", resolution=336)

# Load image
image_path = "data/incoming/images/frame_0001.jpg"
image = np.array(Image.open(image_path))
original_image = Image.open(image_path)

# Extract features
features = encoder.encode(image)  # [N, 768]

print(f"Features shape: {features.shape}")  # 10,549 x 768
print(f"Number of detected features: {features.shape[0]}")

# Visualize features on image
# Create a copy of the original image for annotation
annotated_image = original_image.copy().convert("RGBA")
overlay = Image.new("RGBA", annotated_image.size, (255, 255, 255, 0))
draw = ImageDraw.Draw(overlay)

# Get the spatial locations of features
# DUNE encodes features at multiple scales; we'll map them to the original image
# Features are typically arranged in a grid pattern from the encoder
h, w = original_image.size[::-1]  # Get height and width
num_features = features.shape[0]

# Calculate feature positions in a grid layout
# Assuming features are in row-major order from the encoder
grid_h = int(np.sqrt(num_features))
grid_w = (num_features + grid_h - 1) // grid_h

# Calculate patch size based on grid layout
patch_h = h / grid_h
patch_w = w / grid_w

# Map grid positions to image coordinates and draw patches
for idx in range(num_features):
    row = idx // grid_w
    col = idx % grid_w

    # Calculate patch boundaries
    x_min = int(col * patch_w)
    y_min = int(row * patch_h)
    x_max = int((col + 1) * patch_w)
    y_max = int((row + 1) * patch_h)

    # Draw semi-transparent box
    draw.rectangle(
        [x_min, y_min, x_max, y_max],
        outline=(0, 255, 0, 150),  # Semi-transparent green outline
        width=1
    )

# Composite overlay onto original image
annotated_image = Image.alpha_composite(annotated_image, overlay).convert("RGB")

# Save annotated image
output_dir = "data/intermediates/feature_extraction"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "frame_0001_features.jpg")
annotated_image.save(output_path, quality=95)

print(f"Annotated image saved to {output_path}")




### FUTURE: can we get these features into a format accepted by COLAMP feature matching or use neural feature matching?
# note opensplat compatible formats:
# COLMAP
# Nerfstudio
# OpenMVG (Open Multiple View Geometry) 
# OpenSfM
# ODM (OpenDroneMap)



 #LEARNING:
# For downstream uses, you can compute the cosine similarity between the 768-dimensional vectors of two different images to find exact pixel-level correspondences, or you can feed the entire N×768 tensor into a decoder to generate 3D point clouds.
# like the DuneMASt3R decoder for fast 3D stereo reconstruction