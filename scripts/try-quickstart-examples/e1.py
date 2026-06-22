import numpy as np
from PIL import Image
from mlx_mast3r import DuneMast3r

# Load images
img1 = np.array(Image.open("data/incoming/images/frame_0001.jpg"))
img2 = np.array(Image.open("data/incoming/images/frame_0002.jpg"))

# Load model (downloads weights automatically)
model = DuneMast3r.from_pretrained("base", resolution=336)

# Reconstruct 3D from stereo pair
out1, out2 = model.reconstruct(img1, img2)

# Access outputs
pts3d = out1["pts3d"]      # [H, W, 3] - 3D points
conf = out1["conf"]        # [H, W] - confidence map
desc = out1["desc"]        # [H, W, 24] - descriptors