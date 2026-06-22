import numpy as np
from PIL import Image
from mlx_mast3r import Mast3rFull

# Load images
img1 = np.array(Image.open("data/incoming/images/frame_0001.jpg"))
img2 = np.array(Image.open("data/incoming/images/frame_0002.jpg"))

# Load full MASt3R pipeline
model = Mast3rFull.from_pretrained(resolution=512)

# Reconstruct 3D
out1, out2 = model.reconstruct(img1, img2)