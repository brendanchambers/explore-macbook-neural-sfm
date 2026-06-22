from mlx_mast3r import DUNE
from PIL import Image
import numpy as np

# Load encoder
encoder = DUNE.from_pretrained("base", resolution=336)

# Extract features
image = np.array(Image.open("data/incoming/images/frame_0001.jpg"))
features = encoder.encode(image)  # [N, 768]