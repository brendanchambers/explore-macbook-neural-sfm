from mlx_mast3r import Mast3rFull, make_pairs_retrieval, RetrievalModel
from mlx_mast3r.utils import load_image
import mlx.core as mx


# Load models
model = Mast3rFull.from_pretrained(resolution=512)
retrieval = RetrievalModel.from_pretrained()

# Load images
images = [
    "data/incoming/images/frame_0001.jpg",
    "data/incoming/images/frame_0002.jpg",
    "data/incoming/images/frame_0003.jpg",
    "data/incoming/images/frame_0004.jpg"]
resolution = 336

imgs_data = []
imgs_arrays = []
for idx, path in enumerate(images):
    img = load_image(path, resolution=resolution)
    imgs_data.append({
        "img": mx.array(img).transpose(2, 0, 1)[None],
        "true_shape": img.shape[:2],
        "idx": idx,
        "instance": path,
    })
    imgs_arrays.append(img)

# Select pairs using visual similarity
pairs_indices = make_pairs_retrieval(
    retrieval=retrieval,
    backbone=model,
    images=imgs_arrays,  # List of numpy images
    na=20,              # Number of adjacent candidates
    k=10,               # Pairs per image
)