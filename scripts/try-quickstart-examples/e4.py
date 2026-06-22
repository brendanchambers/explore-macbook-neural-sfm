from mlx_mast3r import DuneMast3r
from mlx_mast3r.cloud_opt import sparse_global_alignment
from mlx_mast3r.image_pairs import make_pairs
from mlx_mast3r.utils import load_image
import mlx.core as mx

# Load model
model = DuneMast3r.from_pretrained(encoder_variant="base", resolution=336)

# Load images
images = [
    "data/incoming/images/frame_0001.jpg",
    "data/incoming/images/frame_0002.jpg",
    "data/incoming/images/frame_0003.jpg",
    "data/incoming/images/frame_0004.jpg"]
resolution = 336

imgs_data = []
for idx, path in enumerate(images):
    img = load_image(path, resolution=resolution)
    imgs_data.append({
        "img": mx.array(img).transpose(2, 0, 1)[None],
        "true_shape": img.shape[:2],
        "idx": idx,
        "instance": path,
    })

# Build pairs (complete graph for small sets)
pairs = make_pairs(imgs_data, scene_graph="complete", symmetrize=True)

# Run global alignment
result = sparse_global_alignment(
    imgs=images,
    pairs_in=pairs,
    cache_path="/tmp/cache",
    model=model,
    lr1=0.07,      # Coarse phase learning rate
    niter1=300,    # Coarse phase iterations
    lr2=0.01,      # Fine phase learning rate
    niter2=300,    # Fine phase iterations
)

# Access results
poses = result.get_im_poses()      # [N, 4, 4] camera-to-world matrices
focals = result.get_focals()       # [N] focal lengths
pts3d, depths, confs = result.get_dense_pts3d()  # Dense reconstruction