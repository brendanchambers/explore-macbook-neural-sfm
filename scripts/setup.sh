source .env

cd $project_dir

# for intial explorations:
git clone https://github.com/aedelon/mlx-mast3r.git
cd mlx-mast3r
rm -rf .git
uv sync

# or, uv add mlx-mast3r