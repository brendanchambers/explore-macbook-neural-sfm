
## Project info
We will be extracting dense features from video frames.
The language will be python, with env managed with uv.
The development and compute will happen on this macbook air M4 24GB laptop.
Configuration will be managed with hydra + omegaconf, and their logs will be written at `logs/hydra`.
Never change validation approaches without discussing it first and receiving a greenlight.

## Project Goal
We will be attempting to create a 3D gaussian splat from extracted frames and structure from motion (`data/intermediate/frames` and `data/intermediate/sfm`). The computation will be happening on a macbook air M4 24gb. We will use opensplat to initialize and train the 3D gaussian splat (`https://github.com/pierotofy/opensplat`).

## Workflow info 
- For other materials, e.g. run instructions, and other project info, maintain documentation in README.md.  
- Do not create new markdown files in the top level project directory.  
- Document work sessions at `reports/work_history` instead of CLAUDE.md or console prints alone.
- Check for out-of-date info in CLAUDE.md and README.md and maintain very concise documentation.  

## Inner and outer repo organization for rapid prototyping
We will sometimes use an inner project repo pasted into the project. Avoid modifying this repository unless absolutely necessary. We will do our work in the outer repository. The inner projects are intended as key dependencies but sometimes it's simpler to have their code available here in the repo during exploration and prototyping phases.

## Experiments
Define `experiment_name` and `experiment_group`. Often experiment_group will simply be `current_experiment`. Experiment group is used to automatically organize results for easy comparison, e.g. `reports/experiments/<experiment_group>.jsonl`. Experiment name is used to organize directories, e.g. `data/intermediates/<experiment_name>`.