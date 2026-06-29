# Reproducing PET

## Installation

```bash
conda create -n metatrain-pet python=3.11
conda activate metatrain-pet
python -m pip install --upgrade pip
python -m pip install -e ".[pet]"
python -m pip install wandb
pip install ipykernel
```

## Reproducing Native PET Tutorial (options-scratch.yaml)

The documentation for how this repo works can be found in [PET_TUTORIAL_REPO_GUIDE.md](PET_TUTORIAL_REPO_GUIDE.md). For details on how training is done, refer to [PET_TUTORIAL_TRAINING_FLOW.md](PET_TUTORIAL_TRAINING_FLOW.md).
```bash
cd examples/0-beginner
python -m metatrain train options-scratch.yaml
```