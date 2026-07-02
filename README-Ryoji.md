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


## Training PET on MPtrj


```bash
# 160k subsplit
python -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-modern-mptrj160k-salex-direct.pt \
  -r training_set.indices=indices/mptrj_160k_seed0.txt

# 160k subsplit on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=39591 \
  -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-modern-mptrj160k-salex-direct-ddp.pt \
  -r architecture.training.distributed=true \
  -r training_set.indices=indices/mptrj_160k_seed0.txt

# Full mptrj on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,3 torchrun \
  --nnodes=1 \
  --nproc_per_node=3 \
  --master_addr=127.0.0.1 \
  --master_port=39591 \
  -m metatrain train options-pet-oam-l-modern-mptrj-salex-direct.yaml \
  -o pet-oam-l-modern-mptrj-salex-direct-ddp.pt \
  -r architecture.training.distributed=true \
```

# Sending data
```bash
rsync -avz -e ssh outputs ryoji@deeplearn24.ddns.comp.nus.edu.sg:/home/ryoji/equivarient/metatrain

```