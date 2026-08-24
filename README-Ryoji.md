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

[July 2026] Additional installation of PyG for structure transformer.
```bash
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.12.1+cu130.html
pip install torch_geometric
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

## Transformer on Mptrj


```bash
# all mptrj training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct.yaml \
  -o structure-transformer-mptrj-salex-ddp.pt \
  -r device=cuda \
  -r architecture.training.distributed=true \
  -r architecture.training.distributed_port=39591

# 160k
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct-160k.yaml \
  -o structure-transformer-mptrj-salex-ddp-160k.pt \
  -r device=cuda \
  -r architecture.training.distributed=true \
  -r architecture.training.distributed_port=39591 \
  -r training_set.indices=indices/mptrj_160k_seed0.txt

# 160k cartessian
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct-cartesian-160k.yaml \
  -o structure-transformer-mptrj-salex-ddp-cartesian-160k.pt \
  -r device=cuda \
  -r architecture.training.distributed=true \
  -r architecture.training.distributed_port=39591 \
  -r training_set.indices=indices/mptrj_160k_seed0.txt

# all mptrj training
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct-200m.yaml \
  -o structure-transformer-mptrj-salex-ddp-200m.pt \
  -r device=cuda \
  -r architecture.training.distributed=true \
  -r architecture.training.distributed_port=39591

# 200m cartesian
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct-200m-cartesian.yaml \
  -o structure-transformer-mptrj-salex-ddp-200m-cartesian.pt \
  -r device=cuda \
  -r architecture.training.distributed=true \
  -r architecture.training.distributed_port=39591

# 160k
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  -m metatrain train options-structure-transformer-mptrj-salex-direct-160k-v37.yaml \
  -o structure-transformer-mptrj-salex-ddp-160k-v37.pt \
  -r device=cuda \
  -r architecture.training.distributed=true \
  -r architecture.training.distributed_port=39591 \
  -r training_set.indices=indices/mptrj_160k_seed0.txt
```

# Sending data
```bash
rsync -avz -e ssh outputs ryoji@deeplearn24.ddns.comp.nus.edu.sg:/home/ryoji/equivarient/metatrain

```