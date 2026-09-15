# Explore PET in a notebook

[pet_isolated.ipynb](pet_isolated.ipynb) shows the PET architecture in editable
PyTorch code, from attention and message passing to energy, force, and stress
predictions. It runs one forward pass, computes the losses, and performs one
Adam update. Intermediate tensor shapes help you follow the calculation.

To get started:

1. Follow [SETUP.md](SETUP.md) to create a small Conda environment.
2. Open the notebook and select the **Python (PET notebook)** kernel.
3. Run all cells in order.

Keep these two files together. You can copy them to another directory and run
without the rest of this repository.

| File | Purpose |
|---|---|
| [pet_isolated.ipynb](pet_isolated.ipynb) | Model code, explanations, and the optimizer step |
| [pet_input.pt](pet_input.pt) | Prepared input and targets for 16 structures / 512 atoms, plus the configuration |

The architecture follows
[the requested PET configuration](options-pet-oam-l-modern-mptrj-salex-direct-epoch@50.yaml).
All preprocessing is already saved in the input file. The notebook needs only
PyTorch and a notebook kernel; the original dataset and checkpoints are not
needed to run it.

Weights start from a random initialization. Predictions are in the normalized
space used for training, so this example is for understanding the model rather
than measuring prediction accuracy. The single Adam update uses the configured
base learning rate; the multi-epoch scheduler is omitted.

The notebook was checked against the original PET for predictions, losses, and
parameter gradients. The full example used about **9 GiB of GPU memory** on an
RTX A5000. CPU execution is available but slower.

To inspect the attention calculation directly, set `USE_MANUAL_ATTENTION = True`
and rerun from model construction.

If you need a different prepared batch, run
`python scripts/prepare_pet_notebook_input.py` from the repository root using
the original `metatrain-pet` environment. Regeneration needs the original MPtrj
data and statistics checkpoint; the small notebook environment is for running
the saved example.
