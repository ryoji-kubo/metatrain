# Minimal Conda environment for the PET notebook

This setup installs Python 3.11, PyTorch 2.12.1, and `ipykernel`. Python and
PyTorch match the versions used to validate the notebook. The two direct pip
packages are `torch` (the model) and `ipykernel` (notebook execution); their
required dependencies are installed automatically.

**1. Create the environment.** Conda must already be installed.

```bash
conda create -n pet-notebook python=3.11 pip --no-default-packages -y
conda activate pet-notebook
```

`--no-default-packages` keeps Conda from adding packages from a personal default
package list. See [Conda's environment guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

**2. Install PyTorch. Choose one command.**

For this Linux machine with an NVIDIA GPU, use the CUDA 13.0 build that was
used to validate the notebook:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
```

For CPU execution on Linux or Windows, use the smaller CPU-only build instead:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

These builds are listed in the [official PyTorch 2.12.1 instructions](https://pytorch.org/get-started/previous-versions/#v2121).
For other hardware, choose the appropriate build from those instructions and
install only `torch`. The notebook does not use `torchvision` or `torchaudio`.
The full example used about 9 GiB of GPU memory; CPU execution is slower.

**3. Install and register the notebook kernel.**

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name pet-notebook --display-name "Python (PET notebook)"
```

Registration makes this environment available to an existing notebook interface,
such as VS Code or Jupyter. This follows the [IPython kernel setup guide](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments).

**4. Check the installation and open the notebook.**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The second line should be `True` for GPU execution and `False` for the CPU-only
build. Keep `pet_isolated.ipynb` and `pet_input.pt` together, open the notebook,
select **Python (PET notebook)**, and run all cells. You can use the files here
in `notebooks/`, or copy both to another directory.

**Optional: run JupyterLab in a browser.**

If you do not already have a notebook interface, add JupyterLab to the activated
environment. From the directory containing the notebook and input file, run:

```bash
python -m pip install jupyterlab
jupyter lab pet_isolated.ipynb
```

JupyterLab is an optional interface, not a model dependency. See the
[JupyterLab installation guide](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html).
