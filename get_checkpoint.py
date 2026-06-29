from huggingface_hub import hf_hub_download
import metatomic.torch   # registers metatomic Torch classes
import metatensor.torch  # useful for metatensor Torch classes
import torch

path = hf_hub_download(
    repo_id="lab-cosmo/upet",
    # filename="pet-oam-l-v0.1.0.ckpt",
    filename="pet-oam-xl-v1.0.0.ckpt",
    subfolder="models",
)

print("checkpoint path:", path)


ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(ckpt.keys())
print(ckpt["model_data"]["model_hypers"])
print(ckpt["model_data"]["dataset_info"])
print(ckpt.get("train_hypers"))