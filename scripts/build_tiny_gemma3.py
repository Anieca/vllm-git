"""Build a small randomly initialised Gemma3ForConditionalGeneration.

`hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration` has a single
text layer, in which the KV of a token never depends on other tokens, so it
cannot show a wrong attention mask. This adds layers and widens the weights so
the logits are far from uniform.

Usage: python scripts/build_tiny_gemma3.py OUT_DIR [--layers 3] [--std 0.5]
"""

import argparse

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

SRC = "hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration"

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--std", type=float, default=0.5)
args = parser.parse_args()

config = AutoConfig.from_pretrained(SRC)
config.text_config.num_hidden_layers = args.layers
config.text_config.layer_types = ["full_attention"] * args.layers

torch.manual_seed(0)
model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
with torch.no_grad():
    for param in model.parameters():
        param.normal_(0, args.std)

model.save_pretrained(args.out_dir)
AutoProcessor.from_pretrained(SRC).save_pretrained(args.out_dir)

# Transformers v5 saves the vision tower without the `vision_model.` level that
# the Hub checkpoints (and vLLM's loader) use.
path = f"{args.out_dir}/model.safetensors"
tensors = load_file(path)
tensors = {
    k.replace("vision_tower.", "vision_tower.vision_model.", 1)
    if k.startswith("vision_tower.") and not k.startswith("vision_tower.vision_model.")
    else k: v
    for k, v in tensors.items()
}
save_file(tensors, path, metadata={"format": "pt"})
print("saved", args.out_dir, sum(p.numel() for p in model.parameters()), "parameters")
