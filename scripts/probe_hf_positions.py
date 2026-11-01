"""Record the text position_ids HF GIT uses during prefill and decode."""

import torch
from PIL import Image
from transformers import AutoProcessor, GitForCausalLM

MODEL = "microsoft/git-base-coco"
proc = AutoProcessor.from_pretrained(MODEL)
model = GitForCausalLM.from_pretrained(MODEL).eval()

seen = []
orig = model.git.embeddings.position_embeddings.forward
model.git.embeddings.position_embeddings.forward = lambda ids: (
    seen.append(ids.flatten().tolist()),
    orig(ids),
)[1]

image = Image.new("RGB", (224, 224), (200, 30, 30))
inputs = proc(images=image, return_tensors="pt")
with torch.no_grad():
    out = model.generate(pixel_values=inputs.pixel_values, max_new_tokens=5)
print("tokens:", out[0].tolist())
for step in seen:
    print(step)
