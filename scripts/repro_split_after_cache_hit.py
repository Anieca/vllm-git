"""A prefix cache hit that ends inside an image must not split the rest of it.

A primer caches the start of an image. A request with the same image and another
question then hits that cache inside the image, in a step whose token budget a
text-only request mostly uses. Its logprobs are compared with the same request
run alone. TRACE=1 prints the tokens scheduled per request in each step.

    python repro_split_after_cache_hit.py [MODEL] [DTYPE]
"""

import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # both requests in one step

from transformers import AutoConfig, AutoProcessor
from vllm.assets.image import ImageAsset

from vllm import LLM, SamplingParams

MODEL = sys.argv[1] if len(sys.argv) > 1 else "unsloth/gemma-3-4b-it"
DTYPE = sys.argv[2] if len(sys.argv) > 2 else "float32"
BLOCK, BUDGET, LEFT = 128, 512, 40

if os.environ.get("TRACE"):
    from vllm.v1.core.sched.scheduler import Scheduler

    schedule = Scheduler.schedule

    def traced(self, *args, **kwargs):
        out = schedule(self, *args, **kwargs)
        print("STEP", dict(out.num_scheduled_tokens))
        return out

    Scheduler.schedule = traced


def prompt(pad: int, question: str) -> str:
    return (
        "<bos><start_of_turn>user\n"
        + "word " * pad
        + f"<start_of_image>{question}<end_of_turn>\n<start_of_turn>model\n"
    )


image = ImageAsset("stop_sign").pil_image.convert("RGB")
processor = AutoProcessor.from_pretrained(MODEL)
image_token = AutoConfig.from_pretrained(MODEL).image_token_index
# Pad so that the image ends >= 100 tokens past a block boundary: the hit then
# stops >= 100 image tokens before the end of the image.
for pad in range(BLOCK):
    ids = processor(text=prompt(pad, "A"), images=image, return_tensors="pt")
    image_end = int((ids["input_ids"][0] == image_token).nonzero()[-1]) + 1
    if image_end % BLOCK >= 100:
        break


def request(question: str) -> dict:
    return {"prompt": prompt(pad, question), "multi_modal_data": {"image": image}}


llm = LLM(
    MODEL,
    dtype=DTYPE,
    max_model_len=2048,
    max_num_batched_tokens=BUDGET,
    block_size=BLOCK,
    enable_prefix_caching=True,
    enforce_eager=True,
    limit_mm_per_prompt={"image": 1},
)
params = SamplingParams(temperature=0.0, max_tokens=4, logprobs=5)
hit = request("What color is it?")

alone = llm.generate([hit], params, use_tqdm=False)[0]
llm.reset_prefix_cache()
llm.generate([request("What is in the image?")], params, use_tqdm=False)
filler = {"prompt": "<bos>" + " lorem" * (BUDGET - LEFT - 1)}
out = llm.generate([filler, hit], params, use_tqdm=False)[1]

diff = max(
    abs(lp.logprob - ref[tok].logprob)
    for got, ref in zip(out.outputs[0].logprobs, alone.outputs[0].logprobs)
    for tok, lp in got.items()
    if tok in ref
)
print(
    f"image ends at {image_end}, cache hit {out.num_cached_tokens} tokens, "
    f"max |logprob diff| vs alone {diff:.2e}"
)
