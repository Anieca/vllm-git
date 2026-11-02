"""A prefix cache hit that ends inside an image must not split the rest of it.

Runs a small Gemma3 (bidirectional attention over the image) on CPU. A primer
request caches the start of the image. Then a request with the same image and a
different question hits that cache inside the image, in a step whose token
budget is mostly used by a text-only request. Its logprobs are compared with
the same request run alone, without prefix caching.

Usage: python scripts/repro_mid_item_split.py MODEL_DIR [MODEL_DIR ...]
(MODEL_DIR from `scripts/build_tiny_gemma3.py DIR --layers 6 --std 2`;
TRACE_SCHEDULE=1 also prints the tokens scheduled per step)
"""

import gc
import os
import sys

from transformers import AutoConfig, AutoProcessor
from vllm.assets.image import ImageAsset

from vllm import LLM, SamplingParams

BLOCK_SIZE = 128  # the CPU default; set explicitly so GPUs (default 16) match
BUDGET = 512
GPU_MEMORY = 0.4  # two engines in one process on GPU; ignored on CPU


def question_prompt(pad: int, question: str) -> str:
    return (
        "<bos><start_of_turn>user\n"
        + "word " * pad
        + f"<start_of_image>{question}<end_of_turn>\n<start_of_turn>model\n"
    )


def trace_schedule():
    """Print the tokens scheduled per request in every step (single process)."""
    from vllm.v1.core.sched.scheduler import Scheduler

    schedule = Scheduler.schedule

    def traced(self, *args, **kwargs):
        output = schedule(self, *args, **kwargs)
        if output.num_scheduled_tokens:
            print("STEP", list(output.num_scheduled_tokens.values()))
        return output

    Scheduler.schedule = traced

    from vllm.v1.attention.backends.cpu_attn import CPUAttentionMetadataBuilder

    build = getattr(CPUAttentionMetadataBuilder, "_build_mm_prefix", None)
    if build is None:  # without Anieca/vllm#2
        return

    def traced_build(self, common):
        meta = build(self, common)
        if meta is not None:
            rows = meta.token_indices
            print(
                "  MM_REPLAY rows",
                (int(rows[0]), int(rows[-1])),
                "kv_lens",
                meta.seq_lens.tolist(),
                "| step seq_lens",
                common.seq_lens.tolist(),
                "upper bound",
                None
                if common.seq_lens_cpu_upper_bound is None
                else common.seq_lens_cpu_upper_bound.tolist(),
            )
        return meta

    CPUAttentionMetadataBuilder._build_mm_prefix = traced_build


def main():
    # One process, so that both requests of the second call reach the scheduler
    # before its first step. Otherwise the text-only request can run alone first.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if os.environ.get("TRACE_SCHEDULE"):
        trace_schedule()
    for path in sys.argv[1:]:
        run(path)


def run(path: str):
    image = ImageAsset("stop_sign").pil_image.convert("RGB")
    processor = AutoProcessor.from_pretrained(path)
    image_token = AutoConfig.from_pretrained(path).image_token_index

    # Pick the padding so that the image ends ~100 tokens past a block boundary:
    # a cache hit then ends ~100 image tokens before the end of the image.
    for pad in range(300):
        ids = processor(
            text=question_prompt(pad, "A"), images=image, return_tensors="pt"
        )["input_ids"][0]
        image_end = int((ids == image_token).nonzero()[-1]) + 1
        if image_end % BLOCK_SIZE >= 100:
            break
    hit_len = image_end - image_end % BLOCK_SIZE
    print(f"pad {pad}: image ends at {image_end}, a cache hit ends at {hit_len}")

    leftover = 40  # budget left for the hit request after the text request
    filler = "<bos>" + " lorem" * (BUDGET - leftover - 1)
    primer = {
        "prompt": question_prompt(pad, "What is in the image?"),
        "multi_modal_data": {"image": image},
    }
    hit = {
        "prompt": question_prompt(pad, "What color is it?"),
        "multi_modal_data": {"image": image},
    }
    params = SamplingParams(temperature=0.0, max_tokens=4, logprobs=5)

    llm = LLM(
        model=path,
        dtype="float32",
        max_model_len=2048,
        max_num_batched_tokens=BUDGET,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=GPU_MEMORY,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    llm.generate([primer], params, use_tqdm=False)
    _, hit_out = llm.generate([{"prompt": filler}, hit], params, use_tqdm=False)
    print(
        f"cache hit: {hit_out.num_cached_tokens} of {len(hit_out.prompt_token_ids)} tokens"
    )
    del llm
    gc.collect()

    ref = LLM(
        model=path,
        dtype="float32",
        max_model_len=2048,
        enable_prefix_caching=False,
        gpu_memory_utilization=GPU_MEMORY,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
    )
    ref_out = ref.generate([hit], params, use_tqdm=False)[0]

    diffs = []
    for a, b in zip(hit_out.outputs[0].logprobs, ref_out.outputs[0].logprobs):
        diffs += [abs(lp.logprob - b[tok].logprob) for tok, lp in a.items() if tok in b]
    same = hit_out.outputs[0].token_ids == ref_out.outputs[0].token_ids
    first = hit_out.outputs[0].logprobs[0]
    print("LOGPROBS", [f"{lp.logprob:.9f}" for lp in first.values()])
    print(
        f"RESULT {path}: same tokens {same}, max |logprob diff| vs alone {max(diffs):.2e}"
    )


if __name__ == "__main__":
    main()
