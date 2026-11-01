"""Throughput / latency of GIT: HF Transformers `generate` vs vLLM (paged attention).

Every request uses a different image (no prefix cache hits) and generates exactly
`--output-len` tokens (EOS ignored), so both sides do the same work. Image
preprocessing is included in the timings of both. Each configuration runs in its
own process to keep memory usage down.

Usage:
    python scripts/bench_hf_vs_vllm.py [--num-requests 64] [--output-len 32]
        [--hf-batch-size 8] [--configs hf,vllm,vllm-no-mm-prefix]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

MODEL = "microsoft/git-base-coco"


def make_images(n: int, seed: int = 0) -> list[Image.Image]:
    """Distinct 384x384 images (random crops + tints of the vLLM image assets)."""
    from vllm.assets.image import ImageAsset

    rng = np.random.default_rng(seed)
    bases = [
        np.asarray(ImageAsset(name).pil_image.convert("RGB"), dtype=np.float32)
        for name in ("stop_sign", "cherry_blossom")
    ]
    images = []
    for i in range(n):
        base = bases[i % len(bases)]
        h, w = base.shape[:2]
        size = int(min(h, w) * rng.uniform(0.5, 0.9))
        y, x = rng.integers(0, h - size), rng.integers(0, w - size)
        crop = base[y : y + size, x : x + size] * rng.uniform(0.7, 1.3, size=3)
        img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
        images.append(img.resize((384, 384)))
    return images


def bench_hf(args, images):
    import torch
    from transformers import AutoProcessor, GitForCausalLM

    processor = AutoProcessor.from_pretrained(MODEL)
    model = GitForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    cls_id = processor.tokenizer.cls_token_id

    def run(batch):
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values
        input_ids = torch.full((len(batch), 1), cls_id)
        with torch.inference_mode():
            model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=args.output_len,
                min_new_tokens=args.output_len,
                do_sample=False,
            )

    run(images[:1])  # warmup
    t0 = time.perf_counter()
    run(images[:1])
    latency = time.perf_counter() - t0

    bs = args.hf_batch_size
    t0 = time.perf_counter()
    for i in range(0, len(images), bs):
        run(images[i : i + bs])
    total = time.perf_counter() - t0
    return {"latency_s": latency, "total_s": total}


def bench_vllm(args, images, mm_prefix: bool):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="float32",
        max_model_len=512,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
        hf_overrides=None if mm_prefix else {"is_mm_prefix_lm": False},
    )
    params = SamplingParams(
        temperature=0.0, max_tokens=args.output_len, ignore_eos=True
    )

    def run(batch):
        reqs = [{"prompt": "", "multi_modal_data": {"image": im}} for im in batch]
        outs = llm.generate(reqs, params, use_tqdm=False)
        assert all(len(o.outputs[0].token_ids) == args.output_len for o in outs)

    run(make_images(4, seed=1))  # warmup (compile, encoder, batch shapes)
    t0 = time.perf_counter()
    run(images[:1])
    latency = time.perf_counter() - t0

    t0 = time.perf_counter()
    run(images)
    total = time.perf_counter() - t0
    return {"latency_s": latency, "total_s": total}


def run_one(args):
    images = make_images(args.num_requests)
    if args.run == "hf":
        result = bench_hf(args, images)
    else:
        result = bench_vllm(args, images, mm_prefix=args.run == "vllm")
    print("RESULT " + json.dumps({"config": args.run, **result}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--hf-batch-size", type=int, default=8)
    parser.add_argument("--configs", default="hf,vllm,vllm-no-mm-prefix")
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run:
        run_one(args)
        return

    env = {
        **os.environ,
        "VLLM_CPU_KVCACHE_SPACE": os.environ.get("VLLM_CPU_KVCACHE_SPACE", "1"),
    }
    results = []
    for config in args.configs.split(","):
        cmd = [sys.executable, __file__, "--run", config, *sys.argv[1:]]
        out = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
        line = next(
            (l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None
        )
        if line is None:
            sys.stderr.write(out.stdout[-3000:] + out.stderr[-3000:])
            raise SystemExit(f"{config} failed")
        results.append(json.loads(line[len("RESULT ") :]))

    n, t = args.num_requests, args.output_len
    print(f"\nGIT ({MODEL}), {n} requests x {t} output tokens, distinct images, fp32")
    print(f"HF batch size {args.hf_batch_size}; vLLM continuous batching\n")
    print("| config | 1-request latency [s] | total [s] | req/s | output tok/s |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['config']} | {r['latency_s']:.3f} | {r['total_s']:.2f} "
            f"| {n / r['total_s']:.2f} | {n * t / r['total_s']:.1f} |"
        )
    hf = next((r for r in results if r["config"] == "hf"), None)
    vl = next((r for r in results if r["config"] == "vllm"), None)
    if hf and vl:
        print(
            f"\nvLLM throughput speedup over HF: {hf['total_s'] / vl['total_s']:.2f}x"
        )


if __name__ == "__main__":
    main()
