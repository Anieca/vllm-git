"""Greedy-decode GIT with HF Transformers and with vLLM, and compare.

Usage:
    python scripts/compare_hf_vllm.py [--model microsoft/git-base-coco] [--eager]
        [--max-num-batched-tokens N] [--max-num-seqs N] [--no-mm-prefix]
        [--no-prefix-caching] [--dump-vllm out.json]

`--max-num-seqs 1` makes vLLM run one request at a time (batch size 1, like the
HF side). `--dump-vllm` saves vLLM's token ids and logprobs, e.g. to check that
vLLM gives the same numbers with and without batching.

`--no-mm-prefix` disables the bidirectional image mask (plain causal attention)
as an ablation: the outputs are then expected to diverge from HF.
"""

import argparse
import json
import math

import torch
from PIL import Image
from transformers import AutoProcessor, GitForCausalLM
from vllm.assets.image import ImageAsset

from vllm import LLM, SamplingParams

MAX_TOKENS = 20
TOP_LOGPROBS = 5


def load_images() -> dict[str, Image.Image]:
    images = {
        name: ImageAsset(name).pil_image.convert("RGB")
        for name in ("stop_sign", "cherry_blossom")
    }
    images["red_square"] = Image.new("RGB", (320, 240), (200, 30, 30))
    return images


def run_hf(model_id, cases):
    processor = AutoProcessor.from_pretrained(model_id)
    model = GitForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
    tokenizer = processor.tokenizer
    results = []
    for image, prompt in cases:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        # Same prompt convention as the vLLM processor: [CLS] + text, no [SEP].
        input_ids = [tokenizer.cls_token_id] + tokenizer(
            prompt, add_special_tokens=False
        ).input_ids
        with torch.no_grad():
            out = model.generate(
                pixel_values=pixel_values,
                input_ids=torch.tensor([input_ids]),
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        new_ids = out.sequences[0, len(input_ids) :].tolist()
        logprobs = [
            torch.log_softmax(score[0].float(), dim=-1)[tok].item()
            for score, tok in zip(out.scores, new_ids)
        ]
        results.append((new_ids, logprobs))
    return results


def run_vllm(
    model_id,
    cases,
    eager,
    max_num_batched_tokens,
    max_num_seqs,
    mm_prefix,
    prefix_caching,
):
    llm = LLM(
        model=model_id,
        dtype="float32",
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        enforce_eager=eager,
        limit_mm_per_prompt={"image": 1},
        hf_overrides=None if mm_prefix else {"is_mm_prefix_lm": False},
        enable_prefix_caching=prefix_caching,
    )
    params = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, logprobs=TOP_LOGPROBS
    )
    outputs = llm.generate(
        [
            {"prompt": prompt, "multi_modal_data": {"image": image}}
            for image, prompt in cases
        ],
        params,
    )
    results = []
    for out in outputs:
        completion = out.outputs[0]
        ids = list(completion.token_ids)
        logprobs = [lp[tok].logprob for lp, tok in zip(completion.logprobs, ids)]
        results.append((ids, logprobs, completion.logprobs, len(out.prompt_token_ids)))
    return results, llm.get_tokenizer()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/git-base-coco")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--no-mm-prefix", action="store_true")
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--dump-vllm", help="save vLLM token ids / logprobs as JSON")
    args = parser.parse_args()

    images = load_images()
    prompts = ["", "a photo of"]
    cases = [(img, p) for img in images.values() for p in prompts]
    labels = [f"{name!r} + prompt {p!r}" for name in images for p in prompts]

    hf = run_hf(args.model, cases)
    vl, tokenizer = run_vllm(
        args.model,
        cases,
        args.eager,
        args.max_num_batched_tokens,
        args.max_num_seqs,
        not args.no_mm_prefix,
        not args.no_prefix_caching,
    )

    if args.dump_vllm:
        with open(args.dump_vllm, "w") as f:
            json.dump(
                {
                    label: {"ids": ids, "logprobs": lps}
                    for label, (ids, lps, *_) in zip(labels, vl)
                },
                f,
            )

    n_match = 0
    max_diff = 0.0
    all_diffs = []
    for label, (hf_ids, hf_lps), (vl_ids, vl_lps, vl_top, n_prompt) in zip(
        labels, hf, vl
    ):
        # Stop at the first divergence; after it the two runs see different
        # prefixes. A divergence only counts as a mismatch if HF's token is
        # not in vLLM's top-k there (i.e. it is not a near tie).
        n = 0
        while n < min(len(hf_ids), len(vl_ids)) and hf_ids[n] == vl_ids[n]:
            n += 1
        diffs = [abs(a - b) for a, b in zip(hf_lps[:n], vl_lps[:n])]
        max_diff = max([max_diff, *diffs])
        all_diffs += diffs
        exact = hf_ids == vl_ids
        near_tie = (
            not exact and n < len(hf_ids) and n < len(vl_top) and hf_ids[n] in vl_top[n]
        )
        n_match += exact
        status = "MATCH" if exact else ("NEAR-TIE" if near_tie else "MISMATCH")
        print(f"[{status}] {label}  (vLLM prompt len {n_prompt})")
        print(f"   HF  : {tokenizer.decode(hf_ids, skip_special_tokens=True)!r}")
        print(f"   vLLM: {tokenizer.decode(vl_ids, skip_special_tokens=True)!r}")
        if diffs:
            print(f"   max |logprob diff| over {n} common tokens: {max(diffs):.2e}")

    print(f"\n{n_match}/{len(cases)} exact matches, max |logprob diff| {max_diff:.2e}")
    n_same = sum(d == 0.0 for d in all_diffs)
    mean = sum(all_diffs) / max(len(all_diffs), 1)
    print(
        f"bit-identical logprobs: {n_same}/{len(all_diffs)}, mean |logprob diff| {mean:.2e}"
    )
    if not math.isfinite(max_diff) or n_match != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
