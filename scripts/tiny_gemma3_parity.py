"""First-token logprobs of a small Gemma3: HF Transformers vs vLLM (CPU).

Usage: python scripts/tiny_gemma3_parity.py MODEL_DIR [--no-mm-prefix]
`--no-mm-prefix` turns the bidirectional image mask off in vLLM as a control.
"""

import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from vllm.assets.image import ImageAsset

from vllm import LLM, SamplingParams

PROMPT = (
    "<bos><start_of_turn>user\n<start_of_image>What is in the image?"
    "<end_of_turn>\n<start_of_turn>model\n"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--no-mm-prefix", action="store_true")
    args = parser.parse_args()

    images = [
        ImageAsset(n).pil_image.convert("RGB") for n in ("stop_sign", "cherry_blossom")
    ]

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32
    ).eval()
    hf_logprobs = []
    for image in images:
        inputs = processor(text=PROMPT, images=image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        hf_logprobs.append(torch.log_softmax(logits, dim=-1))

    llm = LLM(
        model=args.model,
        dtype="float32",
        max_model_len=2048,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        hf_overrides={"is_mm_prefix_lm": False} if args.no_mm_prefix else None,
    )
    outs = llm.generate(
        [{"prompt": PROMPT, "multi_modal_data": {"image": im}} for im in images],
        SamplingParams(temperature=0.0, max_tokens=1, logprobs=5),
    )
    worst = 0.0
    for out, hf in zip(outs, hf_logprobs):
        completion = out.outputs[0]
        diffs = [
            abs(lp.logprob - hf[tok].item())
            for tok, lp in completion.logprobs[0].items()
        ]
        worst = max(worst, max(diffs))
        print(
            f"prompt tokens {len(out.prompt_token_ids)}, top-5 max |logprob diff| {max(diffs):.2e}"
        )
    print(f"RESULT mm_prefix={'off' if args.no_mm_prefix else 'on'} worst {worst:.2e}")


if __name__ == "__main__":
    main()
