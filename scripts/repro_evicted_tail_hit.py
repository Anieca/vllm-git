"""A prefix cache hit ends inside an image when the image's last blocks were
evicted (freed blocks are evicted tail first). Scheduler only, no model.

Usage (from this repo's root, with ./vllm checked out):
    python scripts/repro_evicted_tail_hit.py         # TRACE=1 prints lookups
main splits the 174 tokens left after the hit (128 + 46); with Anieca/vllm#1
they are scheduled in one step. num_blocks=10000 is the control without eviction.

Block size 16 (GPU default). Image: 576 tokens at offset 0 (36 full blocks).
step 1: primer (image, 590 tok) + spacer (text, 320 tok); both finish, so their
        blocks go to the free queue, tail blocks first.
step 2: filler (text, 896 tok) needs every never-used block plus 11 more, so it
        evicts the primer's last blocks. Then hit (same image) is looked up.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm"))

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.outputs import ModelRunnerOutput

BLOCK = 16


def make_output(s):
    return ModelRunnerOutput(
        req_ids=[r.request_id for r in s.running],
        req_id_to_index={r.request_id: i for i, r in enumerate(s.running)},
        # a request still in prefill samples nothing
        sampled_token_ids=[
            [1000] if r.num_computed_tokens >= r.num_prompt_tokens else []
            for r in s.running
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def run(num_blocks):
    s = create_scheduler(
        model="llava-hf/llava-1.5-7b-hf",
        max_num_batched_tokens=1024,
        disable_chunked_mm_input=True,
        enable_prefix_caching=True,
        block_size=BLOCK,
        num_blocks=num_blocks,
        max_model_len=2048,
    )

    def image(rid):
        return create_requests(
            1,
            num_tokens=590,
            mm_hashes_list=[["image"]],
            mm_positions=[[PlaceholderRange(offset=0, length=576)]],
            max_tokens=1,
            block_size=BLOCK,
            req_ids=[rid],
        )[0]

    def text(rid, n, idx):  # idx picks a distinct token id, so no prefix sharing
        ids = [f"_{k}" for k in range(idx)] + [rid]
        return create_requests(
            idx + 1, num_tokens=n, max_tokens=1, block_size=BLOCK, req_ids=ids
        )[idx]

    def step():
        out = s.schedule()
        hits = {r.req_id: r.num_computed_tokens for r in out.scheduled_new_reqs}
        s.update_from_output(out, make_output(s))
        return dict(out.num_scheduled_tokens), hits

    import os

    if os.environ.get("TRACE"):
        orig_enc = s._try_schedule_encoder_inputs

        def enc(request, num_computed_tokens, num_new_tokens, *a, **k):
            r = orig_enc(request, num_computed_tokens, num_new_tokens, *a, **k)
            print(
                f"    enc {request.request_id}: computed={num_computed_tokens} new={num_new_tokens} -> {r[1]}"
            )
            return r

        s._try_schedule_encoder_inputs = enc
        kvm = s.kv_cache_manager
        orig_gcb = kvm.get_computed_blocks

        def gcb(request, *a, **k):
            r = orig_gcb(request, *a, **k)
            print(
                f"    lookup {request.request_id}: hit={r[1]} free={kvm.block_pool.get_num_free_blocks()}"
            )
            return r

        kvm.get_computed_blocks = gcb
        orig_alloc = kvm.allocate_slots

        def alloc(request, num_new_tokens, *a, **k):
            r = orig_alloc(request, num_new_tokens, *a, **k)
            print(
                f"    alloc {request.request_id}: new={num_new_tokens} ok={r is not None} free_after={kvm.block_pool.get_num_free_blocks()}"
            )
            return r

        kvm.allocate_slots = alloc
    s.add_request(image("primer"))
    s.add_request(text("spacer", 320, 1))
    print("  step1", step())
    s.add_request(text("filler", 896, 2))
    s.add_request(image("hit"))
    hit_req = s.requests["hit"]
    print("  step2", step(), "| hit computed after step:", hit_req.num_computed_tokens)
    print("  step3", step())


for nb in (103,) if os.environ.get("TRACE") else (10000, 103):
    print(f"num_blocks={nb}")
    run(nb)
