# vllm-git

GIT (Generative Image-to-text Transformer, `GitForCausalLM`) を vLLM で動かすための作業リポジトリ。
モデル実装そのものは vLLM の fork ([Anieca/vllm@add-git-model](https://github.com/Anieca/vllm/tree/add-git-model), ローカルでは `./vllm`) に in-tree で入れてあり、
そのまま upstream に PR できる形にしている。このリポジトリには検証スクリプトと Docker/serving 周りだけを置く。

## vLLM 側の変更 (`./vllm`, branch `add-git-model`)

| ファイル | 内容 |
|---|---|
| `vllm/model_executor/models/git.py` | GIT 本体。CLIP vision tower (`CLIPVisionModel`, post-LN を全トークンに適用) + projection + BERT 系 post-LN デコーダ。HF の重みをそのまま読む (`WeightsMapper` で `query/key/value` → `qkv_proj` など) |
| `vllm/model_executor/models/registry.py` ほか | `GitForCausalLM` の登録、docs の対応表、tests の example model |
| `vllm/transformers_utils/model_arch_config_convertor.py` | `model_type == "git"` を mm prefix-LM (画像トークン同士は双方向) として扱う |
| `vllm/v1/attention/backends/cpu_attn.py` | CPU attention backend に `supports_mm_prefix` を実装 (下記) |
| `vllm/v1/attention/backends/utils.py` | mm_prefix の query span 計算を `compute_mm_prefix_query_spans` に切り出し (FA と CPU で共用) |
| `vllm/platforms/cpu.py` | CUDA/ROCm と同様に mm_prefix モデルでは `disable_chunked_mm_input` を強制 |
| `vllm/v1/core/sched/scheduler.py` | **既存バグ修正**: prefix cache hit が画像の途中で終わると、`disable_chunked_mm_input` でも残りの画像が分割されてスケジュールされていた (GPU の Gemma3/PaliGemma 等にも影響) |
| `tests/...` | CPU backend の mm_prefix テスト、スケジューラの回帰テスト、GIT の HF 比較テスト (`test_common.py`, `cpu_model` mark) |

### attention mask の扱い

GIT の mask は「画像トークン同士は双方向 / テキストは画像全体 + 過去のテキストに causal」で、
vLLM の **mm prefix-LM (`is_mm_prefix_lm`)** そのもの。モデル側で独自 mask は持たず、paged KV cache 上の
各 attention backend に任せている。

- GPU: FlashAttention / Triton / FlexAttention は upstream で mm_prefix 対応済み → GIT 側の追加作業なしで動く想定 (未検証)
- CPU: 今回対応。causal な paged attention kernel を通常通り走らせた後、画像区間に入る query 行だけを
  「1 span = 1 sequence, seq_len = 区間末尾 + 1, `causal=False`」の合成バッチとして同じ C++ paged kernel で再計算し、上書きする。
  KV cache を Python から読まないので ISA ごとの cache layout (NEON/AMX/...) や fp8 KV に依存しない。

### 位置埋め込み

HF の GIT はテキストの位置 ID を画像トークン抜きで 0 から数える (生成中も同じ; `scripts/probe_hf_positions.py` で確認)。
vLLM では画像トークンが常にプロンプト先頭 `num_image_tokens` 個 (= `(image_size/patch_size)^2 + 1`, git-base で 197) に
入るので、`position - num_image_tokens` をテキスト位置として使い、画像位置には position embedding / LayerNorm を掛けない。

制約: 1 プロンプトにつき画像 1 枚必須 (テキストのみのプロンプトは位置がずれる)。動画版 (`num_image_with_embedding`) は未対応。

### プロンプト

BERT トークナイザが付ける末尾の `[SEP]` (= EOS) は GIT の multimodal processor が落とす。
`prompt=""` でキャプション、`prompt="a photo of"` などで続きを生成。

## 検証結果 (Apple M3 / macOS / CPU, `microsoft/git-base-coco`, fp32)

`scripts/compare_hf_vllm.py`: 画像 3 枚 × プロンプト 2 種を HF Transformers と vLLM で greedy 生成して比較。

| 条件 | 結果 |
|---|---|
| eager | 6/6 完全一致, max \|Δlogprob\| 1.7e-5 |
| torch.compile (default) | 6/6 完全一致, max \|Δlogprob\| 1.7e-5 |
| `--max-num-batched-tokens 256` (prefill と decode の混在 + prefix cache hit) | 6/6 完全一致, max \|Δlogprob\| 2.1e-5 (スケジューラ修正前は 7e-3) |
| `--no-mm-prefix` (双方向 mask を切ったアブレーション) | 3/6 不一致, max \|Δlogprob\| 0.27 → mask が効いていることの確認 |

## ベンチマーク (Apple M3 8GB / CPU, fp32)

`scripts/bench_hf_vs_vllm.py`: リクエストごとに別画像 (prefix cache は効かない)、出力長固定 (`ignore_eos`)、
画像前処理込みで、HF `generate` (固定バッチ) と vLLM (paged attention + continuous batching) を比較。

| workload | config | 1 req latency [s] | total [s] | output tok/s |
|---|---|---:|---:|---:|
| 64 req × 32 tok | HF (batch 8) | 0.26 | 7.67–7.70 | 266 |
| | vLLM | 0.30–0.35 | 4.43–4.84 | 423–462 |
| | vLLM, mm_prefix 補正なし (参考) | 0.30 | 4.27–4.36 | 469–479 |
| 128 req × 64 tok | HF (batch 16) | 0.45 | 18.36 | 446 |
| | vLLM | 0.56 | 13.02 | 629 |
| | vLLM, mm_prefix 補正なし (参考) | 0.52 | 12.68 | 646 |

- スループットは vLLM が HF の **1.4–1.7 倍**。単発リクエストのレイテンシはエンジンのオーバーヘッドで vLLM の方がやや遅い。
- CPU の mm_prefix 補正 (画像行の再計算) のコストは全体の 3–10%。補正なしは出力が HF とずれるので参考値。
- 各 1–2 回の計測で、ばらつきは ±10% 程度。GPU では未計測。

vLLM のテスト:

```bash
cd vllm
pytest tests/kernels/attention/test_cpu_attn.py -k mm_prefix          # CPU backend vs dense reference
pytest tests/v1/core/test_scheduler.py -k mm_input_chunking           # スケジューラ回帰
pytest tests/models/multimodal/generation/test_common.py -k git       # HF 比較 (cpu_model)
```

## ローカル (macOS, ネイティブビルド)

```bash
git clone -b add-git-model https://github.com/Anieca/vllm.git vllm
uv venv -p 3.12 .venv && source .venv/bin/activate
cd vllm && uv pip install -r requirements/cpu.txt && uv pip install -e . && cd ..
VLLM_CPU_KVCACHE_SPACE=1 python scripts/compare_hf_vllm.py
```

8GB マシンでは `VLLM_CPU_KVCACHE_SPACE=1` (GiB) を指定しないと KV cache の確保で落ちる。
`vllm serve` は CPU だと `--dtype auto` で bf16 に落ちるので、HF と揃えたいときは `--dtype float32`。

```bash
python scripts/bench_hf_vs_vllm.py --num-requests 64 --output-len 32   # ベンチマーク
```

## Docker (CPU)

upstream の `docker/Dockerfile.cpu` を fork のチェックアウトからビルドする。

```bash
docker/build_cpu.sh            # → vllm-git:cpu
docker/run_cpu.sh compare      # HF vs vLLM 比較
docker/run_cpu.sh serve        # localhost:8000 で OpenAI 互換サーバ
python scripts/query_server.py --image stop_sign             # キャプション
python scripts/query_server.py --image stop_sign --prompt "a photo of"
```

### Docker での検証結果 (colima: linux/arm64, 4 CPU / 6GB, fork commit `f1766d5b1`)

| 条件 | 結果 |
|---|---|
| `run_cpu.sh compare` (torch.compile) | 6/6 完全一致, max \|Δlogprob\| 2.4e-5 |
| `run_cpu.sh compare --max-num-batched-tokens 256` (prefix cache hit + 混在バッチ) | 6/6 完全一致, max \|Δlogprob\| 1.9e-5 |
| `run_cpu.sh compare --eager` | 6/6 完全一致, max \|Δlogprob\| 2.1e-5 |
| `run_cpu.sh serve` (fp32) + `query_server.py` | stop_sign / cherry_blossom × プロンプト有無の 4 件で HF と同じキャプション |

8GB の Mac でビルドする場合は `MAX_JOBS=2` (4 だと oneDNN のコンパイル中に VM がメモリ不足で落ちた)。

GIT のトークナイザには chat template が無いので、`serving/git_chat_template.jinja` (`[CLS]` + 最後の user メッセージのテキスト) を渡している。

## TODO

- GPU (FlashAttention / Triton) での動作確認と `test_common.py -k git` の実行
- git-large / textvqa 系 checkpoint での確認
- upstream PR (スケジューラ修正は単独 PR に分けるのが良さそう)
