# vllm-git

GIT (Generative Image-to-text Transformer, `GitForCausalLM`) を vLLM で動かすための検証用リポジトリ。
モデル実装は vLLM の fork に PR として置き、ここには検証スクリプトと Docker/serving 周りだけを置く。

| PR (Anieca/vllm) | 内容 |
|---|---|
| [#1](https://github.com/Anieca/vllm/pull/1) | scheduler: prefix cache hit が画像の途中で終わると、`disable_chunked_mm_input` でも残りの画像が分割される問題の修正 |
| [#2](https://github.com/Anieca/vllm/pull/2) | CPU attention backend の mm prefix-LM (画像トークン同士が双方向) 対応 |
| [#3](https://github.com/Anieca/vllm/pull/3) | GIT モデル本体 (#2 の上に積んでいる)。mask の扱い・位置埋め込み・制約はこの PR の説明を参照 |

GIT の attention mask (画像は双方向、テキストは causal) は vLLM の `is_mm_prefix_lm` そのもので、モデル側に独自 mask はない。
GPU の各 backend は upstream で対応済みのはずだが、**GPU は未検証**。

## 検証結果 (Apple M3 / CPU, fp32, 画像 3 枚 × プロンプト 2 種を HF と greedy 比較)

`scripts/compare_hf_vllm.py`。いずれも 6/6 で生成が完全一致。

| 条件 | max \|Δlogprob\| |
|---|---|
| `microsoft/git-base-coco`: torch.compile / eager | 1.7e-5 |
| 同 `--max-num-batched-tokens 256` (prefill と decode の混在 + prefix cache hit) | 2.1e-5 (#1 なしだと 7e-3) |
| 同 `--max-num-seqs 1` (vLLM もバッチサイズ 1) | 2.1e-5 |
| `microsoft/git-base-vqav2` (480px, 画像 901 トークン) | 1.9e-5 |
| 同 `--no-mm-prefix` (双方向 mask を切った対照) | 0.27、3/6 が不一致 |

**logprob は HF と bit 一致しない** (38 値中 0)。バッチサイズ 1 でも同じ。浮動小数点の加算は結合則を満たさず、
HF と vLLM で総和の順序が違うため (fused QKV と別々の Linear、GEMM の形、ブロック単位の online softmax を使う paged attention kernel と
QKᵀ を一括で作る eager attention、GELU・CLIP 側の attention 実装など)。vLLM 自身もバッチの組み方で最大 2e-5 変わるので、
HF との差はこの丸め誤差の範囲内。`VLLM_BATCH_INVARIANT=1` は GPU 専用で CPU では使えず、保証するのも vLLM 内の一致で HF との一致ではない。

## ベンチマーク (Apple M3 8GB / CPU, fp32)

`scripts/bench_hf_vs_vllm.py`: リクエストごとに別画像 (prefix cache は効かない)、出力長固定、画像前処理込みで HF `generate` と vLLM を比較。

| workload | config | 1 req latency [s] | total [s] | output tok/s |
|---|---|---:|---:|---:|
| 64 req × 32 tok | HF (batch 8) | 0.26 | 7.67–7.70 | 266 |
| | vLLM | 0.30–0.35 | 4.43–4.84 | 423–462 |
| | vLLM, mm_prefix 補正なし (参考) | 0.30 | 4.27–4.36 | 469–479 |
| 128 req × 64 tok | HF (batch 16) | 0.45 | 18.36 | 446 |
| | vLLM | 0.56 | 13.02 | 629 |
| | vLLM, mm_prefix 補正なし (参考) | 0.52 | 12.68 | 646 |

スループットは vLLM が HF の 1.4–1.7 倍。単発のレイテンシはエンジンのオーバーヘッドで vLLM の方がやや遅い。
CPU の mm_prefix 補正のコストは 3–10%。各 1–2 回の計測でばらつきは ±10% 程度。補正なしは出力が HF とずれる参考値。

## 使い方

### ローカル (macOS)

```bash
git clone https://github.com/Anieca/vllm.git vllm
git -C vllm switch model/git && git -C vllm merge origin/fix/mm-item-split-after-prefix-hit   # #1 + #2 + #3
uv venv -p 3.12 .venv && source .venv/bin/activate
(cd vllm && uv pip install -r requirements/cpu.txt && uv pip install -e .)
VLLM_CPU_KVCACHE_SPACE=1 python scripts/compare_hf_vllm.py
python scripts/bench_hf_vs_vllm.py --num-requests 64 --output-len 32
```

8GB のマシンでは `VLLM_CPU_KVCACHE_SPACE=1` (GiB) が必要 (無いと KV cache の確保で落ちる)。
`vllm serve` は CPU だと `--dtype auto` で bf16 になるので、HF と揃えるなら `--dtype float32`。

### Docker (CPU)

上と同じ `./vllm` チェックアウトから upstream の `docker/Dockerfile.cpu` でビルドする。

```bash
MAX_JOBS=2 docker/build_cpu.sh   # → vllm-git:cpu (8GB ホストでは 2。4 だと oneDNN のコンパイル中に VM がメモリ不足で落ちた)
docker/run_cpu.sh compare        # HF vs vLLM 比較
docker/run_cpu.sh serve          # localhost:8000 で OpenAI 互換サーバ
python scripts/query_server.py --image stop_sign --prompt "a photo of"
```

GIT のトークナイザには chat template が無いので、`serving/git_chat_template.jinja` (`[CLS]` + 最後の user メッセージ) を渡している。

Docker での確認 (colima, linux/arm64, 4 CPU / 6GB): `compare` の 3 設定 (compile / eager / バッチ混在) がすべて 6/6 一致 (最大 2.4e-5)、
`serve` + `query_server.py` の 4 件が HF と同じキャプション。ただしこれは整理前の commit `f1766d5b1` で確認したもので、
その後のレビュー対応と整理はネイティブでのみ再確認している (イメージは未再ビルド)。

## TODO

- GPU (FlashAttention / Triton) での動作確認
- git-large 系 checkpoint での確認
