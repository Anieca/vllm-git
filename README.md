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

## vLLM に既にあるモデルでの再現 (Gemma3)

PR1 と PR2 は GIT なしで確認できる。`hf-internal-testing/tiny-random-Gemma3ForConditionalGeneration` は 1 層で、
画像トークンの KV が他のトークンに依存しないためバグが数値に出ない。同じ設定で層を増やし、重みを大きくして作る
(`scripts/build_tiny_gemma3.py`、200 万パラメータ、8.7MB)。

| 確認 | スクリプト | 条件 | 結果 |
|---|---|---|---|
| PR2: HF との一致 | `tiny_gemma3_parity.py DIR` | 3 層、画像 2 枚、top-5 logprob | 最大差 4.3e-6。`--no-mm-prefix` では 1.45e-1 |
| PR1: 途中ヒットの分割 | `repro_mid_item_split.py DIR` | 6 層・std 2。単独実行との logprob 最大差 | PR1 あり 1.5e-5、main 3.5e-3 |

`repro_mid_item_split.py` は単一プロセスで走らせる (既定)。マルチプロセスだと、画像の前処理に時間がかかるリクエストが
テキストだけのリクエストより遅れてエンジンに届き、同じステップに乗らないため、競合が起きず差が出ない。
`TRACE_SCHEDULE=1` で、main のとき `[473, 39]` → `[1, 73]` と分割され、PR1 では `[473]` → `[1, 112]` になるのを確認できる。
PR2 の CPU 対応が前提 (main の CPU は `mm_prefix` を無視するので、分割しても数値に出ない)。

## 追い出しで画像の途中にヒットが止まる場合 (PR1)

`scripts/repro_evicted_tail_hit.py` (スケジューラ単体、モデル不要)。解放されたブロックは末尾から先に追い出されるので、
KV キャッシュが足りないと、キャッシュ済みの画像は後ろから欠ける。ブロック 16 で、576 トークンの画像の末尾 10 ブロックが
追い出されると、同じ画像のリクエストのヒットは 416 で止まり、残り 174 トークン (画像 160) を main は 128 + 46 に分割する。
PR1 では一度に処理する。追い出しがなければヒットは画像の終わり (576) まで届き、問題は起きない。

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
`serve` + `query_server.py` の 4 件が HF と同じキャプション。ただしこれは PR のレビュー対応・整理より前のコードで確認したもので、
その後はネイティブでのみ再確認している (イメージは未再ビルド)。

## TODO

- GPU (FlashAttention / Triton) での動作確認
- git-large 系 checkpoint での確認
