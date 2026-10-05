# Handoff: download and test what the cloud sandbox could not

You are picking up a TurboQuant demo project. The code is done and partly tested. The sandbox it was built in could not reach Hugging Face, so **no real model or real dataset was ever downloaded**, and the LLM quality numbers were never measured. Your job is to download those assets, run the tests below, fix anything that breaks, and report the numbers.

## 1. What is in this folder

| File | Purpose |
|---|---|
| `turboquant_core.py` | Pure-PyTorch TurboQuant: Lloyd-Max codebooks, `TurboQuant` (mse / prod), `MixedTurboQuant` (outlier split), `UniformQuant` (INT-b baseline), bit packing, `TurboQuantCache` (Hugging Face transformers v5 cache). |
| `llm_kv_cache_demo.ipynb` / `.py` | KV-cache demo: theory check, perplexity/KL/top-1 vs uncompressed, needle in a haystack, memory, decode speed. |
| `vector_search_demo.ipynb` / `.py` | Vector search: TurboQuant (PyTorch) and `turbovec` vs FAISS PQ, PQ-FastScan, RaBitQ, SQ4. |
| `README.md` | Results measured so far. |

The notebooks embed `turboquant_core.py` in a `%%writefile` cell. The `.py` versions import it from this folder. If you change `turboquant_core.py`, re-sync the notebook cell (or regenerate the notebooks from the `.py` files with `jupytext`).

## 2. What was and was not verified

Verified in the sandbox (CPU, 4 threads):
- MSE distortion matches the paper (0.360 / 0.116 / 0.034 / 0.0093 at 1-4 bits, d = 128); the prod variant is unbiased.
- `TurboQuantCache` works with transformers **5.18.0** on a random-weight Qwen2 model (chunked prefill and `generate`).
- Both notebooks execute end to end with `nbclient` (LLM one with a mocked model and tokenizer).
- Vector search on local real-text embeddings (384-d TF-IDF + SVD, 100k vectors). Results are in `README.md`.

**Not** verified (your tasks):
1. Real model `Qwen/Qwen2.5-1.5B-Instruct` (and 0.5B): quality, needle, memory, speed.
2. `wikitext-2-raw-v1` loading (the perplexity text).
3. The paper's dataset `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`: column name, streaming, results at d = 1536.
4. Behaviour on a real GPU (T4 or better), including peak memory.
5. Optional: vLLM's production TurboQuant kernels.

## 3. Environment

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10-3.12
pip install -U pip
pip install torch                                   # CUDA build if you have a GPU
pip install "transformers>=5.0,<6" accelerate datasets huggingface_hub \
            faiss-cpu turbovec scikit-learn matplotlib pandas jupyter nbclient jupytext
python -c "import torch, transformers, faiss, turbovec; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
```

`turbovec` needs an x86-64 CPU with AVX2 (or ARM NEON). If `pip install turbovec` fails, skip its cells and note it.

## 4. Downloads

```bash
# Models (about 3 GB and 1 GB)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# Perplexity text
python -c "from datasets import load_dataset; d=load_dataset('wikitext','wikitext-2-raw-v1',split='test'); print(len(d))"

# Paper's vector dataset: check the column names first
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M", split="train", streaming=True)
row = next(iter(ds))
print({k: (type(v).__name__, len(v) if isinstance(v, list) else v) for k, v in row.items()})
EOF
```

The vector notebook auto-detects the embedding column (the first list with 256 or more floats). If the printout shows otherwise, fix `load_dbpedia()` in `vector_search_demo.py/.ipynb`.

Optional: cache 101,000 rows to `dbpedia1536_101k.npy` once, then set `DATASET = "dbpedia1536_101k.npy"` (the notebook accepts a local `.npy` path) so reruns skip the download.

## 5. Tests to run

Save every output table as CSV under `results/` (create it) and keep executed notebooks as `results/*_executed.ipynb`.

### T1. Core sanity (about 1 minute, CPU)

```bash
python - <<'EOF'
import torch, math
from turboquant_core import *
for b in [1,2,3,4,5]:
    x = torch.randint(0, 2**b, (3, 5, 64)); assert torch.equal(unpack_bits(pack_bits(x, b), b, 64), x)
x = torch.randn(20000, 128); x /= x.norm(dim=1, keepdim=True)
for b, ref in zip([1,2,3,4], [0.36, 0.117, 0.034, 0.009]):
    m = ((x - TurboQuant(128, b, seed=1).roundtrip(x))**2).sum(1).mean().item()
    assert abs(m - ref) / ref < 0.1, (b, m); print(b, round(m, 4))
print("T1 OK")
EOF
```

### T2. LLM notebook on a GPU (main task)

```bash
jupyter nbconvert --to notebook --execute llm_kv_cache_demo.ipynb \
  --output results/llm_kv_cache_demo_executed.ipynb --ExecutePreprocessor.timeout=3600
```

On CPU only, it switches to Qwen2.5-0.5B, 512 tokens and short haystacks automatically. That is fine as a smoke test, but report GPU numbers if you can.

The first cell upgrades transformers if it is older than v5. If that happens inside a running kernel, restart and run again.

Record:
- **Quality table** (section 3): `bits_per_channel`, `compression`, `perplexity`, `KL_vs_baseline`, `top1_agree` for all 10 configs.
- **Needle table** (section 4): found rate per config and context length.
- **Memory table** (section 5): `kv_cache_MB`, `peak_gpu_MB`.
- **Decode tok/s** (section 6).

Sanity checks. Flag anything that fails:
- The uncompressed baseline finds the needle at every length and depth. If not, the prompt or haystack construction is the problem, not quantization.
- `bits_per_channel` is about 4.125 (TQ 4-bit), 3.75 (3.5 split), 3.125, 2.5 (2.5 split), 2.125, 5.0 (INT4), 4.0 (INT3), 3.0 (INT2).
- Perplexity and KL should get worse as bits go down. TurboQuant 4-bit should be close to baseline. Compare TurboQuant at about 3 bits with INT2 (3.0 effective) and INT3 (4.0 effective) to see the scale-overhead story.
- TurboQuant decode speed is expected to be slower than baseline (reference dequantization in PyTorch). That is documented, not a bug.

If something fails:
- `TurboQuantCache` relies on the transformers v5 cache API (`Cache(layers=...)`, `DynamicLayer`). If a newer v5 release breaks it, adapt `TurboQuantLayer` in `turboquant_core.py` (override `update`, `get_seq_length`, `get_mask_sizes` as needed) and re-sync the notebook cell. Beam search and `crop` are not supported; keep `do_sample=False`, `num_beams=1`.
- Out of memory at 8k tokens on a T4: lower `LENGTHS` to `[2000, 4000, 6000]`.
- Qwen2.5 has strong key outliers. If 2-bit quality is catastrophic, try `residual_length=32` in `make_cache` as an extra row, and report both.

### T3. Vector search on the paper's dataset (CPU is fine)

```bash
jupyter nbconvert --to notebook --execute vector_search_demo.ipynb \
  --output results/vector_search_demo_executed.ipynb --ExecutePreprocessor.timeout=7200
```

Record the full results table (method, bits, compression, build_s, QPS, R1@1/4/16/64, 10@10) and the re-rank line. Expect FAISS PQ 4-bit (LUT256) training to take several minutes at d = 1536; that is the point of the comparison. If it takes more than about 15 minutes, lower `TRAIN` to 10,000 rows and note it.

Sanity checks: TurboQuant and turbovec should beat PQ on R1@1 at equal bits (paper Figure 5), and indexing should take seconds against minutes for PQ. Report it plainly if they don't.

Optional: rerun with `DATASET = "dbpedia-3072"`.

### T4. Optional: vLLM production kernels (Ampere/Hopper GPU, vLLM 0.20.2 or later)

```bash
pip install -U vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --kv-cache-dtype turboquant_4bit_nc
# compare with the default dtype: max KV tokens reported at startup, plus throughput with vllm's benchmark_serving
```

Record the KV capacity (tokens) and throughput for `auto`, `fp8`, `turboquant_4bit_nc` and `turboquant_k3v4_nc`.

## 6. What to hand back

1. `results/` with the CSVs and executed notebooks.
2. A short `results/SUMMARY.md` covering the four LLM tables, the vector table, the environment (GPU model, CPU, library versions), and any code changes with the reason for each.
3. Update `README.md` with the real numbers. Keep the sandbox numbers and label them as sandbox results.
4. Don't change measured numbers by hand, and don't drop configs that look bad. Bad results are results.

These slides in the TurboQuant deck (a separate artifact) should get the real numbers afterwards: **"Demo 1 · Memory"** (add quality and needle results) and **"Demo 2 · Vector search, measured"** (swap in the DBpedia-1536 table).
