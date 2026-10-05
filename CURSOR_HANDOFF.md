# Handoff: download and test what the cloud sandbox could not

You are picking up a TurboQuant demo project. The code is done and partly tested. The sandbox it was built in could not reach Hugging Face, so **no real model or real dataset was ever downloaded**, and the LLM quality numbers were never measured. Your job is to download those assets, run the tests below, fix anything that breaks, and report the numbers.

> **Execution rule (agreed with the owner): the notebooks run only on Google Colab, never on this machine.** The local GPU is too small. Locally you may edit code, rebuild the notebooks (`python build_notebooks.py`) and run CPU-only checks such as T1. For T2 and T3, upload the notebook to Colab (T4 runtime for the LLM demo; CPU is fine for vector search), run it there, then download the executed notebook (File › Download › Download .ipynb) and every table it prints into `results/` in this folder. Results that stay only in Colab or Drive cannot be reviewed.

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

**Colab (T2, T3).** Nothing to install by hand: the first cells of each notebook install what they need (transformers v5, datasets, faiss-cpu, turbovec, scikit-learn). Models and datasets download inside the Colab runtime, not on this machine. `turbovec` needs an x86-64 CPU with AVX2, which Colab has; if its install fails anyway, skip its cells and note it.

**Local (editing and T1 only).** The project `.venv` already has CPU-only torch and numpy:

```bash
.venv/Scripts/python -c "import torch; print(torch.__version__)"   # Windows; .venv/bin/python under Linux/WSL
```

Do not download the Qwen models or the DBpedia vectors locally.

## 4. Data used by the notebooks (downloaded in Colab)

- Models: `Qwen/Qwen2.5-1.5B-Instruct` on the T4 (the notebook falls back to `Qwen2.5-0.5B-Instruct` only on CPU).
- Perplexity text: `wikitext`, `wikitext-2-raw-v1`, test split.
- Vector dataset: `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M`, streamed (101,000 rows).

The vector notebook auto-detects the embedding column (the first list with 256 or more floats) and prints its name. If that printout looks wrong, fix `load_dbpedia()` in `vector_search_demo.py`, rebuild with `python build_notebooks.py`, and upload again.

## 5. Tests to run

Save every output table as CSV under `results/` (create it) and keep the executed notebooks, downloaded from Colab, as `results/*_executed.ipynb`. In Colab, a table can be saved with `df.to_csv("name.csv")` and downloaded from the Files pane, or copied from the cell output.

### T1. Core sanity (about 1 minute, local CPU, already passed on 2026-10-05)

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

### T2. LLM notebook on Colab, T4 GPU (main task)

1. colab.research.google.com › File › Upload notebook › `llm_kv_cache_demo.ipynb`.
2. Runtime › Change runtime type › **T4 GPU**. Check the second code cell prints `device: cuda`; a CPU runtime silently switches to Qwen2.5-0.5B with short inputs, and those numbers are not the ones we want.
3. Runtime › Run all. The first cell upgrades transformers if Colab ships an older one; if it prints "Please restart the runtime", do Runtime › Restart session, then Run all again.
4. File › Download › Download .ipynb, save as `results/llm_kv_cache_demo_executed.ipynb`.

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

### T3. Vector search on Colab (CPU runtime is fine)

1. Upload `vector_search_demo.ipynb` to Colab; the default CPU runtime is enough (a GPU only speeds up the PyTorch scorer).
2. Runtime › Run all. Streaming 101,000 DBpedia rows takes a few minutes.
3. File › Download › Download .ipynb, save as `results/vector_search_demo_executed.ipynb`.

Record the full results table (method, bits, compression, build_s, QPS, R1@1/4/16/64, 10@10) and the re-rank line. Expect FAISS PQ 4-bit (LUT256) training to take several minutes at d = 1536; that is the point of the comparison. If it takes more than about 15 minutes, lower `TRAIN` to 10,000 rows and note it.

Sanity checks: TurboQuant and turbovec should beat PQ on R1@1 at equal bits (paper Figure 5), and indexing should take seconds against minutes for PQ. Report it plainly if they don't.

Optional: rerun with `DATASET = "dbpedia-3072"`.

### T4. Optional: vLLM production kernels (Ampere/Hopper GPU, vLLM 0.20.2 or later)

Only on a Colab runtime with an Ampere or newer GPU (L4 or A100, paid tiers), never locally; skip it if none is available. Run the commands below in a Colab terminal or `!` cells.

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
