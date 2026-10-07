# Handoff: document every test behind the case study

You are picking up a finished study, so there is nothing to measure. Two benchmarks checked whether vector
compression (TurboQuant, and the built-in int8 / binary / BBQ options) could help two "find similar" features
of a legal-documents platform. Both verdicts are **no**. The workshop chapter that tells that story
(`docs/guide/Case_Study_Provision_Similarity_EN.md` and its Spanish twin) shows a selection of the results.
**Your job is to document all the tests that were run in both use cases**, so a reader can see every
instrument, every table and every limit, not only the headline numbers.

The results already exist. Copy numbers from the result files listed in §3. Do not recompute them, round
them differently or "improve" them.

---

## 1. Rules (read first)

1. **No client data in git.** The result folders hold aggregates. Next to them, in the same folders, are
   embedding caches (`*.npz`) and the source corpora, which are client documents. Never copy a `.npz`, a CSV,
   a provision or memo text, or any fund, investor, firm or law-firm name into this repo. Use tables of
   numbers only. Refer to data by its role ("2,501 provision texts from extracted side letters", "1,238
   comment-memo comments"). `.gitignore` already blocks `es_bench/results/` and `*.npz`. Keep it that way.
2. **Read-only outside this repo.** The result files live in another repository's working tree (§3). Read
   them and do not edit, move or delete anything there.
3. **No commits, no pushes.** Leave your changes uncommitted. The owner reviews and commits.
4. **EN and ES stay in step.** Every section added to an English doc gets the same section in the Spanish
   doc, in natural Spanish (the existing `Caso_Estudio_Similitud_Provisiones_ES.md` shows the register).
   Table numbers are identical in both.
5. **Say what each number is.** Every provision result comes from a **local stand-in** of the platform's
   index, not from the dev environment, which was unreachable. Label it as such wherever it appears. Do not
   present stand-in percentages as platform-wide facts.
6. **No re-runs are needed.** If you choose to re-run anything, run `es_bench/canary.py` first (it must
   print `ALL PASS`). Write into a folder outside git, and say in the doc which run each table comes from.
   Never point `run_es_bench.py --es` at a non-local cluster without the owner's say-so.
7. **Do not change the benchmark code** (`es_bench/*.py`, `memo_bench/*.py`, `provision_bench.py`,
   `turboquant_core.py`). This is a documentation task. If you find a code bug, note it in §6 of your
   report instead.
8. Style: plain language, short sentences, no marketing tone. Follow the existing guides.

---

## 2. The two use cases in one paragraph each

**A. Similar provisions.** The product shows "provisions similar to this one, above X %" across a firm's
provision database. The score of record is **`rapidfuzz.fuzz.ratio`** (edit distance, 0–100) on the
cleaned text. It is stored in a precomputed pair matrix when it is ≥ 30, and the UI defaults to 70.
Embeddings (`text-embedding-3-small`, 1,536-d) exist in the Elasticsearch index but no feature reads them.
The question was whether cosine neighbours, compressed or not, or trigram candidates could choose which
pairs to score instead of scoring all N². Code: `es_bench/`.

**B. Comment-memo response suggestions.** While a user answers a comment, the platform suggests past
responses to similar questions. It uses `text-embedding-3-large` (3,072-d) in Qdrant, with **exact**
search filtered by firm, the own thread excluded, limit 50, a score floor of 0.5 and a retry at 0.42. Then
a re-ranker and an LLM relevance check run. The question was whether compression helps latency, memory or
the suggestions themselves. Code: `memo_bench/`.

A premise correction comes before both. The repo's first benchmark (`provision_search_benchmark.*`,
`provision_bench.py`, `PROVISION_SEARCH_CHECKS.md`) assumed a Qdrant `provisions` collection with a cosine
auto-merge threshold of 0.90. Reading the platform code showed that neither exists. Qdrant holds comment
memos only, and auto-merge also uses `fuzz.ratio`. That is why `es_bench/` exists. The first benchmark files
are still in the repo and nothing marks them as built on a wrong premise (task T5).

---

## 3. Where the results are

Windows path prefix `D:\ILS\document-parser-lambda\data\changes\monolith-provision-similarity-perf\`
(WSL: `/mnt/d/ILS/document-parser-lambda/data/changes/monolith-provision-similarity-perf/`).

| File | Use case | What it holds |
|---|---|---|
| `es_bench_data/results_local_2026-10-06/REPORT.md` | A | Generated report, sections 1–8 (all tables below) |
| `es_bench_data/results_local_2026-10-06/4_fuzz_neighbours_by_floor.csv` | A | K and density d per floor 30…80 |
| `…/5_candidate_recall.csv` | A | 240 rows: every candidate generator × k × floor (true pairs, found, pair recall, per-query full recall). **REPORT.md shows only 3 of the floors**: the CSV has more |
| `…/6_compression_vs_exact_cosine.csv` | A | 13 rows: method × rescore setting, MB, QPS, recall@10, top-1 checks |
| `…/7_live_knn.csv` | A | Live kNN on the index: p50/p95, overlap with exact |
| `…/8_es_index_types.csv` | A | `hnsw` / `int8_hnsw` / `int4_hnsw` / `bbq_hnsw` / `flat`, ± rescore |
| `_es_bench_local_results_2026-10-06.md` | A | Human write-up: findings 1–5, limits, next step |
| `memo_bench_data/REPORT.md` | B | Generated report, sections 0–6 |
| `memo_bench_data/3_compression.csv`, `5_scale.csv` | B | Compression flips; scale runs |
| `memo_bench_data/run.log` | B | Run log (a Qdrant client/server version warning, harmless) |
| `_qdrant_usage_2026-10-06.md` | A + B | Where vectors live (Qdrant = memos only; ES embeddings write-only), §6 memo verdict |
| `_state_2026-10-06.md` | A | The product context: store floor 30 is also the UI/API minimum; the matrix design options |
| `README.md` (same folder) | both | Index of the folder and the closing summary |

Do not open the `*.npz` files. You do not need them.

---

## 4. Inventory of every test

"Shown?" = whether `Case_Study_Provision_Similarity_EN.md` already shows it. ✗ and ◐ items are the gaps.

### A. `es_bench` — provisions (stand-in: 2,501 distinct provisions, one firm id; 500 query provisions)

| # | Test | Source | Shown? |
|---|---|---|---|
| A0 | **Canary**, known-answer checks: C1 fuzz truth = a plain `fuzz.ratio` loop; C2 trigram = real Postgres `pg_trgm similarity()` (to ~5e-9); C3 / C3b candidate recall on a hand case and a full case; C4 the platform's text cleaning; C5 ES `flat` kNN = exact top-10; C5b ES `hnsw` ≥ 0.95 of exact; C6 sample loader returns exactly the expected rows; C6b vectors round-trip; C6c live kNN with firm filter = exact top-5; C6d index info reads dims and similarity; C7 refuses to write to a non-local cluster | `es_bench/canary.py` | ◐ Only "13 checks" and two examples. ⚠️ **The count is 12**, not 13. Fix it in EN, ES and `es_bench/README.md` |
| A0b | Bugs the canary caught in the benchmark itself: the trigram self-pair (a text compared with itself inflated recall, so self-pairs are now excluded); live kNN missing the population filter and over-fetch; the canary's own index needing `flat` because ES 8.18 defaults to `int8_hnsw`. The doc says "two bugs". Check the code comments and decide whether two or three is right, and name them | `es_provision_bench.py`, `canary.py` | ◐ |
| A1 | Index facts: ES 8.18.0, docs, store MB, effective `index_options` = `int8_hnsw` m16 ef100 although the mapping does not set it | REPORT §1 | ◐ the int8 default is mentioned |
| A2 | Sample: 2,501 rows loaded in 5.7 s via PIT; 500 queries | REPORT §2 | ✗ |
| A3 | **Embedding health**: norms (already normalised), random-pair cosine mean 0.570 / p95 0.706, mean-direction norm 0.755, effective dim 90 % = 254, nn1 median 0.956, near-duplicate queries 14.6 %, hub share 0.059 | REPORT §3 | ✗ Worth a short explainer: what each health metric means and why 3-small provision vectors sit in a narrow cone |
| A4 | **fuzz K / density by floor** 30, 40, 50, 60, 70, 80 (avg neighbours K and d %) | REPORT §4, CSV 4 | ◐ The doc shows d at 30–70 only, not K and not 80 |
| A5 | **Candidate recall**: cosine top-k for 5 storage methods × k ∈ {10, 25, 50, 100, 200, 500} × floors; trigram `%` cutoffs 0.1–0.4 (with the average number returned); trigram top-k (`<->`) | REPORT §5, CSV 5 | ◐ The doc shows float32 / TQ 2-bit / trigram at floor 70, k ≤ 200. Missing: floors 30 and 50, k = 25 and 500, int8, binary, TQ 4-bit, the trigram `%` cutoffs (0.3 → 37 % of the firm; 0.4 → 100 % at ≥ 70 with ~4 % of the firm), per-query full recall |
| A6 | **Compression vs exact cosine** (the original notebook's section 4): MB, compression ratio, build s, QPS, recall@10, top-1 found in top-1 / top-10, without rescore and with 2× / 4× rescore | REPORT §6, CSV 6 | ◐ The doc shows recall@10 only. Missing: QPS, build time, the 4× rescore, MB. ⚠️ `top1_found_in_top10` is 1.0 for every row, which is **not** "same top-10". An earlier draft made that mistake. Explain the column correctly |
| A7 | **Live kNN** on the index, num_candidates 100 / 500: p50 ~21 ms, overlap 0.983 / 0.989 | REPORT §7, CSV 7 | ✗ |
| A8 | **ES built-in quantization** on a local throwaway cluster: build s, store MB, p50, overlap, fuzz ≥ 70 and ≥ 30 recall @200 for 7 configurations | REPORT §8, CSV 8 | ◐ Overlap only. Missing: store MB (e.g. bbq 42.7 vs hnsw 70.6), build time, p50, the recall columns |
| A9 | Limits of the run: many sources mixed under one firm id; a cluster of 160 near-identical provisions dominates the ≥ 70 pairs; median neighbours at ≥ 70 is 1 and 27 % have none; trigram computed in Python with pg_trgm semantics (shows recall, not GIN-index speed); no 500k timing | `_es_bench_local_results_2026-10-06.md` "Limits" | ✗ A study that hides its limits misleads. Add them |
| A10 | Scale arithmetic: 500k firm → ~87 B pairs at floor 30, ~2 B at 50 | write-up finding 1 | ✓ |

### B. `memo_bench` — comment memos (1,238 unique comments: 685 questions, 553 responses, 15 memo documents, 326 threads; queries = the questions)

| # | Test | Source | Shown? |
|---|---|---|---|
| B0 | **Canary**: Qdrant exact search (firm filter, own thread excluded, floor 0.5) = numpy exact search on 200 / 200 queries (ids, order, scores to 1e-4). The script exits non-zero if it fails | REPORT §0 | ◐ one sentence |
| B1 | Corpus facts | REPORT §1 | ◐ |
| B2 | **Scores around the floors**: top-1 median 0.723, p10 0.611, p90 0.935; 98.8 % of queries have any result ≥ 0.5; 1.2 % need the 0.42 retry; **152.4 results ≥ 0.5 per query on average**; 5.8 % of top-50 scores within 0.01 of the floor | REPORT §2 | ◐ only the 152 / 0.38 finding |
| B3 | **Embedding health** (3-large): random-pair cosine mean 0.381 / p95 0.564, effective dim 263, nn1 median 0.756, near-duplicates 1.6 %, hub share 0.048 | REPORT §2 "Embedding health" | ✗ Contrast it with A3: the provision vectors are far more crowded (0.57 vs 0.38) |
| B4 | **Compression quality**: bytes per vector, top-10 overlap, suggestions added / dropped at 0.5 **and at 0.42** for int8, binary, TQ 4-bit, TQ 2-bit; Qdrant scalar and binary with rescore (0 / 0 of 104,379) | REPORT §3, CSV 3 | ◐ The 0.42 column and binary are missing. Explain why binary has no flip count (`nan`): find the reason in `memo_bench.py` and state it, do not guess |
| B5 | **Request latency, step by step**: embedding 239 / 474 ms (p50 / p95), `collection_exists` 3.2 / 3.5 ms (called before every search), exact search 13.9 / 15.0 ms | REPORT §4 | ◐ The `collection_exists` round trip and the p95s are missing |
| B6 | **Scale**: 10k / 50k / 200k points in one firm, float32 vs Qdrant scalar: build s, p50 / p95, vector RAM; TurboQuant 4-bit RAM projection at 100k / 1M | REPORT §5, CSV 5 | ◐ p95, build time and the projection rows are missing |
| B7 | **LLM relevance-check-sized call** (12 candidates, reasoning low): p50 3,798 ms, max 5,045 ms. ⚠️ An approximation of the platform's gate, not the gate itself | REPORT §6 | ◐ Say that it is an approximation |
| B8 | Limits: the platform's query-side legal-term expansion is not reproduced; text normalisation = visible text, lower-cased; local Qdrant (no network); 1.17.1 server with a 1.19 client | `memo_bench/README.md`, `run.log` | ✗ |

### C. Premise checks (both use cases)

| # | Check | Source | Shown? |
|---|---|---|---|
| C1 | Qdrant holds only the `comment_memo` collection; no branch puts provisions in Qdrant; provision embeddings are ES-only and unread since a semantic-search mode was removed | `_qdrant_usage_2026-10-06.md` §1–3 | ✓ (§2 of the case study) |
| C2 | The store floor of 30 is also the UI minimum and the API's accepted range (30–100), so raising it is a product change | `_state_2026-10-06.md` §3 | ✗ It matters to the verdict ("the lever is the floor"). Add one paragraph |
| C3 | "%" in compare means fuzz from the matrix when the text matches exactly, live trigram otherwise | `_state_2026-10-06.md` §3 | ✗ It explains why trigram as a candidate filter adds no new semantics |
| C4 | Still open, not run: the dev-environment re-run (blocked on cluster access) and a live listing of Qdrant collections | `_state_…` §5, `_qdrant_usage_…` §5 | ✗ List it as "what would change the numbers, not the verdict" |

---

## 5. Tasks

**T1. Test catalogue (main deliverable).** Create `docs/guide/Case_Study_Test_Catalogue_EN.md` and
`docs/guide/Catalogo_Pruebas_Caso_Estudio_ES.md`. Give one section per use case. For every row in §4
(A0–A10, B0–B8, C1–C4), give: what the test asks, how it is run (script + section), the full result table
copied from the source, and one or two sentences on what it shows and what it cannot show. End each use
case with its limits. Link it from the case study (end of §3 and §4) and from both technical guides where
they already link the case study.

**T2. Fix the case study's errors.** These are known: "13 known-answer checks" → 12 (EN §3, ES twin,
`es_bench/README.md`); the "two bugs" claim (A0b); any wording that reads `top1_found_in_top10` as
"same top-10". Grep both languages.

**T3. Root README.** `README.md` lists `es_bench/` but not `memo_bench/`. Add the row, plus the new
catalogue docs.

**T4. Bench READMEs.** In `es_bench/README.md` and `memo_bench/README.md`, add a short "What each section
measures" list (sections 1–8 and 0–6), and a "Last results" line that points to the catalogue. Do not paste
result tables here: the catalogue is their one home.

**T5. Mark the superseded premise.** At the top of `PROVISION_SEARCH_CHECKS.md` and in the first cell /
docstring of `provision_search_benchmark.py` and `.ipynb`, add a note: the Qdrant `provisions` collection and
the 0.90 merge threshold do not exist in the platform; see `es_bench/` and the case study. If you edit the
notebook, regenerate it with `python build_notebooks.py` instead of editing the JSON by hand (see
`CURSOR_HANDOFF.md`), and touch nothing else in the code.

**T6 (optional). Figures.** `docs/guide/make_figures.py` already makes the guide's figures. If you add any,
use only the aggregate CSVs in §3 (read in place, never copied into the repo). Two that would help: recall vs
k per method at floors 30 / 50 / 70 (A5), and the memo request-time breakdown (B5 + B7). Save the PNGs to
`docs/guide/img/en` and `img/es`.

---

## 6. Report back

When done, list:
1. The files created and changed.
2. Every number you could not trace to a source file. There should be none; if there is one, leave a
   `TODO(source)` in the doc and do not invent it.
3. Discrepancies found between the existing docs and the result files.
4. Anything in the code that looks wrong (not fixed, per rule 7).

## 7. Done when

- [ ] Every row of §4 appears in both catalogue docs, with its full table and its limits.
- [ ] `grep -rn "13 known\|13 checks\|13 comprobaciones" docs es_bench` returns nothing.
- [ ] Every number in the new docs is in a §3 source file (spot-check 10 at random).
- [ ] `git status` shows only `.md` files (plus regenerated notebooks / PNGs if T5 / T6 were done). No CSV,
      no `.npz`, no result folder.
- [ ] `git diff` contains no client, fund, investor or firm names and no provision or memo text.
- [ ] EN and ES have the same sections and the same numbers.
