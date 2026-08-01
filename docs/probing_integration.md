# Probing harness integration (Milestone M4)

**Status: implemented and ready to run.** `notebooks/m4_probing.ipynb` is
the M4 pipeline — clone this repo on Kaggle, run the notebook, done. This
doc is now background/reference (what upstream's harness expects, and why
we built our own instead of forking it) rather than a to-do list.

## Pivot: self-contained pipeline instead of forking upstream

The original plan (below, kept for reference) was to fork
`github.com/liaodisen/Tokenization-Phonology` (Liao & Shi 2026) and reuse
its `probing/generate_embedding.py` + `probing/train_probe_*.py`. In
practice:

- `generate_embedding.py`'s model-loading/extraction functions are simple
  enough (load tokenizer+model, one forward pass, take last-token hidden
  states per layer) that reimplementing them directly is less code than
  cloning-and-patching a third-party repo with known bugs (a hardcoded HF
  token, Compute-Canada-specific absolute model paths).
- `train_probe_{g2p,syl,rhyme}.py` are hardcoded to exactly TWO conditions
  (their good/bad Hamming-distance split) and specific English-dataset
  filenames — not reusable as-is for our three-way A/M/CB split (Spec
  C.4). Adapting them would mean rewriting most of the file anyway.

So `scripts/kaggle_probing_lib.py` reimplements both pieces natively in
this repo: `extract_g2p` (file-level-resumable hidden-state extraction)
and `run_probe_battery` (the same RidgeCV/LogisticRegression protocol,
generalized to N categories with automatic pairwise one-sided t-tests).
Both are unit-tested with a mock model (`tests/test_kaggle_probing_lib.py`,
no GPU/network needed) — see `docs/DEVELOPMENT_LOG.md` for what was
validated and how, since none of this could be run end-to-end without a
real GPU before shipping it.

The upstream fork is still worth cloning **separately, later**, if you
want a direct English-vs-Bangla comparison using their own `datasets/`
(arpabet/rhyming English data) under the identical protocol — that's a
nice-to-have for the paper's cross-linguistic section, not part of the
core M4 pipeline.

## What upstream expects (reference — not what we actually run)

Inspected `probing/generate_embedding.py`, `probing/train_probe_g2p.py`,
`probing/train_probe_syl.py`, and the example CSVs in `datasets/`:

- **G2P / syllable-count probe** (`train_probe_g2p.py`, `train_probe_syl.py`):
  reads a CSV with columns `word, phon_vec, syllables`. `phon_vec` is a
  **padded phoneme-index vector** (list of ints, 0 = padding), normalized
  per-column before RidgeCV regression. `syllables` is a Python-literal list
  used only for `len()` (syllable count). Trained per hidden-state layer,
  10 seeds (10–19), 80/20 split, with `--control_task_label` /
  `--control_task_embed` flags for the random-embedding control — exactly
  the protocol in Spec A.6, and exactly what `run_probe_battery` reproduces.
- **Rhyme probe** (`train_probe_rhyme.py` + `generate_embedding.py --feature
  rhyme`): reads `word1, word2, label`; prompt is `f"{word1} {word2}"` (or
  IPA/slash variants via `--use_IPA`/`--use_slash` — this is the delimiter
  intervention hook for Spec C.6.4). Not yet wired into the notebook — see
  its final markdown cell for what's needed.
- **Their A/CB split analogue**: they don't have GTAD, so they split
  `arpabet_data_{LLM}_good.csv` / `_bad.csv` by a Hamming-distance-based
  tokenization-quality proxy (`hamming_distance` column) computed per LLM.
  Two separate probes are trained (`fp1`=bad, `fp2`=good) and compared —
  this is the slot our **A / M / CB categories** (Spec C.4) fill directly,
  three-way instead of two, via `run_probe_battery`.
- Embeddings come from `extract_embeddings_for_all_layers`: last-token
  hidden state at every layer, `output_hidden_states=True`, causal LM or
  `T5EncoderModel` for ByT5/mT5 — same approach `kaggle_probing_lib`'s
  `extract_g2p` uses.
- **Known upstream issue** (only relevant if you clone the fork
  separately for the EN/BN comparison mentioned above):
  `generate_embedding.py` has a hardcoded `access_token` (an HF token) and
  Compute-Canada-style absolute paths (`/model-weights/...`) for
  `llama3.1`/`mistral`. Patch the token to `os.environ["HF_TOKEN"]` and the
  paths to the plain HF repo ids (already the commented-out fallback in
  that file) before running anything from the fork directly.

## Our export (already runnable, no GPU)

`python scripts/export_for_probing.py` reads `data/tasks/g2p.jsonl` and
`data/task3a_rhyme_pairs.jsonl` and writes, per tokenizer, into
`data/probing_export/`:

- `g2p_{tokenizer}_{A,M,CB}.csv` — word, phon_vec (padded phoneme indices),
  syllables, ipa. Category = Spec C.4's A/M/CB, computed from our GTAD/STAD.
- `rhyme_pairs.csv` — word1, word2, label (shared across tokenizers, flat).
- `rhyme_{tokenizer}_{A,M,CB}.csv` — same columns, split by per-pair
  category. Unlike g2p.jsonl, task3a_rhyme_pairs.jsonl carries no
  precomputed per-tokenizer tokenization field, so `build_rhyme_category_
  export` in `export_for_probing.py` computes GTAD/STAD fresh per word per
  tokenizer (loads tokenizers — CPU-only, no GPU) and combines each pair's
  two per-word categories into one via `categorize_pair` (the worse of the
  two, since either word's misalignment compromises the joint
  `f"{word1} {word2}"` prompt embedding the rhyme probe extracts from).
- `data/tasks/phoneme_vocab.json` — the phoneme→index map + pad length,
  needed to decode `phon_vec` back to IPA later.

`notebooks/m4_probing.ipynb` clones this repo on Kaggle and reads
`data/probing_export/` directly — no manual copying needed.

## Numbers from the current freeze (3,000-word G2P set)

- Phoneme vocabulary: 57 symbols. Pad length: **18** (max phonemes in one
  word — a long tatsama compound). This exceeds the spec's estimate of
  12–14; worth a sentence in the paper — Bangla compounding produces
  longer tails than expected even against English's ~8.
- Category counts per tokenizer (out of 3,000; remainder = quarantined or
  single-akshara words where STAD is undefined and no A/M/CB label applies):

  | tokenizer | A | M | CB | excluded |
  |---|---|---|---|---|
  | llama3   | 0   | 0    | 2,982 | 18 |
  | gpt2     | 0   | 0    | 2,982 | 18 |
  | byt5     | 0   | 0    | 2,982 | 18 |
  | tigerllm | 697 | 671  | 1,614 | 18 |
  | banglat5 | 647 | 1,822| 442   | 89 |

  This *is* the headline result, not a bug: every English-centric tokenizer
  puts 99%+ of Bangla words in the cluster-broken (CB) category — GTAD > 0
  for nearly every word — so their G2P/syllable probes will only ever see
  the worst category (the notebook handles this: fewer than 2 available
  categories for a tokenizer just skips probe training for it, logged, not
  silently zero-filled). TigerLLM and BanglaT5 are the ones where the A vs
  M vs CB comparison is actually informative.

## Running M4

1. Upload/import `notebooks/m4_probing.ipynb` to Kaggle (or open it and
   copy cells in) with a GPU accelerator (T4 x1 is fine to start).
2. First run: nothing to restore, just run all cells top to bottom.
   Recommended order (already the notebook's default `RUN_ORDER`):
   BanglaT5 first (smallest, ~580M, fast — a pipeline smoke test), then
   TigerLLM/GPT-2/ByT5/Llama-3.1.
3. When you stop (end of session, quota, or just done for now): click
   **Save Version** to persist `/kaggle/working/checkpoints/`.
4. Next session: attach that saved output as an input dataset, point
   `CHECKPOINT_INPUT` at it, re-run — already-finished (tokenizer,
   category) combos are skipped automatically (see the notebook's own
   resumability section for the reasoning: file-level only, since each
   combo is minutes not hours).
5. Once results exist for TigerLLM and BanglaT5 (the two with real 3-way
   A/M/CB splits): one-sided t-tests A > M > CB per layer per task are
   computed by `run_probe_battery` and flattened into
   `checkpoints/probe_pairwise_summary.csv` by the notebook's summary cell
   (section 7) — ready to read directly into the paper's Table-3-style
   reporting format, no manual aggregation needed.
