# Probing harness integration (Milestone M4)

Fork target: `github.com/liaodisen/Tokenization-Phonology` (Liao & Shi 2026).
Clone it fresh when you reach M4 — it is **not** committed to this repo:

```
git clone https://github.com/liaodisen/Tokenization-Phonology.git external/Tokenization-Phonology
```

(`external/` is gitignored here; it's third-party code with its own history.)

## What upstream expects

Inspected `probing/generate_embedding.py`, `probing/train_probe_g2p.py`,
`probing/train_probe_syl.py`, and the example CSVs in `datasets/`:

- **G2P / syllable-count probe** (`train_probe_g2p.py`, `train_probe_syl.py`):
  reads a CSV with columns `word, phon_vec, syllables`. `phon_vec` is a
  **padded phoneme-index vector** (list of ints, 0 = padding), normalized
  per-column before RidgeCV regression. `syllables` is a Python-literal list
  used only for `len()` (syllable count). Trained per hidden-state layer,
  10 seeds (10–19), 80/20 split, with `--control_task_label` /
  `--control_task_embed` flags for the random-embedding control — exactly
  the protocol in Spec A.6.
- **Rhyme probe** (`train_probe_rhyme.py` + `generate_embedding.py --feature
  rhyme`): reads `word1, word2, label`; prompt is `f"{word1} {word2}"` (or
  IPA/slash variants via `--use_IPA`/`--use_slash` — this is the delimiter
  intervention hook for Spec C.6.4).
- **Their A/CB split analogue**: they don't have GTAD, so they split
  `arpabet_data_{LLM}_good.csv` / `_bad.csv` by a Hamming-distance-based
  tokenization-quality proxy (`hamming_distance` column) computed per LLM.
  Two separate probes are trained (`fp1`=bad, `fp2`=good) and compared —
  this is the slot our **A / M / CB categories** (Spec C.4) fill directly,
  just three-way instead of two.
- Embeddings come from `extract_embeddings_for_all_layers`: last-token
  hidden state at every layer, `output_hidden_states=True`, causal LM or
  `T5EncoderModel` for ByT5/mT5. GPU if available, else CPU (too slow for
  8B models — this is the M4 Kaggle-T4 step).
- **Known upstream issue**: `generate_embedding.py` has a hardcoded
  `access_token` (an HF token) and Compute-Canada-style absolute paths
  (`/model-weights/...`) for `llama3.1`/`mistral`. Both need patching after
  cloning — swap the token for `os environ["HF_TOKEN"]` and the paths for
  the plain HF repo ids (already the commented-out fallback in that file).

## Our export (already runnable, no GPU)

`python scripts/export_for_probing.py` reads `data/tasks/g2p.jsonl` and
`data/tasks/rhyme_pairs.jsonl` and writes, per tokenizer, into
`data/probing_export/`:

- `g2p_{tokenizer}_{A,M,CB}.csv` — word, phon_vec (padded phoneme indices),
  syllables, ipa. Category = Spec C.4's A/M/CB, computed from our GTAD/STAD.
- `rhyme_pairs.csv` — word1, word2, label (shared across tokenizers; the
  category split for rhyme probing happens at analysis time by joining on
  GTAD/STAD per tokenizer, not by pre-splitting the file).
- `data/tasks/phoneme_vocab.json` — the phoneme→index map + pad length,
  needed to decode `phon_vec` back to IPA later.

At M4, copy `data/probing_export/*.csv` into the cloned fork's
`probing/data/` (create that directory; it doesn't exist upstream — their
default paths are `datasets/` and `embeddings/`, adjust `--file_dir` flags
accordingly, or just point `--file_dir` at `data/probing_export` directly).

## Numbers from the current freeze (3,000-word G2P set)

- Phoneme vocabulary: 59 symbols. Pad length: **20** (max phonemes in one
  word — পূর্বপরিকল্পিতভাবে "in a pre-planned way", a long tatsama compound).
  This exceeds the spec's estimate of 12–14; worth a sentence in the paper
  — Bangla compounding produces longer tails than expected even against
  English's ~8.
- Category counts per tokenizer (out of 3,000; remainder = quarantined or
  single-akshara words where STAD is undefined and no A/M/CB label applies):

  | tokenizer | A | M | CB | excluded |
  |---|---|---|---|---|
  | llama3   | 0   | 0    | 2,980 | 20 |
  | gpt2     | 0   | 0    | 2,980 | 20 |
  | byt5     | 0   | 0    | 2,980 | 20 |
  | tigerllm | 684 | 658  | 1,638 | 20 |
  | banglat5 | 653 | 1,818| 456   | 73 |

  This *is* the headline result, not a bug: every English-centric tokenizer
  puts 99%+ of Bangla words in the cluster-broken (CB) category — GTAD > 0
  for nearly every word — so their G2P/syllable probes will only ever see
  the worst category. TigerLLM and BanglaT5 are the ones where the A vs M
  vs CB comparison is actually informative. banglat5's larger "excluded"
  count comes from this G2P sample (drawn across the full frequency range,
  Spec A.4.1) including rarer/longer words than the wordfreq-filtered
  top-3000 set used for Figure 2, where banglat5 had zero quarantines.

## Still to do at M4 (needs GPU — Kaggle T4, not local)

1. Patch `generate_embedding.py` (token + model paths, see above).
2. Run `generate_embedding.py --file_dir data/probing_export --file_name
   g2p_{tokenizer}_{cat}.csv --LLM {llm} --layers -1 --feature g2p` per
   tokenizer × category (only for tokenizers with an actual causal LM /
   encoder behind them — TigerLLM-9B and Llama-3.1-8B need real GPU;
   BanglaT5 fits on a T4 easily as it's an encoder-decoder ~580M).
3. Run `train_probe_g2p.py --LLM {llm}` three times (once per category)
   instead of upstream's twice (good/bad), plus `--control_task_label` for
   the random-embedding control (10 seeds each, matches Spec A.6).
4. Repeat for `train_probe_syl.py` and `train_probe_rhyme.py`.
5. One-sided t-tests A > M > CB per layer per task (Spec C.6.2), same
   reporting style as Liao & Shi Table 3.
