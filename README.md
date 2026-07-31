# BanglaPhonologyBench

A phonology benchmark for Bangla LLMs, combined with a tokenization-misalignment
analysis quantifying how badly modern subword tokenizers mangle an abugida
script. Replicates and extends two papers:

- **PhonologyBench** (Suvarna et al., 2024) — task design (G2P, syllable
  counting, rhyme).
- **How Tokenization Limits Phonological Knowledge Representation in LMs**
  (Liao & Shi, 2026; [code](https://github.com/liaodisen/Tokenization-Phonology))
  — probing protocol + the STAD metric, lifted here from characters to
  aksharas (Bengali grapheme clusters).

Full spec: [`BanglaPhonologyBench_Research_Spec.md`](BanglaPhonologyBench_Research_Spec.md).
Complete build history, including bugs found and fixed along the way:
[`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md).

## What this measures

Three metrics, reported as a triple — never collapsed into one scalar
(Spec C.4):

| Metric | What it measures |
|---|---|
| **GTAD** | Fraction of a tokenizer's internal boundaries that violate akshara (grapheme-cluster) integrity — splitting *inside* a Bengali codepoint's 3-byte UTF-8 encoding, or between a consonant and its vowel sign. ≡ 0 for alphabetic scripts under any codepoint-respecting tokenizer, so it isolates a misalignment axis that doesn't exist in English. |
| **STAD_bn** | Hamming distance between a tokenizer's boundaries and the *phonemic syllable* boundaries, over the akshara gaps. Lifted from Liao & Shi's character-level STAD (a matra is a character but never a phonological unit — the wrong base unit for an abugida). |
| **ρ (rho)** | Fraction of phonemic syllable boundaries that fall *inside* an akshara and thus can't be represented at any tokenizer boundary, no matter how good the tokenizer is — a property of the script itself, not any particular tokenizer. |

**Headline finding so far:** English-centric byte-BPE tokenizers (Llama-3.1,
GPT-2, ByT5) put ~99% of Bangla words in the worst "cluster-broken"
category (GTAD > 0) — they don't just misalign syllables, they routinely
split *inside* a single Bengali codepoint. Bangla-aware tokenizers
(TigerLLM, BanglaT5) don't.

## Repo layout

```
bangla_phonology.py          reference implementation: akshara segmenter,
                              phonemic syllabifier, orthography<->phoneme
                              aligner, GTAD/STAD_bn/rho metrics
src/
  tokenizer_adapter.py        real-tokenizer -> byte-span adapter (byte-BPE,
                              SentencePiece, ByT5)
  rhyme.py                    rime()/assonance_key()/is_trivial_pair() +
                              Success@k scoring for the rhyme tasks
scripts/                      dataset builders, ingestion, metrics, figures,
                              annotation tooling, M4 probing library
notebooks/m4_probing.ipynb    Kaggle notebook: hidden-state extraction +
                              linear probes (file-level resumable)
data/                         lexicon, frozen task datasets, annotation
                              sheets, probing exports
tests/                        pytest suite (127 tests)
docs/
  DEVELOPMENT_LOG.md           full chronological build history
  probing_integration.md       M4 design notes
  annotation_guide.md          human-annotation workflow
```

## Status

- **M1–M3: done**, including a full solo-annotator human verification pass
  on every frozen task dataset:

  | Task | Dataset | Human review |
  |---|---|---|
  | 1 — G2P | `data/tasks/g2p.jsonl` (3,000 words) | etymology tags 353/353 reviewed |
  | 2a — syllable count | `data/tasks/syllable_count_word.jsonl` | (shares G2P's words) |
  | 3a — rhyme classification | `data/task3a_rhyme_pairs.jsonl` (400 pairs) | **400/400, 100% agreement** |
  | 3b — rhyme generation | `data/task3b_rhyme_generation.jsonl` (300 prompts) | **300/300, no corrections** |
  | 4 — schwa deletion | `data/tasks/schwa_deletion.jsonl` (1,000 words, 160 reviewed) | **160/160, 100% agreement** |

  The aligner underlying all of this reaches **99.95% coverage** on the
  60K-entry pronunciation lexicon (see the development log for the two
  encoding bugs that got it there). Etymology auto-tagging, by contrast, is
  a genuinely hard problem — the orthography-only heuristic tagger was only
  **38.2% accurate** against human judgment, an honestly-reported result,
  not a smoothed-over one.

- **M4 (Kaggle probing)**: pipeline built (`scripts/kaggle_probing_lib.py`,
  unit-tested with a mock model; `notebooks/m4_probing.ipynb`), not yet run
  against real models — needs GPU.
- **M5 (zero-shot evals)**: not started.

## Reproducing

```bash
pip install -r requirements.txt
pytest                              # 127 tests
python scripts/ingest_lexicon.py    # rebuild the cleaned lexicon
python scripts/compute_metrics.py data/wordlist_top3000.tsv -o results/metrics_top3000.csv
python scripts/make_figure2.py      # GTAD/STAD/rho distributions across tokenizers
```

See `CLAUDE.md` for coding conventions and Bangla-specific edge cases
(ri-kar, anusvara, ZWJ/ZWNJ in conjuncts, the composition-exclusion trap
around ড়/ঢ়/য়) and `docs/annotation_guide.md` for the human-review workflow.

## Data sources

- Pronunciation lexicon: Google `language-resources` bn-BD (CC BY 4.0),
  65,037 entries.
- Frequency data: [`wordfreq`](https://github.com/rspeer/wordfreq) (bn).

## License

Code: Apache-2.0 (see `pyproject.toml`). The Google bn-BD lexicon is
CC BY 4.0 — see `data/google_bn_LICENSE.txt`.
