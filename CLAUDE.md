# CLAUDE.md — BanglaPhonologyBench + GTAD/STAD Analysis

## Project goal

Build **BanglaPhonologyBench**, a phonology benchmark for Bangla LLMs (G2P, syllable
counting, rhyme, schwa-deletion prediction), combined with a **tokenization-misalignment
analysis**. Replicates and extends two papers:

- *PhonologyBench* (Suvarna et al., 2024) — task design (G2P, syllable counting, rhyme).
- *How Tokenization Limits Phonological Knowledge Representation in LMs*
  (Liao & Shi, 2026; code: `github.com/liaodisen/Tokenization-Phonology`) — probing
  protocol + STAD metric, lifted here from characters to aksharas.

Full spec: `BanglaPhonologyBench_Research_Spec.md` (read Parts B–C before touching the
segmenter, syllabifier, aligner, or metrics).

## The metric triple (Spec Part C) — do NOT change definitions without asking

Hierarchy: bytes ⊂ codepoints ⊂ aksharas ⊂ syllables. The three metrics are reported as
a triple, never collapsed into one scalar:

- **GTAD** ∈ [0,1] — fraction of tokenizer-internal boundaries that violate akshara
  integrity (boundary ∉ legal akshara gaps). Decomposed into byte-internal / matra-split /
  conjunct-split violations. GTAD ≡ 0 for alphabetic scripts, so it isolates the
  abugida-specific misalignment axis.
- **STAD_bn** ∈ [0,1] — Hamming distance between v_syl and v_tok over the m−1
  inter-akshara gaps, normalized by (m−1); `None` when m < 2. This is Liao & Shi's STAD
  lifted to akshara (cluster-level) gaps; reduces to their STAD when every akshara is one
  codepoint.
- **rho (ρ)** ∈ [0,1] — fraction of phonemic syllable boundaries with *no* legal akshara-gap
  image (e.g., /ʃan.to/ inside the conjunct ন্ত of শান্ত). A property of the script itself:
  the ceiling on what any akshara-respecting tokenizer can represent.

Word categories for split analyses: **A** (aligned: GTAD=0 ∧ STAD=0), **CB**
(cluster-broken: GTAD>0), **M** (syllable-misaligned: GTAD=0 ∧ STAD>0.25).

## Code layout

- `bangla_phonology.py` — validated reference implementation:
  1. `segment_aksharas` / `legal_gap_positions` — akshara segmenter (Spec B.1)
  2. `syllabify` / `is_nucleus` / `LEGAL_ONSETS` — phonemic syllabifier (Spec B.2)
  3. `align` — greedy monotone orthography→phoneme aligner; failures are *flagged*
     (`Alignment.ok=False`), never silently accepted (Spec C.4)
  4. `gtad`, `stad_bn`, `token_byte_spans_from_strings` — metrics (Spec C.2–C.4)
- `tests/` — pytest suite (gold words + metric regression values). Run: `pytest`
- `test_bangla_phonology.py` — original human-readable validation harness (prints tables;
  kept as a demo, excluded from pytest collection via `testpaths`).

## Bangla edge cases already handled — don't regress these

- **ri-kar ৃ emits /ri/**: one consonant (r) + one vowel (i); the aligner counts ৃ as an
  extra expected consonant (কৃষ্ণ → কৃ→/kri/).
- **Vowel-initial aksharas with anusvara**: অংক → [অং][ক]; ং attaches to the vowel cluster
  and emits coda /ŋ/ (a consonant, NOT nasalization).
- **Candrabindu ঁ = nasalization**, not a consonant: চাঁদ → /tʃãd̪/, one nucleus.
  `is_nucleus` NFD-normalizes so nasalized vowels work whether precomposed or combining
  (U+0303), and strips the inverted-breve U+032F so diphthongs like /oi̯/ are one nucleus.
- **ZWJ/ZWNJ in conjuncts**: hasanta+ZWNJ *blocks* conjunct formation (splits the akshara);
  hasanta+ZWJ keeps it joined. Both are consumed into the cluster, flag recorded upstream.
- **Khanda-ta ৎ** is coda-only: never starts a cluster, attaches to the preceding akshara,
  emits /t̪/.
- **Geminates split across syllable boundaries** automatically (বিশ্ব → /biʃ.ʃo/): identical
  CC is never in `LEGAL_ONSETS`.
- **Loanword s-clusters** (/sʈ/, /st/, /sk/…) are legal onsets: স্টেশন → /sʈe.ʃon/.

## Coding conventions

- **NFC-normalize all Bangla input** (`unicodedata.normalize("NFC", w)`) before segmenting.
- **Phonemes are lists of strings**, one phoneme per element (`["ʃ","a","n","t̪","o"]`).
  Multi-char IPA (t̪, gʱ, dʒ, ã) is ONE list element. Never operate on joined IPA strings.
- **Tokenizers plug in as byte spans**: `List[Tuple[int, int]]` of UTF-8 byte [start, end)
  offsets. Adapt any HF/tiktoken tokenizer via offsets or
  `token_byte_spans_from_strings`. Every Bengali codepoint is 3 bytes in UTF-8 —
  byte-internal splits are real and must map to the containing codepoint's interior.
- Alignment failures and m<2 words are flagged/None, not coerced — downstream analyses
  restrict to aligner-confident items and report coverage.
- Gold pronunciations come from human-curated lexica, never from rule-based G2P (Spec A.1).

## Milestone plan (Spec Part D)

1. **M1 (wks 1–3):** ✅ lexicon acquired (Google bn-BD, CC BY 4.0, 65,037 entries) +
   licensing recorded; akshara segmenter + phonemic syllabifier validated (96 tests,
   99.86% syllable-count agreement, aligner 99.95% coverage against the lexicon —
   31 residual failures are heterogeneous long-tail items: English-loan অ্যা-digraph
   transliteration, archaic double-consonant spellings, proper-name hiatus).
2. **M2 (wks 3–6):** ✅ Tasks 1–2 datasets frozen (`data/tasks/g2p.jsonl`,
   `syllable_count_word.jsonl`, 3,000 words each); GTAD/STAD/ρ computed for 5 real
   tokenizers (`results/metrics_top3000.csv`); descriptive stats (`figures/figure2.*`).
3. **M3 (wks 5–8):** ✅ DONE, including the human annotation pass. Task 3a rhyme pairs:
   `src/rhyme.py` (proper open/closed-final-syllable rime rule, Spec A.4 3a) +
   `scripts/build_rhyme_dataset.py` -> `data/task3a_rhyme_pairs.jsonl` (200 positive +
   200 negative, negatives split 80 assonance_decoy / 40 ortho_decoy / 80 random;
   anti-leakage via `is_trivial_pair` + phoneme-sequence stem check) — **human-verified
   400/400, 100% agreement**. Task 3b rhyme generation: `scripts/build_rhyme_generation_
   dataset.py` -> `data/task3b_rhyme_generation.jsonl` (300 prompts: 200 common/100 rare;
   gold = lexicon words sharing the prompt's rime, ADDITIONALLY filtered through
   `is_trivial_pair` — without it, short grammatical suffixes like -এর dominate gold sets
   almost entirely, see the script's docstring and the Spec A.4 3b note) — **human-verified
   300/300, no corrections**. Task 4 schwa: `schwa_deletion.jsonl` 1,000 words from the aligner
   — **human-verified 160/160 (stratified sample), 100% agreement**. Task 1 etymology:
   4-way tag (added `deshi`, see Spec A.3.5 note) — **human-verified 353/353; heuristic
   tagger accuracy 38.2%**, a real and reportable weakness of orthography-only etymology
   guessing. Annotation tooling: `scripts/tag_etymology_heuristic.py`,
   `scripts/build_annotation_sheet.py`, `scripts/apply_annotations.py`, sheets in
   `data/annotation/` — see `docs/annotation_guide.md`. No inter-annotator κ anywhere
   (single annotator, documented limitation vs Spec A.4's 2-annotator design).
4. **M4 (wks 7–10):** pipeline BUILT, not yet run (needs GPU). Pivoted away from
   forking `liaodisen/Tokenization-Phonology` for the core pipeline — see
   `docs/probing_integration.md`'s "Pivot" section for why. Instead:
   `scripts/kaggle_probing_lib.py` (file-level-resumable hidden-state extraction +
   a generalized A/M/CB probe battery, unit-tested with a mock model —
   `tests/test_kaggle_probing_lib.py`, no GPU needed) + `notebooks/m4_probing.ipynb`
   (clone-and-run on Kaggle; BanglaT5 first as a smoke test, then TigerLLM/GPT-2/
   ByT5/Llama-3.1 with 4-bit for the 8-9B models). Export already runs locally via
   `scripts/export_for_probing.py`. Rhyme-probe A/M/CB category join not yet built
   (flagged in the notebook's final cell, not silently skipped).
5. **M5 (wks 9–12):** zero-shot evals (closed + open models); human baselines; regression
   analysis; writing.

Workflow note: dataset construction, segmenter/metric work, and tokenizer analyses run
fine locally (CPU-only). Only M4/M5 GPU forward passes go to Kaggle — this repo is the
source of truth; Kaggle notebooks pull it.

## Commands

- `pytest` — run the test suite (must stay green).
- `python test_bangla_phonology.py` — human-readable validation tables + worked example.
- `python scripts/validate_gold_tokenizers.py` — real vs simulated tokenizer table on gold words.
- `python scripts/ingest_lexicon.py` — rebuild data/lexicon_clean.tsv + aligner report.
- `python scripts/build_top3000.py` — top-3000 frequency wordlist (wordfreq bn).
- `python scripts/compute_metrics.py data/wordlist_top3000.tsv -o results/metrics_top3000.csv`
- `python scripts/make_figure2.py` — Figure 2 (metric distributions + violation types).

Run Python scripts with `-X utf8` on Windows (Bangla output to console).

## Pipeline data files

- `data/google_bn_lexicon.tsv` — raw Google bn-BD lexicon (65,037 entries, CC BY 4.0,
  pre-syllabified, NO vowel nasalization by policy — see data/phoneme_map.tsv header).
- `data/phoneme_map.tsv` — lexicon notation -> our IPA (glides merge into diphthongs).
- `data/lexicon_clean.tsv` — 60,087 ingested entries; aligner_ok column gates analyses.
- `data/aligner_failures_sample.tsv` — 30-sample of aligner failures awaiting review
  (main open categories: silent য-ফলা/ব-ফলা, ক্ষ/জ্ঞ simplification, vowel hiatus/য়,
  acronym spellings).
- Tokenizer registry lives in `src/tokenizer_adapter.py` (llama3 needs HF_TOKEN;
  ungated NousResearch mirror is the fallback and is byte-identical for tokenization).
- `data/tasks/*.jsonl` — frozen task datasets (Spec A.5 schema): `g2p.jsonl`,
  `syllable_count_word.jsonl` (3,000 words each, 1,500 HighFreq/1,500 LowFreq by
  wordfreq(bn) zipf quartile), `schwa_deletion.jsonl` (1,000 words, environment-
  stratified). Built by `scripts/build_tasks.py`. Human-verified subsets noted in
  the M3 milestone entry above; unreviewed items still `annotation.verified=false`.
- `data/task3a_rhyme_pairs.jsonl` — Task 3a rhyme pairs (200 pos + 200 neg), built
  separately by `scripts/build_rhyme_dataset.py` + `src/rhyme.py` (NOT
  `build_tasks.py` — that script's earlier ad hoc rhyme code was retired in favor
  of this properly open/closed-syllable-aware implementation). `--stats` mode first,
  dry-run by default, `--confirm` to write; see the script's docstring.
  **Human-verified, 400/400, 100% agreement.**
- `data/task3b_rhyme_generation.jsonl` — Task 3b rhyme-generation prompts (300: 200
  common/100 rare) + gold rhyme sets, built by `scripts/build_rhyme_generation_dataset.py`
  (reuses `build_rhyme_dataset.py`'s lexicon loading + `src/rhyme.py`'s `rime()`).
  Gold sets are filtered through `is_trivial_pair` (deviation from the spec's literal
  "all lexicon words with matching rime" — documented in the spec and script docstring;
  without it, productive grammatical suffixes like -এর dominate gold sets almost
  entirely). `src/rhyme.py`'s `success_at_k`/`mean_success_at_k` implement the
  PhonologyBench-style Success Rate@k scoring for when M5 zero-shot evals run against
  this. Poetry-corpus gold enrichment (spec's stretch goal) NOT implemented — no corpus
  text in this repo, so Success Rate@k here is a lower bound. **Human-verified, 300/300,
  no corrections.**
- `data/probing_export/` — CSV views of the above for the M4 probing harness, split by
  A/M/CB category per tokenizer (`scripts/export_for_probing.py`); see
  `docs/probing_integration.md`.
- `data/annotation/` — M3 human review sheets (etym/rhyme/schwa), see `docs/annotation_guide.md`.
  Edit the CSVs, then `python scripts/apply_annotations.py` to merge into `data/tasks/*.jsonl`.
- `normalize_bn()` in bangla_phonology.py: use this, not bare NFC, for any new ingestion
  — NFC alone leaves ড়/ঢ়/য় decomposed (they're Unicode composition exclusions).
