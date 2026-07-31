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

1. **M1 (wks 1–3):** lexicon acquisition + licensing audit; akshara segmenter + phonemic
   syllabifier validated (✅ implementation exists, 28 gold checks pass).
2. **M2 (wks 3–6):** Tasks 1–2 datasets frozen; GTAD/STAD/ρ computed for all tokenizers;
   descriptive stats.
3. **M3 (wks 5–8):** Tasks 3–4 built (rhyme mining + schwa alignment); annotation pass.
4. **M4 (wks 7–10):** probing harness (fork liaodisen/Tokenization-Phonology, swap data
   loaders); hidden-state extraction on **Kaggle T4s** (fp16 batch; 4-bit if needed for 8B).
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
