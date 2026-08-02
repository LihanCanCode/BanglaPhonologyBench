# Development Log — BanglaPhonologyBench

Chronological record of every work session on this project: what was built,
what broke, how it was diagnosed and fixed, and what each milestone's
numbers came out to. Written so a future Claude session (or the user,
returning after a break) can read this once and have full context —
**read this before `CLAUDE.md`** if you need to understand *why* the repo
looks the way it does, not just what it currently contains. `CLAUDE.md` is
the living reference (conventions, edge cases, current status); this file
is the history that explains how it got there.

Commit hashes are given so you can `git show <hash>` for the literal diff
behind any bullet.

---

## 2026-07-31 — Repo bootstrap

**Goal:** turn a bare spec + reference implementation into a working,
tested repo.

- Read `BanglaPhonologyBench_Research_Spec.md` in full, `bangla_phonology.py`
  (the pre-existing reference implementation: akshara segmenter, phonemic
  syllabifier, orthography↔phoneme aligner, GTAD/STAD_bn/ρ metrics), and
  `test_bangla_phonology.py` (a human-readable validation harness, not a
  pytest suite).
- Wrote the initial `CLAUDE.md` (project goal, metric triple definitions,
  coding conventions, milestone plan).
- `git init`; converted the validation harness into `tests/` (pytest,
  28/28 gold-word checks passing); `pyproject.toml` + `requirements.txt`.
- **Result:** 28 tests green. Commit `7599e4e`.

### Real tokenizer adapter (same day)

- Built `src/tokenizer_adapter.py`: converts real HF tokenizer output
  (byte-level BPE for GPT-2/Llama-3, SentencePiece for T5/Gemma-family
  models including `<0xNN>` byte-fallback pieces, ByT5's literal-byte
  scheme) into UTF-8 byte spans, replacing the spec's illustrative
  "SIMULATED" tokenizer tables.
- Registered 5 tokenizers: `llama3` (gated — needs `HF_TOKEN`; falls back
  to the ungated `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror, which is
  byte-identical for tokenization purposes), `gpt2`, `byt5`, `tigerllm`
  (`md-nishat-008/TigerLLM-9B-it`), `banglat5` (`csebuetnlp/banglat5`).
- Validated against the 14 gold words: real tokenizers are dramatically
  worse than the illustrative SIMULATED tables suggested (e.g. Llama-3.1
  GTAD as high as 0.86 on some words vs. the SIMULATED table's 0.5).
- **Result:** 87 tests green. Commit `3ae3f08`.

### Lexicon ingestion + Figure 2 (same day)

- Sourced the Google `language-resources` bn-BD pronunciation lexicon
  (65,037 entries, CC BY 4.0, archived repo — pinned a local copy since
  upstream won't change but also won't disappear-proof itself).
- Wrote `data/phoneme_map.tsv` (lexicon notation → project IPA) and
  `scripts/ingest_lexicon.py`. First run: **86.85% aligner coverage**
  (many failures from anusvara/khanda-ta being consumed *before* the vowel
  instead of after — a real Spec C.4 compliance bug, fixed same session,
  bumping coverage to 89.44%).
- Built `scripts/build_top3000.py` (wordfreq-based frequency list) and
  `scripts/compute_metrics.py`, producing `results/metrics_top3000.csv`
  and Figure 2 (GTAD/STAD/ρ distributions across all 5 tokenizers).
- **Finding worth remembering:** English-centric tokenizers (Llama-3, GPT-2,
  ByT5) put ~99% of Bangla words in the worst "cluster-broken" (GTAD>0)
  category — they don't just misalign syllables, they routinely split
  *inside* a single Bengali codepoint's 3-byte UTF-8 encoding.
- **Result:** 96 tests green. Commit `db7a526`.

---

## 2026-08-01 — Aligner hardening, task datasets, first critical bugfix

### Aligner v2: backtracking parser (`20bdac7`)

The original aligner was a single-pass greedy walk that couldn't handle
vowel hiatus, digraph vowels, or optional consonant deletion (silent
ফলা, ক্ষ/জ্ঞ simplification). Replaced with a backtracking parser
(`solve(ci, i)` with `lru_cache`) that tries the maximal/canonical parse
first and falls back through documented deletion rules. Added:
deletable non-initial য/ব-ফলা, deletable ষ after ক্ষ and ঞ after জ্ঞ,
deletable ম in শ্ম/স্ম/হ্ম, ঋ/ৠ emitting r+vowel.

- **Result at this point:** lexicon aligner coverage 89.44% → **99.57%**.
  Also froze `data/tasks/g2p.jsonl`, `syllable_count_word.jsonl`,
  `schwa_deletion.jsonl` (via `scripts/build_tasks.py`) and wrote
  `docs/probing_integration.md` (read-through of the
  `liaodisen/Tokenization-Phonology` fork's probing scripts, documenting
  upstream issues to patch before M4 — a hardcoded HF token and
  Compute-Canada-specific model paths).

### ❌ Critical bug found by manually reading all 258 aligner failures (`e8a90a1`)

Reviewing the residual failures (rather than accepting 99.57% as good
enough) found the dominant failure pattern: **~140 words containing
ড়/ঢ়/য় all failed with "unconsumed phonemes."**

- **Root cause:** `normalize_bn()` composes these three letters into
  single precomposed codepoints (they're Unicode composition
  *exclusions*, so plain NFC leaves them decomposed as C+nukta). But the
  `CONSONANTS` set literal in the source file — typed the normal way, like
  most source text — held the *decomposed* form. So after normalization,
  precomposed ড়/ঢ়/য় were silently not recognized as consonants at all:
  0 consonants counted for that akshara.
- **Fix, attempt 1:** added the precomposed codepoints to `CONSONANTS`.
  Coverage went **down** (99.57% → 98.94%) — a regression I caught by
  re-running ingestion after the "fix" instead of assuming success.
- **Second root cause:** the `YYA` constant used for glide-deletion (য়
  being optionally silent) had the *same* decomposed-vs-precomposed bug,
  independently. Once `CONSONANTS` recognized precomposed য়, the
  previously-dead `ch == YYA` deletion branch activated — but with the
  wrong (decomposed) comparison value, so it silently *never* matched,
  making bare-য় aksharas newly require a full consonant they often don't
  have. Fixed by forcing `YYA` via an explicit `য়` escape.
- **Two more phonological rules added** in the same pass, found via the
  same failure review: visarga ঃ modeled as a coda slot (দুঃখ →
  /d̪ukkʰo/, gemination of the next onset — recovered ~35 words), and
  word-initial হ before ঋ-kar treated as optionally silent (হৃদয় →
  colloquially /rid̪ɔe̯/, not /hrid̪ɔe̯/ — recovered ~19 words).
- **Lesson learned, worth repeating:** any Bangla literal in source code
  involving ড়/ঢ়/য় is a landmine — always verify with
  `chr(0x09DC) in X` rather than trusting how the string *looks* in an
  editor. This bit us twice in the same fix.
- **Final result:** aligner coverage 99.57% → **99.95%** (60,056/60,087).
  Remaining 31 failures are genuinely heterogeneous long-tail items
  (English-loan অ্যা-digraph transliteration, archaic spellings, proper-name
  hiatus) — not chased further, reported as coverage.

### M3 annotation tooling (`e23dcb3`)

Built the human-review infrastructure ahead of actually doing the review:
`scripts/tag_etymology_heuristic.py` (orthography-only candidate tagger —
conjunct presence → tatsama-leaning, অ্যা-digraph/স্ট-স্ক-স্প → foreign-leaning,
else tadbhava), `scripts/build_annotation_sheet.py` (stratified CSV review
sheets), `scripts/apply_annotations.py` (merges `reviewed=TRUE` rows back
into the JSONL task files). Scoped for a **solo annotator** — no
inter-annotator κ anywhere in this project, documented as a limitation
against Spec A.4's 2-annotator design rather than silently claimed.

---

## Task 3a — rhyme pairs (classification)

### Build (`8ef777d`)

Wrote `src/rhyme.py` from scratch: `rime()` (closed final syllable →
nucleus+coda; open final syllable → onset+nucleus, matching Bangla poetic
convention that vowel-only matches are assonance, not rhyme; returns
`None` for onset-less open syllables), `assonance_key()`, `is_trivial_pair()`
(anti-leakage: shared final-2-akshara OR shared inflectional suffix),
`final_matra()` (mines orthographic decoys). `scripts/build_rhyme_dataset.py`
generates 200 positive + 200 negative pairs (negatives: 80 assonance_decoy,
40 ortho_decoy, 80 random), gated behind `--stats` → dry-run-with-sample →
`--confirm`.

### ❌ Two bugs found by manually reading the 40-sample review printout

Per the task brief's explicit instruction to read samples "as a native
speaker" before confirming, not just trust the pipeline ran without errors:

1. **Exact-homophone duplicate**: কন্টাক্ট vs কনট্যাক্ট — two spellings of
   the English loanword "contact," phonemically identical
   (`k ɔ n ʈ æ k ʈ` both). Not a real rhyme pair, it's the same word twice.
   Fix: reject positive candidates where `phonemes1 == phonemes2` in full.
2. **Cross-spelling compound stem-sharing**: খেত ("field") vs নীলক্ষেত
   ("blue-field," a compound literally containing "field") — the
   orthographic substring filter missed it because নীলক্ষেত spells the
   shared /kʰet̪/ morpheme with the conjunct ক্ষ while খেত spells it
   plainly with খ. Fix: added a phoneme-sequence contiguous-sublist stem
   check (≥3 phonemes) alongside the orthographic one.

Regenerated after both fixes, re-reviewed a fresh sample — clean.
**Result:** `data/task3a_rhyme_pairs.jsonl`, 400 pairs, sanity gates green.

### Dataset consolidation (`044a2f1`)

Discovered a second, older, cruder rhyme dataset existed:
`build_tasks.py` had its own inline `rime_of()`/`build_rhyme()` (predating
`src/rhyme.py`, no open/closed-syllable distinction) writing to
`data/tasks/rhyme_pairs.jsonl`. Two competing rhyme datasets in one repo is
a landmine for future confusion. Checked the old review sheet had **zero
rows reviewed** (safe to retire, nothing lost), removed the ad hoc code
from `build_tasks.py`, repointed `export_for_probing.py` at
`data/task3a_rhyme_pairs.jsonl`, renamed the review sheet to
`task3a_rhyme_review.csv`.

### ✅ Human verification (`16b1c68`)

User manually read all 400 pairs. **Zero corrections. 100% agreement.**
Bulk-marked `reviewed=TRUE`, ran `apply_annotations.py`,
`annotation.verified=true` across all 400.

---

## Etymology taxonomy: 3-way → 4-way (`d7d5caa`)

User encountered বোরো ("boro rice") while reviewing and correctly
identified it as neither tatsama nor tadbhava — it's দেশি (deshi),
indigenous substrate vocabulary (often Austroasiatic/Munda-layer) that was
**never** Sanskrit-descended at all, unlike tadbhava which specifically
means "evolved *from* Sanskrit." The original spec's 3-way scheme missed
this real 4th category from traditional Bengali lexicography.

- Presented 3 options (add 4th category / fold into tadbhava / footnote
  only); user chose "add 4th category."
- Updated `BanglaPhonologyBench_Research_Spec.md` A.3.5, the heuristic
  tagger's docstring (deshi is deliberately never auto-guessed — no
  orthographic signature distinguishes it from tadbhava, purely an
  annotator judgment call), `docs/annotation_guide.md`.
- No code validation needed changing — `etym_corrected` was always a free
  string.

## Task 4 — schwa deletion: human verification (`c5b663a`)

User manually read all 160 stratified schwa-vector rows.
**Zero corrections. 100% agreement.** Same bulk-mark → `apply_annotations.py`
flow as rhyme 3a.

## Task 1 — etymology: human verification (`cfa0efd`)

- User edited the CSV directly in `etym_heuristic` (rather than the
  intended `etym_corrected` column) — mechanically harmless since
  `apply_annotations.py` falls back to `etym_heuristic` when
  `etym_corrected` is empty, but it destroyed the ability to compute a
  correction rate from that file alone.
- **10 words flagged for adjudication**: several `deshi` tags looked like
  clear tatsama words by recognizable Sanskrit roots (মানবতাবিরোধী, বংশ,
  পাদুকার, সংস্পর্শে, মিলিত, অংশগ্রহণের, বৃষ্টিস্নাত, সারাবিশ্বের → tatsama;
  খ্রিষ্টাব্দ → foreign, "Christ" root; ভেবেছিল → tadbhava, not deshi).
  User adjudicated all 10; applied as given.
- **❌ Misleading stat caught before it entered the record:**
  `apply_annotations.py` reported "100% agreement" — but that number only
  compares `etym_corrected` against `etym_heuristic`, and since
  `etym_corrected` was empty throughout (see above), it was comparing the
  final answer against itself. Diffed the working file against the
  pre-edit **committed** version of `etym_review.csv` to recover the real
  number: **heuristic accuracy was 38.2% (135/353)** — 218 of 353 words
  were actually corrected. Recorded as the true, citable result instead of
  the spurious 100%.
- **❌ Second near-miss, self-caught:** while fixing an unrelated stale-row
  inconsistency in `etym_review.csv` (1 row had drifted from `g2p.jsonl`'s
  actual value, from an earlier regeneration), ran
  `scripts/build_annotation_sheet.py` to regenerate it — but that script
  regenerates **all three** review sheets at once, and blindly overwrote
  `task3a_rhyme_review.csv`'s confirmed 400/400 `reviewed` marks. Caught
  immediately via a row-count check, restored from git history (it was
  already committed with the marks intact). No data lost. **Lesson:**
  check `git diff` scope before running whole-sheet regeneration scripts
  that touch already-annotated files.
- **Result:** `annotation.verified=true` for 353/3000 g2p.jsonl words;
  final etym distribution across all 3,000: tadbhava 1671, tatsama 1169,
  foreign 145, deshi 15.

---

## Task 3b — rhyme generation (`2aa3af1`)

- Added `success_at_k` / `mean_success_at_k` to `src/rhyme.py`
  (PhonologyBench-style Success Rate@k, ready for M5).
- Built `scripts/build_rhyme_generation_dataset.py`: 300 prompts (200
  common / 100 rare by wordfreq zipf), gold = lexicon words sharing the
  prompt's rime.
- **Methodology concern raised before writing anything:** raw gold sets
  were enormous (median 659, max 7,127) and dominated by short grammatical
  suffixes (-এর genitive, -রা plural) rather than genuine lexical rhyme —
  একের's raw gold of 7,127 words was essentially "any noun + -এর."
  Verified concretely with numbers before deciding: applying 3a's
  `is_trivial_pair` anti-leakage filter to gold-set construction shrank
  একের's gold 7,127 → 20 and সহকর্মীরা's 950 → 23, while a genuine
  derivational rime (ভাতা, -তা) barely moved (639 → 637) — proof the
  filter targets grammatical leakage specifically, not blunt shrinkage.
  User approved applying the filter; documented as a deliberate,
  reasoned deviation from the spec's literal "all lexicon words with
  matching rime" wording (both in the script docstring and a spec note).
- **Performance fix required:** filtering one prompt against a
  7,000+-member rime group pairwise was too slow with `is_trivial_pair`'s
  per-call `segment_aksharas`. Refactored into `trivial_pair_key()`
  (precompute once per word) + `is_trivial_pair_fast()` (O(1) per pair
  after that), tested for exact equivalence with the original.
- **One "bug" that wasn't, caught by testing instead of eyeballing:**
  সুফলও appeared to share the "ও" blocklist suffix with অংশগুলো on visual
  inspection. Direct test showed `is_trivial_pair` correctly returns
  `False` — গুলো ends in the *matra* ো (dependent vowel sign), while the
  blocklisted "ও" is the *independent* vowel (different Unicode
  codepoints, same rendered look). Correct behavior: exactly the
  "rhyme invisible in spelling" case the anti-leakage design exists to
  preserve, not remove. Lesson repeated from the aligner bugs: never trust
  eyeballed Bengali text over a direct codepoint check.
- Flagged, not solved: the spec's "mine additional attested rhymes from
  poetry corpora to enrich gold" stretch goal is **not implemented** — no
  poetry corpus text is present in this repo. Success Rate@k against this
  dataset is a lower bound, documented as such rather than silently
  treated as complete.
- **Result:** `data/task3b_rhyme_generation.jsonl`, 300 prompts, 116 tests
  green.

### ✅ Human verification (same session, not yet its own commit at doc-writing time)

User manually read the 300 prompts + gold sets. **Zero corrections.**
Bulk-marked `annotation.verified=true` across all 300 records.

---

## Repo published to GitHub

Pushed to `https://github.com/LihanCanCode/BanglaPhonologyBench` (all 13
commits, full history). Pre-push safety check: scanned full git history
for secret-shaped strings (only false positives — legitimate Bangla
lexicon entries for "password"/"secret"), confirmed no `.env`/credential
files were ever tracked, checked repo size (16MB, largest file 5.6MB,
comfortably under GitHub limits). `external/` (the third-party probing
fork clone) correctly stays gitignored.

---

## M4 build: probing pipeline + Kaggle notebook

Asked to start M4 with two explicit constraints: the Kaggle notebook must
survive a session restart without redoing finished work, and a GitHub
README was also wanted.

### Pivot away from forking the upstream harness

The plan up to this point (see `docs/probing_integration.md`'s original
version) was to fork `liaodisen/Tokenization-Phonology` and reuse its
`generate_embedding.py` / `train_probe_*.py`. Reconsidered once actually
building it: `generate_embedding.py`'s extraction functions are simple
enough to reimplement directly (load model, one forward pass, take
last-token hidden states per layer), and `train_probe_*.py` are hardcoded
to exactly TWO conditions with English-dataset-specific filenames — not
reusable as-is for our three-way A/M/CB split, would need rewriting most
of the file anyway. Built a self-contained `scripts/kaggle_probing_lib.py`
instead: `extract_g2p` (extraction) + `run_probe_battery` (RidgeCV/
LogisticRegression probe, generalized to N categories with automatic
pairwise one-sided t-tests) — avoids depending on cloning-and-patching a
third-party repo with known bugs (hardcoded HF token, wrong model paths)
for two functions that were easier to just write. Documented as a pivot,
not silently changed, since the earlier M4 plan had already been described
to the user multiple times.

### ❌ Resumability scope, caught by asking rather than assuming

Started implementing ROW-LEVEL checkpointing (save progress every N rows
within a single extraction, resume from a partial mid-file state) —
genuinely more robust in the abstract, but before finishing it, stepped
back and did the actual scale math: each (tokenizer, category) combo is at
most ~3,000 words, a single forward pass per word, so even an 8B model on
a T4 finishes a combo in minutes, not the hours where a session timeout
becomes a real risk. Asked the user directly ("do we need the
resumability??" prompted a genuine reconsideration) rather than shipping
the more-complex version by default. Simplified to **file-level only**:
skip a combo entirely if its output already exists and looks complete —
no partial-state files, no resume-from-row-index machinery. User confirmed
this was the right call.

### Testing what could be tested without a GPU

None of this could be run end-to-end on Kaggle before shipping it (no GPU
in this environment), so correctness had to come from testing what
*could* be tested locally, not from trusting hand-written code:

- `tests/test_kaggle_probing_lib.py` (11 tests, mock model, no GPU/network):
  confirms a completed combo is truly skipped (extract_fn not called
  again), a corrupted checkpoint is detected and redone rather than
  silently trusted, and — the part most likely to have a subtle bug —
  `run_probe_battery` actually recovers a known linear signal (R² > 0.8 on
  a constructed near-deterministic relationship) while correctly failing
  to find one in pure noise (R² < 0.3), plus the control-label condition
  measurably destroys a real signal, and 3-way A/M/CB produces exactly the
  6 expected pairwise comparison keys.
- Ran the full pipeline (real exported CSV → `extract_g2p` →
  `run_probe_battery`) end to end locally with a mock model against
  actual `data/probing_export/g2p_banglat5_{A,CB}.csv` rows — catches any
  real-data shape mismatch (e.g. the pad_len=18 phon_vec) that synthetic
  test data might not expose.
- Every notebook code cell syntax-checked via `ast.parse` after generating
  the `.ipynb` programmatically (a Python script building the JSON, not
  hand-typed — far less error-prone for a format this fiddly).

### Deliverables

- `scripts/kaggle_probing_lib.py`, `tests/test_kaggle_probing_lib.py`
- `notebooks/m4_probing.ipynb`: clone-and-run on Kaggle, BanglaT5 first
  (smallest, ~580M, best smoke test) then TigerLLM/GPT-2/ByT5/Llama-3.1
  (4-bit quantization for the 8-9B models via bitsandbytes), per-model
  error isolation (one model failing doesn't kill the whole run), a
  results-aggregation cell, and explicit Kaggle Save-Version/attach-as-
  input instructions for the resumability workflow.
- `docs/probing_integration.md` rewritten to document the pivot and point
  at the notebook instead of listing M4 as a to-do.
- `README.md`: project overview, metric-triple table, headline findings,
  status table with the human-verification results, repro instructions.
  Did not add a `LICENSE` file unprompted — `pyproject.toml` already
  declares Apache-2.0 for the code; the README states that plus the
  lexicon's CC BY 4.0 rather than making a new licensing decision.
- Rhyme-probe A/M/CB category join (per-word GTAD/STAD join at analysis
  time, per the spec) flagged as NOT built, in both the notebook's final
  cell and `probing_integration.md` — not silently omitted.

---

## M4 rhyme-probe run + M5 baselines and zero-shot harness

Picked up after the M4 rhyme-probe cells (section 5b/6b) had been added to
`m4_probing.ipynb` but never run on real Kaggle GPU.

### Rhyme-probe (Task 3a) run to completion on Kaggle

Ran the notebook's g2p/syllable + rhyme sections for BanglaT5 and TigerLLM
end to end (the only 2 tokenizers with a real 3-way A/M/CB split — the
other 3 registered tokenizers collapse to ~99% CB, so their probes are
skipped by design, not a bug). Output:
`results/probe_pairwise_BanglaT5+TigerLLM.csv` (per-layer, per-task,
per-control-condition one-sided p-values for all 6 pairwise A/M/CB
comparisons) and `results/probe_pairwise_summary.csv`.

**Result is genuinely mixed, not the spec's clean A>M>CB hypothesis:**
computed layer-wise significance rates (fraction of layers where
`p_value < 0.05`, real condition only) directly from the CSV:

| | BanglaT5 | TigerLLM |
|---|---|---|
| G2P | M beats A and CB in ~92% of layers (opposite of A>M) | A beats M and CB in ~50–55% of layers (matches, weakly) |
| Syllable count | M beats A and CB in ~69% of layers | A beats M in 76%, but CB beats M in 57% (partially inverted) |
| Rhyme | CB beats M in 92% of layers (opposite of M>CB) | CB beats M in 71% of layers (also opposite) |

Not noise-shaped (too consistent within a model/task), but not the
predicted ordering either. Left unexplained for now — flagged for the M5
regression step (probe residuals ~ GTAD + STAD + ρ + log-freq + etym) to
test whether category sample-size imbalance (BanglaT5's A/M/CB split is
647/1822/442 words — M is the largest by far) or word-length/frequency
confounds are driving it, rather than treating it as a clean finding.

### M5 baselines (`scripts/build_baselines.py`)

Four naive, non-learned reference points (Spec A.6's "Baselines" row),
each deliberately weak so beating it is a low bar:
- **G2P**: direct grapheme->phoneme table walk over `segment_aksharas`
  clusters, no assimilation/gemination/silent-letter rules (those live
  only in `bangla_phonology.align`, which produced the gold labels and is
  never reused here — that would make the "baseline" the gold labeler).
  17.6% mean PER, 36.2% exact match on the 3,000-word set.
- **Syllable count**: `len(segment_aksharas(word))` — "one syllable per
  akshara." 31.9% exact match, MAE 0.83; undercounts whenever a coda
  cluster spans two aksharas (geminate splits).
- **Schwa deletion**: majority-rule heuristic, the exact one named in the
  spec ("delete word-final, keep the rest") — confirmed empirically
  correct per-environment before hardcoding it (final: 82.2% majority-
  delete; post_conjunct/medial/conjunct: 69–92% majority-keep, all same
  direction). 77.1% per-position accuracy, 59.8% exact-vector match.
- **Rhyme (3a)**: "dictionary rime lookup" — predicts rhyme iff the pair's
  precomputed `rime1 == rime2` (derived from lexicon pronunciation, not
  spelling). **100% accuracy**, expected and not a bug: it's a
  ceiling-style baseline assuming perfect pronunciation knowledge: the
  real M5 question is whether an LLM can recover the same answer from
  spelling alone, without a dictionary.

### M5 zero-shot harness (`scripts/zeroshot_lib.py` + `notebooks/m5_zeroshot.ipynb`)

Built and unit-tested locally (`tests/test_zeroshot_lib.py`, 7 tests, mock
generator, no GPU) — same "test what can be tested before touching a real
model" discipline as M4. One prompt-builder + parser + scorer per task
(g2p, syllable_count, rhyme_awareness, rhyme_generation, schwa_deletion),
Bangla-language prompts by default with an English-ablation switch
(`lang="en"`) per Spec A.6's prompt-language ablation.

**Schwa elicitation design note**: the model is never shown IPA. It's
given the word's schwa-eligible aksharas (the orthographic grapheme
clusters at `schwa_positions` — e.g. পটলের's eligible clusters are
প/ট/র, লে is excluded because it already has an explicit matra) numbered
in order, asked Y/N per position ("is this consonant's default vowel
pronounced"). This is the only elicitation format that doesn't require the
model to already produce correct IPA as a side channel just to answer a
binary-vector question.

**Row-level resumability, unlike M4's file-level-only approach**: M4's
extraction was one forward pass per word (minutes per combo even for an
8B model). Zero-shot generation is autoregressive and far slower per item
— a full run across 5 tasks is thousands of `generate()` calls, a real
multi-hour job with real restart risk. So `m5_zeroshot.ipynb` saves one
JSONL line per generated item (`checkpoints/zeroshot/{task}_{lang}.jsonl`)
and skips ids already present on restart; scoring is a separate, cheap,
re-runnable step over whatever's been generated so far, not fused into
generation. Documented as a deliberate deviation from M4's precedent, with
the reasoning, rather than silently copying a design that doesn't fit
here.

**Status: built, not yet run.** TigerLLM-9B-it only for this first pass
(cost-free, per user direction — more open models next, closed/paid models
later). Next action: upload the notebook to Kaggle, run the
`SAMPLE_SIZE=300` smoke test across all 5 tasks, sanity-check parse rates
and prompt wording against real model output before committing to a full
run.

### Kaggle run #1 (TigerLLM-9B-it): smoke test, notebook UX fixes, g2p + syllable_count at full scale

Ran `m5_zeroshot.ipynb` on Kaggle for real. Two bugs found and fixed before
trusting any numbers, then two tasks finished at full scale — one of them a
result surprising enough to need its own writeup.

**Bug 1 — `generate()` crash on first run**: `AttributeError` inside
`model.generate()` (`BatchEncoding` has no `.shape`) from passing
`input_ids` positionally as a bare tensor — fragile across transformers
versions when `apply_chat_template`'s return value isn't unwrapped
consistently. Fixed by using `return_dict=True` and unpacking the encoding
(`**encoded`) into `generate()` instead (the pattern HF's own docs
recommend). Fixed, pushed, notebook re-pulled on Kaggle, no data lost
(nothing had been generated yet at the point of the crash).

**Notebook UX change — split section 5 into one cell per task (5a-5e)**:
originally one loop fired all 5 tasks' ~6,200 remaining `generate()` calls
back-to-back with nothing to inspect until it finished (hours, for a job
with real restart risk). Split into 5a (g2p) / 5b (syllable_count) / 5c
(rhyme_awareness) / 5d (rhyme_generation) / 5e (schwa_deletion), each
independently runnable, plus a `peek(task)` helper cell to eyeball raw
completions between tasks — catches a bad prompt/parser on a small batch
instead of after committing hours to the full run. Requested by the user
specifically to enable "one at a time" execution; matches this project's
established "read the sample before trusting the pipeline" discipline.

**Gotcha caught mid-run — scoring cell truncates by `SAMPLE_SIZE` same as
generation does**: `score_task()` does `rows = rows[:SAMPLE_SIZE]` before
matching against generated ids, so leaving `SAMPLE_SIZE` at a smaller
value after generation had moved past it silently under-reported `n` in
the scoring table (g2p briefly showed `n=1500` despite 2,341+ completions
existing on disk). Not a data-loss bug — just a reporting trap. Fix:
confirm `SAMPLE_SIZE = None` before trusting any scoring-cell row.

**G2P at full scale (3,000/3,000): PER 0.029, exact match 93.6%.** Holds
up from the 300-item smoke test's 92% — a real, strong, believable result,
dramatically ahead of the naive baseline (PER 0.176, 36.2% exact).

**Syllable count at full scale (3,000/3,000): exact match 1.0, MAE 0.0 —
literally zero errors.** Independently re-verified outside the notebook's
own scorer (not just trusting `run_syllable_count`'s output number):
downloaded the raw checkpoint JSONL, diffed every one of the 3,000
predictions against gold directly (0 mismatches), and confirmed the gold
and predicted syllable-count distributions are bucket-for-bucket identical
(1→166, 2→974, 3→973, 4→614, 5→211, 6→48, 7→11, 8→3 on both sides). A
25-item random sample (fixed seed) spot-checked by eye, including a rare
7-syllable compound (প্রতিদ্বন্দ্বিতাকারী), all correct. This rules out a
scoring/parsing bug — the raw model completions genuinely are correct,
word for word, on all 3,000.

**This is flagged as a result requiring a contamination caveat, not
reported as a clean win.** A literal 3,000/3,000 with zero errors is
itself suspicious in the opposite direction from "the pipeline is broken":
even the *gold labels'* own internal validation only reached 99.86%
self-agreement (see M1's validation numbers), so a model landing on
literally 0 disagreements against a gold set that isn't 100%
self-consistent is unusual for genuine independent zero-shot inference.
Two non-exclusive explanations recorded for the eventual writeup:
(1) genuine competence — Bangla syllable division (অক্ষর বিভাজন) is a
common school-curriculum drill, plausibly well-represented in a
heavily-Bangla-tuned model's training data, and it's a more
pattern-matchable task than full G2P (naive orthographic baseline: 31.9%
exact, so it's not *trivially* solvable from spelling alone, but it may be
learnable from enough examples); (2) contamination — the word list is
derived from a public Google lexicon + `wordfreq`, and if TigerLLM's
training corpus included the same or a similar syllable-annotated Bangla
dictionary, this could be memorized recall rather than zero-shot
reasoning. Not distinguished yet. A follow-up check worth doing before
the numbers go in the paper: test on invented/nonce Bangla-looking words
that cannot appear in any training corpus, and see if perfect accuracy
survives — flagged as a TODO, not yet done.

### Kaggle run #1 complete: all 5 tasks at full scale (TigerLLM-9B-it, bn prompts)

rhyme_awareness (400/400) and schwa_deletion (1,000/1,000, 929 parsed)
finished. Full M5 zero-shot-vs-baseline comparison, TigerLLM-9B-it,
Bangla-language prompts:

| Task | Baseline (`results/baselines_summary.csv`) | TigerLLM zero-shot | Read |
|---|---|---|---|
| G2P | PER 0.176, 36.2% exact | **PER 0.029, 93.6% exact** | Model dramatically beats the naive baseline — solid, credible win. |
| Syllable count | 31.9% exact, MAE 0.83 | **100.0% exact, MAE 0.0** | Perfect — carries the contamination caveat above (nonce-word check still a TODO before this goes in the paper as-is). |
| Rhyme awareness | 100% (ceiling/oracle, not a real floor) | 53.0% acc, F1 0.236 | Barely above chance on the full balanced 400-pair set — confirms the smoke test's "answers না almost unconditionally" finding wasn't a small-sample fluke. |
| Rhyme generation | — (no baseline built yet) | success@5 = 0.21 | Stands alone; dataset is fixed-size at 300, this is the final number. |
| Schwa deletion | 77.1% per-position, 59.8% exact-vector | 52.4% per-position, 31.8% exact-vector | Model is *worse* than the hardcoded majority-rule heuristic — a genuine negative result, confirmed at full scale (929/1,000 parsed, 7.1% unparseable). |

**Headline framing for the writeup**: not "the model knows Bangla
phonology" — it's *uneven* across sub-tasks, which is the more informative
and honest story. TigerLLM is excellent at G2P, plausibly-but-unverified
excellent at syllable counting (contamination caveat pending), and
actively *worse than a one-line heuristic* at schwa deletion, with rhyme
judgment sitting near chance. All 5 tasks are now done for this model —
**M5's first full pass (TigerLLM, bn-only) is complete.**

Not yet done, in priority order: (1) the syllable_count nonce-word
contamination check; (2) English-prompt ablation (`lang="en"`, per Spec
A.6) for the same 5 tasks/same model; (3) a second open model
(Llama-3.1-8B-Instruct, already registered in `TOKENIZER_SPECS`); (4)
closed/paid models, deferred per user direction; (5) human baseline (2
annotators, 100 items/task); (6) the regression step tying probe/zero-shot
performance to GTAD+STAD+ρ+freq+etym, which needs at least one more
model's worth of data to be worth running.

### ρ-ceiling analysis + rhyme-generation dictionary baseline (`scripts/analyze_rho_ceiling.py`, `build_baselines.py` update)

An external AI (Perplexity) was consulted for a post-M5-results action
plan (`docs/perplexity_idea.md`); several of its ideas overlapped with
already-planned Spec C.6 work (the regression step, A/M/CB breakdown) and
two were cheap enough to do immediately with data already in hand — done
here. Others (P-CoT prompting, a cited rule-based schwa algorithm,
feature-vector G2P probes, stem-only rhyme) were deliberately NOT acted
on: they cite sources (a "Jang et al. 2025" P-CoT paper, "Nair 1999 /
Niyogi 2006" schwa rules, specific PhonologyBench/human-baseline
percentages) that were not verified against the actual papers before
this session, and shouldn't be taken as citable facts in the thesis
without that verification — flagged, not silently trusted or acted on.

**ρ-ceiling analysis** (`scripts/analyze_rho_ceiling.py`, reads a task's
raw `{id, raw_output}` completions + `syllable_count_word.jsonl`'s
precomputed `rho` field, no new metric computation needed): bins words by
ρ and compares TigerLLM's syllable-count accuracy against the naive
akshara-count baseline's, per bin. Result, run against the full 3,000-word
completions:

| ρ bin | n | TigerLLM acc | baseline acc |
|---|---|---|---|
| 0.0 (no ceiling) | 2,050 | 1.000 | 0.288 |
| 0.01–0.34 | 337 | 1.000 | 0.401 |
| 0.35–0.99 | 392 | 1.000 | 0.342 |
| 1.0 (fully misaligned) | 221 | 1.000 | 0.443 |

**This sharpens the contamination concern rather than resolving it**:
TigerLLM scores 1.000 even at ρ=1.0 — the 221 words where the true
syllable boundary has NO possible orthographic representation at all.
Genuine spelling-based reasoning should degrade there if it were reasoning
from orthographic pattern alone; it doesn't degrade even slightly. This is
a concrete, falsifiable point for the thesis (stronger than "100% looks
suspicious") and makes the still-pending nonce-word contamination test
more important, not less. Side finding, worth a caveat in the writeup: the
naive baseline's accuracy does NOT track ρ monotonically (worst at ρ=0,
its supposedly "easiest" bin) — because ρ only captures "boundary has no
orthographic image at all," not the separate, more common Bangla pattern
where one syllable's coda consonant cluster spans two aksharas (which
makes the "1 syllable per akshara" baseline overcount, most often exactly
when ρ=0). Don't conflate the two failure modes when writing this up.

**Rhyme-generation dictionary baseline** added to `build_baselines.py`
(`run_rhyme_generation_baseline`): picks any 5 words from the prompt's
precomputed `gold_rhymes` set. Scores 100% success@5 — by construction,
same tautology as the existing 3a rhyme baseline (the "prediction" is
drawn from the gold set itself), not a meaningful floor, but the same
"here's the ceiling a dictionary achieves" reference point 3a's baseline
provides, so TigerLLM's 21% success@5 has something concrete to be
compared against. `results/baselines_summary.csv` now has all 5 rows.

### A/M/CB breakdown + regression, attempt 1: degenerate for syllable_count (`scripts/analyze_regression.py`)

Second follow-up from the external-AI action plan (`docs/ideas2.md`) —
same disclaimer as before: specific numeric claims in that doc (P-CoT's
"52% gains," Nair 1999's "94.66% for Hindi") are unverified against the
actual papers and were NOT acted on; only the two items buildable from
data already in hand were.

Built `scripts/analyze_regression.py`: per-word A/M/CB accuracy breakdown
(reusing `export_for_probing.categorize`, the same fixed Spec C.4
definition M4 uses — not re-derived) + a small hand-rolled IRLS logistic
regression (`correct ~ gtad + stad + rho + log_freq`, standardized
coefficients) since adding `statsmodels` as a new dependency for one
script felt like overkill. **Explicit limitation, not silently
upgraded**: this gives coefficients + McFadden's pseudo-R², NOT
p-values/standard errors — treat as directional/exploratory, redo with
`statsmodels.Logit` before anything in the thesis needs real significance
testing.

Ran against syllable_count (the only task with full clean local data):
**A/M/CB accuracy is 1.000 in all three categories, and the regression is
mathematically degenerate** (intercept diverges to 17.5, all coefficients
≈0, McFadden pseudo-R² negative). Not a bug — the correct consequence of
regressing against an outcome with zero variance (syllable_count is
100% correct on all 2,982 usable words, so there's nothing for
GTAD/STAD/ρ to explain).

**Genuinely useful finding buried in the null result**: TigerLLM is
invariant not just to ρ (previous section) but to its own tokenizer's
GTAD/STAD category too — CB (cluster-broken, the worst category) scores
identically to A (aligned). Worth a specific methodological point in the
thesis: GTAD/STAD's original motivation (Liao & Shi) is about a linear
**probe** reading one frozen last-token hidden state, where token boundary
placement mechanically determines what that vector contains — a
zero-shot **generation** model attends over its whole input regardless of
internal boundary placement, so there's no a priori reason tokenization
misalignment has to bottleneck generation the same way it bottlenecks
probing. These may be genuinely different mechanisms, not the same claim
measured twice — don't conflate M4's probing story with M5's zero-shot
story as if they must agree.

**The regression only becomes informative on a task with real variance in
the outcome** — g2p (93.6%, not saturated), rhyme_awareness (53%), and
schwa_deletion (52.4%) are where this analysis will actually produce a
result. `build_dataset()` originally only implemented `correct` scoring for
syllable_count; g2p scoring (exact match against gold IPA) added next
(same pattern for rhyme_awareness/schwa_deletion once their data exists).

### g2p regression: the real "money plot" (`results/regression_g2p_tigerllm.json`)

User supplied fresh full-scale raw completions for g2p (`results/
g2p_results.csv`, 3,000/3,000) — confirmed clean this time. Ran
`analyze_regression.py g2p` against it:

**A/M/CB breakdown**: A=0.927 (n=697), CB=0.930 (n=1,614), M=0.958
(n=671) — same pattern as syllable_count: nearly flat, CB (the
"worst" category) is NOT worse than A. The discrete A/M/CB split
(M4's probing framework) does not cleanly separate G2P zero-shot
performance.

**But the continuous regression tells a different story**: `correct ~
gtad + stad + rho + log_freq`, standardized coefficients — gtad
coef=−0.338 (higher GTAD → lower accuracy, matching the spec's
hypothesis direction), stad coef=+0.203, rho coef=+0.142, log_freq
coef=−0.229 (counterintuitive — flagged, not explained away; possibly
confounded with word length or etym, not investigated further this
session). McFadden pseudo-R²=0.022 — real but small; these four
predictors explain little of the variance in whether TigerLLM gets a
word's phonemes right.

**Honest framing for the thesis**: the discrete A/M/CB categorization
doesn't predict G2P zero-shot performance, but the continuous GTAD score
has a weak, directionally-correct negative association. Misalignment
matters a little, in the predicted direction, but isn't close to the
dominant driver of correctness — a more nuanced, more defensible claim
than either "tokenization doesn't matter" or "tokenization is everything."
Remember this regression's stated limitation: hand-rolled IRLS, no
p-values/standard errors, directional/exploratory only until redone with
`statsmodels.Logit`.

**Still blocked (at time of writing above)**: rhyme_awareness and
schwa_deletion regressions needed their real full-scale data — user's
first two uploads were checked and were still capped at 300 items each
(max ids `rhyme3a_00299` / `schwa_00299`), the smoke-test-scale copies,
not the true 400/1,000. Resolved same session: user re-uploaded
`results/rhyme_awareness_results.csv` (400/400, verified via max id
`rhyme3a_00399`) and `results/schwa_deletion_results.csv` (1,000/1,000,
verified via line count).

### Regression complete for all 3 non-degenerate tasks — the honest "money plot"

Neither `schwa_deletion.jsonl` nor `task3a_rhyme_pairs.jsonl` carry
precomputed per-word `tokenization`/`rho`/`zipf` fields the way
`g2p.jsonl`/`syllable_count_word.jsonl` do (schwa_deletion was never built
with them; task3a needs a joint two-word prompt's tokenization, not a
single word's — same reason `export_for_probing.py`'s rhyme export
computes fresh rather than reading a precomputed field). Extended
`analyze_regression.py` with `_build_schwa` and `_build_rhyme_awareness`:
both load the real TigerLLM tokenizer locally (network access, tokenizer
files only, no model weights — fast) and compute GTAD/STAD/rho fresh via
the same primitives `compute_metrics.py`/`export_for_probing.py` already
use, plus `wordfreq.zipf_frequency` for log_freq since neither file has it
precomputed either. For rhyme_awareness (a PAIR, not a word), GTAD/STAD
take the max of the two words' values (worst-of-two, same reasoning as
`categorize_pair`), rho is averaged, log_freq is averaged. schwa's
`correct` is whole-word exact-vector-match (stricter than the
per-position accuracy reported earlier — sanity-checked: weighted
A/M/CB average here comes out to ~29.5%, matching the previously-reported
31.8% exact-vector-match from the Kaggle scoring cell).

Full results, all four tasks with real variance (`results/regression_
{task}_tigerllm.json`):

| Task | n | GTAD coef | STAD coef | ρ coef | log_freq coef | McFadden R² |
|---|---|---|---|---|---|---|
| G2P | 2,982 | −0.338 | +0.203 | +0.142 | −0.229 | 0.022 |
| Schwa deletion | 999 | +0.146 | −0.142 | −0.218 | +0.042 | 0.014 |
| Rhyme awareness | 386 | +0.198 | −0.079 | −0.149 | +0.283 | 0.023 |
| Syllable count | 2,982 | degenerate (0 variance — 100% correct throughout) | | | | |

**The honest finding: there is no universal, uniformly-signed GTAD/STAD/ρ
story.** GTAD is negative for G2P (matches the naive "misalignment hurts"
hypothesis) but POSITIVE for both schwa_deletion and rhyme_awareness.
STAD and ρ are negative for schwa/rhyme (matches hypothesis direction) but
POSITIVE for G2P. Every R² is small (0.013–0.023) — these metrics explain
a little of the variance in each task, never a lot, and the *direction*
of the effect flips depending on the task. This directly contradicts the
`docs/ideas2.md` action plan's specific per-task hypotheses ("GTAD should
predict G2P," "STAD/rho should predict syllable count," stated as
near-certain expected findings) — the actual result is messier and more
interesting: **tokenization misalignment metrics have small, task-specific,
not-uniformly-directioned effects on TigerLLM's zero-shot performance**,
a materially different and more defensible claim than "misalignment
predicts difficulty," and worth stating exactly this way rather than
cherry-picking G2P's one clean-looking coefficient. A/M/CB category
breakdowns tell the same story at a coarser grain — never a clean A>M>CB
or even A>CB ordering in any of the four tasks.

Also worth noting for the thesis, since a reviewer will ask: this doesn't
contradict M4's probing story, because it's plausibly measuring a
different mechanism (see the earlier "TigerLLM invariant to A/M/CB
category" section) — a linear probe on one frozen hidden state is
structurally coupled to token boundary placement in a way that
autoregressive generation over the full input isn't.

**Not done**: a rhyme_generation regression (success@5 is not a per-item
binary the same way, would need a different formulation — not attempted
this session, flagged as optional/future work, not blocking). This
regression's stated limitation carries over from the g2p section above:
hand-rolled IRLS, no p-values/standard errors, directional/exploratory
only until redone with `statsmodels.Logit` before anything here goes in
the thesis as a claimed-significant result.

### Result-file cleanup + `scripts/score_zeroshot_results.py`

Repo had accumulated several stale/duplicate zero-shot result files from
the back-and-forth download process this session (300-item smoke-test
copies sitting alongside full-scale re-downloads under different names,
some misplaced in `notebooks/`/`data/tasks/`). Cleaned up:
- Removed the stale 300-item `results/{task}_bn_tigerllm.jsonl` files and
  the misplaced `notebooks/g2p_results.csv`, `notebooks/
  syllable_count_results.csv`, `data/tasks/Schwa_deletion.csv`, `data/
  tasks/zero_shot_summary_tigerllm.csv`.
- Renamed the genuine full-scale downloads to the canonical
  `results/{task}_bn_tigerllm.jsonl` naming (matches what the Kaggle
  notebook itself would write): g2p (3,000), syllable_count (3,000),
  rhyme_awareness (400), rhyme_generation (300), schwa_deletion (1,000).
- Added `scripts/score_zeroshot_results.py`: scores a full set of local
  raw-completion files against `zeroshot_lib`'s scorers without needing
  Kaggle/a model loaded, producing the same `zeroshot_summary_{model}.csv`
  shape the notebook's own section-6 cell writes. Ran it end to end —
  output matches the Kaggle-reported numbers exactly, confirming the
  cleaned-up local files are the real, correct full-scale data (not a
  silently-different copy).

`results/` now holds one clean copy per task, no duplicates, matching what
this document's tables above already report.

### The G2P frequency paradox: partially a length confound

The g2p regression's counterintuitive `log_freq` coefficient (−0.229,
higher frequency → lower accuracy, opposite of PhonologyBench/most
psycholinguistic literature) was checked rather than left unexplained:

- `log_freq` vs `gtad`/`stad`/`rho` correlations are all ~0 (−0.013,
  +0.011, +0.032) — not a collinearity artifact with the tokenization
  metrics.
- `log_freq` vs word length (phoneme count): **−0.29** — real, moderate.
  Rarer words in this dataset skew longer (tatsama compounds, technical
  vocabulary); common words skew short (তুমি, ঘর, মা).
- length vs correctness: **+0.099** — longer words are, if anything,
  slightly easier for exact-match G2P.
- Adding `length` as a 5th regression predictor: `log_freq`'s coefficient
  shrinks from −0.229 to **−0.121** (roughly halves), `length` itself is
  the single strongest predictor in the model (+0.401), and McFadden
  pseudo-R² nearly doubles (0.022 → 0.036).

**Honest conclusion**: partially a length confound, not fully explained
by it — a residual negative frequency effect survives controlling for
length.

**Etymology stratification test (follow-up, same session)** — directly
tests the tadbhava/tatsama story instead of leaving it as a plausible
guess:

| Etymology | n | Accuracy | Mean length | Mean log_freq |
|---|---|---|---|---|
| Tadbhava | 1,660 | 0.926 | 6.11 | 3.11 |
| Tatsama | 1,166 | **0.947** | 8.28 | 3.02 |
| Foreign | 142 | 0.951 | 7.01 | 2.92 |
| Deshi | 14 | 1.000 (n too small to trust) | 6.86 | 2.58 |

Tatsama beats tadbhava (94.7% vs 92.6%) despite being longer and no more
frequent — supports the hypothesis, and it isn't just length riding
along (tatsama is both longer AND more accurate, so length alone doesn't
explain the gap). User flagged a Wikipedia claim during this session that
Bangla orthography is "opaque" for tatsama vs "transparent" for tadbhava
(neither the Wikipedia page nor the underlying claim independently
verified this session) — worth a cautious reconciliation, not a rebuttal:
"opaque" there plausibly means unpredictable *without* Sanskrit/conjunct
knowledge (hard for an unfamiliar human reader), which is a different
axis from "regular/rule-governed once learned" (relevant to what a model
trained on enough tatsama vocabulary can pick up). Tadbhava's exception-
laden colloquial sound-drift may be the harder target for exact-match
transcription precisely because it's less rule-governed, not because it's
more "transparent" in the everyday-reading sense the Wikipedia claim
addresses. Stated as an interpretation to flag in the thesis, not a
settled fact — the two claims may simply be about different things
(reading difficulty for a human vs. rule-learnability for a model).

---

## Current state summary (see `CLAUDE.md` for the authoritative live version)

- **M1–M3: done**, including the full human annotation pass (rhyme 3a
  400/400 100%, rhyme 3b 300/300 100%, schwa 160/160 100%, etymology
  353/353 at a genuine 38.2% heuristic-tagger accuracy).
- **M4 (Kaggle probing)**: g2p/syllable + rhyme probing run to completion
  for BanglaT5 + TigerLLM (the 2 tokenizers with a real A/M/CB split);
  result is a mixed, not-yet-explained deviation from the spec's A>M>CB
  hypothesis (see above). GPT-2/ByT5/Llama-3.1 (expected ~all-CB) not run.
- **M5 (zero-shot evals)**: baselines done, all 5 tasks (`results/
  baselines_summary.csv`). **TigerLLM-9B-it bn-prompt zero-shot run
  COMPLETE, all 5 tasks at full scale, plus the ρ-ceiling analysis and the
  GTAD/STAD/ρ regression for all 4 non-degenerate tasks** (see "Kaggle run
  #1 complete" and the regression sections above for full tables): g2p
  93.6% exact (dramatically beats baseline; regression: GTAD is a weak
  negative predictor, matches hypothesis); syllable_count 100% exact
  (contamination caveat — nonce-word check still not done, don't cite
  unqualified; regression is degenerate, 0 outcome variance); rhyme_awareness
  53% acc / F1 0.236 (near chance, model defaults to "না"; regression:
  GTAD positive, STAD/ρ negative); rhyme_generation success@5=0.21 (no
  regression run, success@5 isn't a per-item binary); schwa_deletion 52.4%
  per-position / 31.8% exact-vector (*worse* than the 77.1%/59.8% naive
  baseline; regression: GTAD positive, STAD/ρ negative). **Core honest
  finding**: no universal, uniformly-signed GTAD/STAD/ρ story across
  tasks — effects are small (R²<0.023 throughout) and flip sign by task,
  contradicting `docs/ideas2.md`'s specific per-task predictions. Not done
  yet: the syllable_count nonce-word contamination check, English-prompt
  ablation, a second open model (Llama-3.1-8B-Instruct next), closed/paid
  models, human baseline.
- **Not done, spec-mentioned stretch goals**: Task 5 (conjunct
  pronunciation, optional per spec), poetry-corpus rhyme enrichment.
- **Test suite**: 142 tests, all green, run with `pytest` from repo root.

## How to pick this back up

1. Read this file top to bottom for context, then skim `CLAUDE.md` for the
   current living-reference state (conventions, file locations, exact
   current milestone status — it may have moved since this log entry was
   last appended to).
2. `git log --oneline` to see if commits exist beyond what's logged here.
3. Check `data/annotation/*.csv` for any sheets with `reviewed` rows that
   haven't been merged (`python scripts/apply_annotations.py` is
   idempotent and safe to re-run).
4. If picking up remaining M4: g2p/syllable + rhyme probing for BanglaT5 +
   TigerLLM is done (`results/probe_pairwise_BanglaT5+TigerLLM.csv`).
   GPT-2/ByT5/Llama-3.1 haven't been run (expected ~all-CB, low priority
   but not zero-value — completeness). `notebooks/m4_probing.ipynb` is
   ready; upload to Kaggle with a GPU accelerator, attach a prior
   `/kaggle/working/checkpoints/` dataset as input and set
   `CHECKPOINT_INPUT` to resume — already-finished combos are skipped
   automatically. See `docs/probing_integration.md` for the design
   rationale.
5. If picking up M5: TigerLLM-9B-it's bn-prompt zero-shot run is DONE, all
   5 tasks at full scale (see the "Kaggle run #1 complete" section above
   for the full comparison table against `results/baselines_summary.csv`
   — g2p and rhyme_generation are solid results; syllable_count's 100%
   needs the contamination-vs-genuine-competence nonce-word check before
   citing it unqualified; rhyme_awareness and schwa_deletion are real,
   confirmed-at-full-scale negative/near-chance results). Next: (a) the
   syllable_count nonce-word check, (b) `LANGS = ["bn", "en"]` in the
   Config cell for the English-prompt ablation (Spec A.6) on the same
   model, (c) a second open model — Llama-3.1-8B-Instruct is already
   registered in `TOKENIZER_SPECS`, add it as a second `MODEL_KEY` pass
   through the same 5a-5e cells. (d) after that, closed/paid models per
   user direction, then human baseline (2 annotators, 100 items/task), then
   the regression step (probe/zero-shot performance ~ GTAD+STAD+ρ+freq+etym).
