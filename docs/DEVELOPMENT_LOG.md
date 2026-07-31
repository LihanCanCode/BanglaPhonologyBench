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

## Current state summary (see `CLAUDE.md` for the authoritative live version)

- **M1–M3: done**, including the full human annotation pass (rhyme 3a
  400/400 100%, rhyme 3b 300/300 100%, schwa 160/160 100%, etymology
  353/353 at a genuine 38.2% heuristic-tagger accuracy).
- **M4 (Kaggle probing)**: prepped (`docs/probing_integration.md`,
  `scripts/export_for_probing.py`), not started — needs GPU.
- **M5 (zero-shot evals)**: not started.
- **Not done, spec-mentioned stretch goals**: Task 5 (conjunct
  pronunciation, optional per spec), poetry-corpus rhyme enrichment.
- **Test suite**: 116 tests, all green, run with `pytest` from repo root.

## How to pick this back up

1. Read this file top to bottom for context, then skim `CLAUDE.md` for the
   current living-reference state (conventions, file locations, exact
   current milestone status — it may have moved since this log entry was
   last appended to).
2. `git log --oneline` to see if commits exist beyond what's logged here.
3. Check `data/annotation/*.csv` for any sheets with `reviewed` rows that
   haven't been merged (`python scripts/apply_annotations.py` is
   idempotent and safe to re-run).
4. If picking up M4: start with `docs/probing_integration.md`, run
   BanglaT5 first on Kaggle (smallest model, ~580M, good pipeline smoke
   test before spending quota on TigerLLM-9B/Llama-3.1-8B).
