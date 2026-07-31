# M3 annotation guide (solo annotator)

This is the practical, scoped-down version of Spec A.3.5/A.4's annotation
pass, sized for one native-Bangla-speaker annotator working through a few
focused sittings rather than a funded multi-annotator lab. Everything below
is work only a human can do — the tooling exists to make it fast, not to
replace your judgment.

## What you're actually deciding

Three separate review sheets in `data/annotation/`, each answering a
different question:

| Sheet | Question you're answering | Rows |
|---|---|---|
| `etym_review.csv` | Is this word tatsama (Sanskrit-origin), tadbhava (native), or foreign (English/Perso-Arabic/Portuguese)? | 353 |
| `rhyme_review.csv` | Do these two words actually rhyme in your pronunciation? | 400 |
| `schwa_review.csv` | Is the inherent-vowel (schwa) pronunciation vector correct for this word? | 160 |

All three were pre-filled by code — a heuristic etymology tagger
(`scripts/tag_etymology_heuristic.py`), the lexicon's rime for rhyme pairs,
and the orthography→phoneme aligner for schwa vectors. **Your job is to
correct, not create.** Where the prediction looks right, leave the
`*_corrected` cell blank — you only type something when you disagree.

## Column convention (same in all three sheets)

- **`reviewed`**: type `TRUE` once you've looked at the row. This is the
  only thing that controls whether a row gets applied — leave it blank on
  anything you skip or aren't sure about, and it's left untouched.
- **`*_corrected`** (`etym_corrected` / `label_corrected` / `vector_corrected`):
  leave **blank** if you agree with the prediction shown. Fill it in only
  to override.
- **`notes`**: freeform. Useful for "unsure, dialectal," "loanword via
  Portuguese not English," etc. — not machine-read, but good for your own
  future reference and the paper's qualitative discussion.

You can do this in Excel, LibreOffice Calc, or Google Sheets — all handle
UTF-8 CSV with Bangla text fine. If you use Google Sheets, import as CSV
(File → Import), and when you're done, File → Download → CSV, overwriting
the same filename in `data/annotation/`.

## Sheet-by-sheet notes

### etym_review.csv

Columns: `id, orth, ipa, freq_bucket, tigerllm_category, etym_heuristic,
etym_corrected, reviewed, notes`.

The heuristic is orthography-only (conjunct presence → tatsama-leaning,
অ্যা-digraph or স্ট/স্ক/স্প clusters → foreign-leaning, everything else →
tadbhava). It will be wrong on:
- Perso-Arabic loans with simple CV(C) shape (looks tadbhava, is actually
  foreign) — e.g. words for everyday objects/administration with Arabic/
  Persian roots.
- Tatsama compounds that happen to have no conjunct.
- Foreign words that happen to contain a conjunct-like cluster coincidentally.

All 53 heuristic-`foreign` words are included (foreign is the rarest class
and most likely to need correction), plus 150 random tatsama + 150 random
tadbhava. `tigerllm_category` (A/M/CB) is shown for context only — it's the
tokenizer-misalignment category from Spec C.4, included so you can see if a
skewed sample of A/M/CB happens to fall out of your corrections (useful for
sanity-checking the cognate-conjecture hypothesis later, not something to
act on now).

### rhyme_review.csv

Columns: `id, word1, word2, ipa1, ipa2, rime1, rime2, subset,
predicted_label, label_corrected, reviewed, notes`.

`predicted_label` is 1 (predicted to rhyme) or 0 (predicted not to rhyme).
`subset` tells you which pool it came from:
- `pos`: mined from the lexicon by matching rime (final-syllable nucleus+coda).
- `hard`: same final-akshara *spelling*, different actual rime — orthographic
  decoys, meant to be hard negatives. Check these particularly carefully;
  they're the ones most likely to trip up a rule-based rhyme detector, which
  is exactly why they're valuable, but also where mining errors are most
  likely.
- `easy`: matched on syllable count, spelling doesn't coincide.

Type `0` or `1` in `label_corrected` only if you disagree with
`predicted_label`.

### schwa_review.csv

Columns: `id, orth, aksharas, schwa_vector, schwa_environments,
vector_corrected, reviewed, notes`.

`schwa_vector` has one bit per eligible akshara (consonant-initial, no
matra of its own — i.e. potential inherent-vowel positions), in the same
order as `aksharas`. `1` = vowel pronounced, `0` = deleted (silent). E.g.
for কলম with aksharas `ক ল ম`, vector `[1, 1, 0]` means /kɔ.lo.m/ — final ম
loses its inherent vowel. `schwa_environments` tags each position as
`final` / `post_conjunct` / `conjunct` / `medial` (Spec A.4 oversamples
`final` and `post_conjunct` since those are where schwa deletion is most
ambiguous — that's why the sample is stratified evenly across all four
environments rather than drawn at natural frequency).

If you disagree with even one bit, retype the **whole** vector
space-separated in `vector_corrected` (e.g. `1 0 1`) — don't try to edit a
single position.

## Applying your edits

Once you've marked some rows `reviewed=TRUE` (you don't have to finish a
whole sheet in one sitting — partial progress is fine, re-run any time):

```
python scripts/apply_annotations.py
```

This merges reviewed rows back into `data/tasks/*.jsonl`, setting
`annotation.verified=true` and `annotation.annotators=1` on every item it
touches, and prints an agreement summary (how often you agreed with the
heuristic/lexicon-derived prediction — a rough precision estimate for that
pipeline stage, worth citing in the paper). Unreviewed rows are left
exactly as-is. Safe to re-run repeatedly as you review more.

## What this does NOT cover (be upfront about these limitations)

- **No inter-annotator agreement / κ** (Spec A.4 Task 1 wants 10%
  dual-annotated with reported phoneme-level kappa) — this project has one
  annotator. Report this as a limitation, not silently claim it was done.
- **Not exhaustive** — 353/3,000 G2P words get an etymology label reviewed
  (~12%), 400/400 rhyme pairs (100%, it was already small), 160/1,000 schwa
  words (~16%). This is deliberately a *quality-audit* sample, not full
  annotation of every item; report coverage numbers explicitly in the
  paper's data section rather than implying every item is human-verified.
- **POS tags** are still null throughout — out of scope for this pass,
  lower priority (only needed for homograph disambiguation, which affects a
  small minority of items).
