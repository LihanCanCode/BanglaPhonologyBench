# BanglaPhonologyBench + GTAD/STAD Analysis — Research Specification (v0.1)

Scope: Pipeline 1 (benchmark construction + performance evaluation) combined with the
misalignment analysis (GTAD + Bangla-STAD) from Pipeline 2. Target venue style: ACL/EMNLP
long paper or KnowLLM/SIGMORPHON workshop.

---

## Part A — Dataset Specification: BanglaPhonologyBench

### A.1 Design principles

1. **Gold labels come from a human-curated pronunciation lexicon wherever possible**, not from
   rule-based G2P. Rule-based Bangla G2P fails precisely on the phenomena we want to test
   (schwa deletion, conjuncts, loanwords), so using it as gold would leak the errors we're measuring.
2. **Every item is stored with three parallel representations**: orthography (NFC-normalized
   Unicode), phonemic form (IPA), and grapheme-cluster segmentation. This lets all downstream
   analyses (frequency splits, tokenization splits, GTAD/STAD splits) run off one file.
3. **Mirror the two source papers' sizes and controls** so results are directly comparable to
   Liao & Shi (2026) and Suvarna et al. (2024).

### A.2 Source resources

| Resource | Use | Notes |
|---|---|---|
| Google `language-resources` bn-BD pronunciation lexicon (Apache-2.0) | Primary gold IPA + syllable counts | Human-curated, built for Bangla TTS; verify current availability and entry count before committing |
| CRBLP / prior-work G2P dictionary (~135K entries, Ayadi et al. line of work) | Secondary gold / coverage extension | License check required; used by the encoder-decoder Bangla G2P paper |
| Bangla Wikipedia + IndicCorp-bn | Word frequency counts | Frequency stratification proxy (analogous to WIMBD/C4 counts in PhonologyBench) |
| Public-domain poetry (Tagore's Sanchayita, Nazrul's Agnibeena, etc.) | Rhyme pair mining, sentence-level items | Copyright-safe; rhyme-rich |
| Your curated GK/NCTB QA corpus (~3,254 entries) | Sentence-level syllable counting; word sampling | Already cleaned with Bangla-character-ratio filtering |
| Epitran `ben-Beng` / bnG2P tools | Candidate generation ONLY | Outputs always human-verified before entering gold |

### A.3 Preprocessing (applies to all tasks)

1. Unicode NFC normalization; strip ZWJ/ZWNJ except where they change rendering of conjuncts
   (keep a flag if removed).
2. Bangla-character ratio filter ≥ 0.9 per word (reuse your datathon regex); drop mixed-script items.
3. Deduplicate on (orthography, POS) — Bangla homographs with different pronunciations
   (e.g., মত /mɔt/ 'opinion' vs. মতো /mɔto/ 'like') are kept as separate entries with POS tags.
4. Frequency annotation: log-frequency from IndicCorp-bn; split at top-quartile = HighFreq,
   bottom-quartile = LowFreq (middle half held out of frequency analyses).
5. Loanword annotation: 3-way tag {tatsama (Sanskrit-origin, conjunct-heavy), tadbhava/native,
   foreign (English/Perso-Arabic/Portuguese borrowings)} — via etymological wordlists +
   annotator adjudication. This replaces Liao & Shi's CogNet analysis with a linguistically
   richer Bangla-specific version.

### A.4 Tasks

#### Task 1 — Grapheme-to-Phoneme (G2P)
- **Item:** single word → IPA transcription.
- **Size:** 3,000 words (1,500 HighFreq / 1,500 LowFreq), sampled from the lexicon ∩ frequency list.
- **Gold:** lexicon IPA; 10% dual-annotated by native speakers, report agreement (phoneme-level κ).
- **Metrics:** PER (Levenshtein over phonemes / reference length) for open-ended eval;
  exact-match accuracy as secondary. For probing: multi-label ridge regression over padded
  phoneme-index vectors (pad length = max phonemes in dataset; expect 12–14 for Bangla vs. 8 in English).
- **Controls:** per-tokenizer whole-word vs. split-word tag; STAD/GTAD values precomputed per model.

#### Task 2 — Syllable Counting
- **2a (word-level):** same 3,000 words; gold count = number of vowel nuclei in the phonemic
  form (Part B algorithm), spot-checked by annotators. Regression probe (ridge) + zero-shot accuracy.
- **2b (sentence-level):** 1,000 sentences (700 simple / 300 complex by clause count via a
  Bangla dependency parser or heuristic conjunction detection), drawn from NCTB corpus +
  poetry lines. Gold = sum of word-level counts after sandhi-aware adjustment (flag and
  manually resolve external sandhi cases).
- **Metric:** exact-count accuracy; report by sentence length buckets (mirrors PhonologyBench Fig. 3).

#### Task 3 — Rhyme (অন্ত্যমিল)
- **3a Rhyme awareness (classification, for probing + zero-shot):** 200 positive + 200 negative pairs.
  - Positive = identical IPA rime from the final stressed nucleus onward (Bangla stress is
    word-initial and non-contrastive; operationalize rhyme as identity of the final syllable's
    nucleus+coda, optionally last two syllables for স্বরান্ত্যমিল-style rich rhyme).
  - **Anti-leakage filter (critical, from Liao & Shi):** exclude pairs sharing the final 2
    grapheme clusters orthographically — forces phonological, not orthographic, matching.
    Bangla makes this genuinely possible: e.g., pairs where different vowel signs yield the
    same rime, or schwa deletion creates rhymes invisible in spelling.
  - Negative pairs: matched on length and frequency; include "orthographic decoys" (same final
    spelling, different pronunciation) as a hard subset.
- **3b Rhyme generation:** 300 prompt words (200 common / 100 rare). Gold = all lexicon words
  with matching rime (expect large gold sets; report Success Rate @5 as in PhonologyBench).
  Mine additional attested rhymes from poetry corpora to enrich gold.

#### Task 4 — Schwa (inherent vowel) deletion prediction  *(Bangla-unique contribution)*
- **Item:** word → binary vector over consonant graphemes marking whether the inherent vowel
  is pronounced. Example: কলম → [1, 1, 0] (/kɔ-lo-m/); ঘর → [1, 0] (/gʱɔ-r/); শান্ত → final
  vowel retained after conjunct (/ʃan-to/).
- **Size:** 1,000 words stratified to over-sample ambiguous environments (word-final position,
  post-conjunct, verb inflections, compound boundaries).
- **Gold:** derived by aligning orthography to lexicon IPA (deterministic once alignment is fixed;
  disagreements → annotation).
- **Metrics:** per-position F1 and whole-word exact match. Both probing (per-position logistic
  probes) and zero-shot prompting.
- Why it matters: this is a pure orthography→phonology inference problem with zero orthographic
  signal — the cleanest possible test of latent phonological knowledge, and it has no English analogue.

#### Task 5 (optional stretch) — Conjunct pronunciation
- 300 words containing high-ambiguity conjuncts (ক্ষ, জ্ঞ, হ্ম, gemination cases);
  item = word → IPA of the conjunct region. Useful qualitative section even if small.

### A.5 Data schema (JSON Lines)

```json
{
  "id": "g2p_00042",
  "task": "g2p",
  "orth": "মানুষ",
  "orth_nfc_codepoints": ["\u09AE", "\u09BE", "\u09A8", "\u09C1", "\u09B7"],
  "grapheme_clusters": ["মা", "নু", "ষ"],
  "ipa": "manuʃ",
  "phonemes": ["m", "a", "n", "u", "ʃ"],
  "syllables_phonemic": [["m","a"], ["n","u","ʃ"]],
  "syllable_count": 2,
  "schwa_vector": null,
  "freq_bucket": "high",
  "etym": "tadbhava",
  "pos": "NOUN",
  "tokenization": {
    "llama3": {"tokens": ["<illustrative>"], "gtad": 0.0, "stad": 0.5},
    "tigerllm": {"tokens": ["<illustrative>"], "gtad": 0.0, "stad": 0.0}
  },
  "annotation": {"source": "google_lexicon", "verified": true, "annotators": 2}
}
```

Per-model tokenization fields are generated by a script, not stored by hand; keep the raw file
tokenizer-agnostic and materialize per-model views.

### A.6 Evaluation matrix

- **Models (performance-based, zero-shot):** GPT-4o-class, Claude, Gemini (closed);
  Llama-3.1-8B-Instruct, TigerLLM-9B, BanglaLLaMA / titulm-class Bangla models, ByT5 (open).
- **Models (probing):** open models only — extract hidden states at 0/20/40/60/80/100% depth,
  last-token representation, linear probes, 10 seeds, 80/20 split, random-embedding control
  (exactly Liao & Shi's protocol for comparability).
- **Baselines:** rule-based Bangla G2P; vowel-nucleus counter (syllables); dictionary rime
  lookup (rhyme); majority-rule schwa heuristic ("delete word-final, keep post-conjunct");
  human baseline (2 native annotators, 100 items/task, report cost).
- **Prompting:** zero-shot, Bangla-language prompts, with English-prompt ablation
  (prompt-language effect is itself a reportable finding).

---

## Part B — Bangla Syllabifier

Two layers: an **orthographic akshara segmenter** (feeds GTAD and gives a fallback syllable
proxy) and a **phonemic syllabifier** (gives gold syllables for STAD and Task 2).
Syllabification runs on the **phonemic form**, after schwa deletion is resolved by the lexicon.

### B.1 Akshara (orthographic syllable / grapheme-cluster) segmentation

Character classes (Unicode Bengali block U+0980–U+09FF):
- `C`  = consonant letters (ক–হ, ড়, ঢ়, য়) + khanda-ta ৎ (coda-only, always cluster-final)
- `V`  = independent vowels (অ–ঔ)
- `M`  = dependent vowel signs / matras (া ি ী ু ূ ৃ ে ৈ ো ৌ)
- `H`  = virama/hasanta (্)
- `D`  = diacritics: anusvara ং, visarga ঃ, candrabindu ঁ

Akshara grammar (regex over classes, longest match, left to right):

```
AKSHARA := ( C (H C)* M? D? )   # consonant core with optional conjunct chain, matra, diacritic
         | ( V D? )              # independent vowel
         | ( C H )$              # word-final hasanta-killed consonant (rare, e.g., loans)
```

Rules and edge cases:
1. A `C H C` sequence is ONE akshara (conjunct), recursively: ন্ত্র = ন+্+ত+্+র → one cluster.
2. ZWNJ after H blocks conjunct formation → split into two aksharas; ZWJ requests joined
   rendering → keep as one. Record the flag either way.
3. ৎ and D never begin a cluster; attach to the preceding akshara.
4. This grammar is equivalent to Unicode extended grapheme clusters for Bengali except that
   UAX#29 treats `C H | C` boundaries permissively; our grammar is the linguistically correct
   akshara and is what GTAD uses as ground truth. Implement directly (≈40 lines of Python)
   rather than trusting `\X` regex behavior.

### B.2 Phonemic syllabification

Input: phoneme sequence from the lexicon (schwa deletion already applied).
Bangla syllable canon: **(C)(C)V(C)(C)** with complex onsets essentially restricted to
tatsama clusters and loans; native vocabulary is overwhelmingly CV(C).

**Step 1 — Nucleus marking.** Each vowel is a nucleus. Diphthongs count as ONE nucleus.
Bangla diphthong inventory (treat as units): /oi̯ ou̯ ai̯ au̯ eu̯ æe̯ oe̯ …/ — operationally,
any V + glide-offglide (/i̯/, /u̯/, orthographic ই/উ/য়-offglides) pair in the lexicon
transcription. Nasalized vowels (from ঁ) are single nuclei; anusvara ং contributes /ŋ/ as a
**coda consonant**, not nasalization.

**Step 2 — Onset maximization with a legality whitelist.**
Assign intervocalic consonants to the following onset only if the resulting cluster is in
the legal-onset set O; otherwise split them (earlier consonants → coda of previous syllable).

Legal onset clusters O (beyond single C):
```
/pr br tr dr kr gr sr ʃr mr  pl bl kl gl  
 pj bj tj dj kj gj mj nj lj  (C+j from য-ফলা)  
 sp st sk  (loanwords: স্টেশন, স্কুল — often with vowel prothesis in speech;
            follow the lexicon transcription, don't force)  
 tw dw ʃw  (C+w from ব-ফলা, mostly realized as gemination — again, follow lexicon)/
```
Practical note: ব-ফলা and য-ফলা frequently surface as gemination or vowel change rather than
true clusters (বিশ্ব /biʃʃo/). Since we syllabify the *lexicon transcription*, these resolve
automatically — the whitelist only arbitrates genuinely ambiguous intervocalic strings.

**Step 3 — Coda assignment.** Remaining consonants attach leftward. Legal codas: any single C;
cluster codas only in unassimilated loans (/rd/, /st/); geminates split across the syllable
boundary (/biʃ.ʃo/, /an.na/).

**Step 4 — Boundary vector.** Output syllable boundaries projected back to (a) phoneme
positions, and (b) grapheme-cluster gaps via the orthography–phonology alignment
(Part C.4), which is what STAD consumes.

Pseudocode:

```python
def syllabify(phonemes, onsets=LEGAL_ONSETS):
    nuclei = [i for i, p in enumerate(phonemes) if is_nucleus(p)]  # diphthongs pre-merged
    bounds = []
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = phonemes[a+1:b]              # consonants between two nuclei
        k = len(cluster)
        while k > 0 and tuple(cluster[-k:]) not in onsets:
            k -= 1                             # maximize onset, fall back to legality
        bounds.append(b - k)                   # boundary index before onset
    return split_at(phonemes, bounds)
```

**Validation plan:** run on the full lexicon; compare nucleus counts against lexicon syllable
fields where available; hand-check 200 stratified words (conjunct-heavy, loans, diphthongs);
report syllabifier accuracy in the paper's appendix (this is your analogue of their
`syllabify` toolkit citation, but you own it).

---

## Part C — Formal write-up: GTAD, and Bangla-STAD

### C.1 Setup and notation

Let a word w consist of Unicode codepoints c₁…c_N. Let the akshara segmentation (B.1) induce
grapheme clusters g₁…g_m, defining the set of **legal inter-cluster gap positions**

  B_G(w) ⊆ {1, …, N−1},  |B_G| = m − 1,

where position i denotes the gap between c_i and c_{i+1}.

A tokenizer T maps w (as UTF-8 bytes) to tokens t₁…t_k. Project token boundaries back to
codepoint gap positions, giving the multiset B_T(w). Two pathologies arise that are impossible
in ASCII English:

- **Byte-internal splits:** a boundary inside one codepoint's multi-byte UTF-8 encoding
  (every Bengali codepoint is 3 bytes). Map such a boundary to the *containing codepoint's*
  interior; it is by definition ∉ B_G.
- **Cluster-internal splits:** a boundary at a codepoint gap that lies inside an akshara
  (e.g., between a consonant and its matra, or inside a conjunct chain).

### C.2 Grapheme-cluster–Tokenization Alignment Distance (GTAD)

**Definition (boundary form).**

  GTAD(w; T) = |{ b ∈ B_T(w) : b ∉ B_G(w) }| / |B_T(w)|,  with GTAD = 0 when B_T = ∅.

GTAD ∈ [0,1] is the fraction of tokenizer-induced internal boundaries that violate akshara
integrity. GTAD ≡ 0 for alphabetic scripts under any codepoint-respecting tokenizer, so GTAD
isolates the abugida/byte-BPE-specific misalignment axis absent from Liao & Shi.

**Diagnostic decomposition.** Report the violation rate split into
GTAD = GTAD_byte + GTAD_matra + GTAD_conjunct, attributing each illegal boundary to
(i) byte-internal, (ii) between consonant and dependent sign (M/D), (iii) inside a C-H-C chain.
This tells you *what kind* of damage each tokenizer does, not just how much.

**Token-form variant (secondary).**
GTAD_tok(w;T) = (# tokens whose span is not a concatenation of whole aksharas) / k.
Use the boundary form as primary (it composes with STAD, below); report GTAD_tok in appendix.

### C.3 Bangla-STAD: cluster-level syllabification–tokenization alignment

Liao & Shi define STAD over inter-character gaps. Characters are the wrong unit for an abugida
(a matra is a character but never a phonological unit), so we lift the definition to
**akshara gaps**.

Represent the m−1 inter-cluster gaps as binary vectors:

  v_syl = [s₁,…,s_{m−1}],  s_i = 1 iff a phonemic syllable boundary maps to gap i
  v_tok = [b₁,…,b_{m−1}],  b_i = 1 iff some tokenizer boundary lands exactly at gap i

**Definition.**

  STAD_bn(w; T) = ( Σᵢ |s_i − b_i| ) / (m − 1),  defined for m ≥ 2.

This reduces exactly to Liao & Shi's STAD when every akshara is one codepoint (i.e., for
English), making cross-paper comparison legitimate.

**The mapping from phonemic syllables to orthographic gaps** (needed for v_syl) uses a
monotone alignment between grapheme clusters and phoneme spans: each akshara emits a
contiguous phoneme span (consonant(s) + inherent-or-marked vowel, possibly empty vowel under
schwa deletion; anusvara emits /ŋ/ into the preceding span's coda). Build this with a small
finite-state aligner over the lexicon; ambiguous alignments (<2% expected) go to annotation.
A phonemic boundary maps to gap i iff it falls between the spans of gᵢ and gᵢ₊₁; a boundary
that falls *inside* one akshara's span (possible: CVC akshara like ষ in মানুষ closing the
previous syllable — it can't, single cluster; but e.g. অংক /ɔŋ.ko/ where ং coda and ক onset
sit in adjacent clusters — fine; true intra-akshara syllable breaks occur with conjuncts like
শান্ত /ʃan.to/ where ন্ত spans the boundary) is recorded in a residual term:

  ρ(w) = (# phonemic syllable boundaries with no legal gap image) / (# syllable boundaries).

ρ measures how much of Bangla's phonology is *unrepresentable* by any akshara-respecting
tokenizer — a property of the script itself, and a genuinely novel descriptive statistic:
conjunct-heavy tatsama words will have high ρ.

### C.4 Composite and hierarchy

Total misalignment decomposes hierarchically:

  bytes ⊂ codepoints ⊂ aksharas ⊂ syllables

- GTAD: violations of level 3 by boundaries at levels 1–2.
- STAD_bn: mismatch between levels 3-respecting token boundaries and level 4.
- ρ: intrinsic level-3/level-4 incompatibility of the script.

We deliberately do **not** collapse these into one scalar; the paper reports the triple
(GTAD, STAD_bn, ρ) per word/model. For split-based analyses, define:

- **Aligned (A):** GTAD = 0 ∧ STAD_bn = 0
- **Cluster-broken (CB):** GTAD > 0    (a category with no English analogue)
- **Syllable-misaligned (M):** GTAD = 0 ∧ STAD_bn > 0.25   (threshold mirrors Liao & Shi)

Predictions to test: probe R² and zero-shot accuracy order A > M > CB on G2P and syllable
counting; CB is the novel, expected-worst group; loanword/foreign words over-represented in
M and CB (Bangla replication of the cognate conjecture); ρ correlates with schwa-deletion
task difficulty.

### C.5 Worked example (illustrative segmentations — recompute with real tokenizers before publishing)

Word: শান্ত "calm", IPA /ʃan.to/.
- Codepoints: শ া ন ্ ত  (N = 5)
- Aksharas: [শা][ন্ত]  (m = 2; B_G = {2}: the gap after া)
- Phonemic syllables: /ʃan/ + /to/ → the boundary falls *inside* the akshara ন্ত ⇒ this word
  contributes to ρ; v_syl = [0] at the single legal gap.
- Suppose tokenizer T1 splits bytes as [শা][ন্ত] → B_T = {2} ⊆ B_G ⇒ GTAD = 0;
  v_tok = [1]; STAD_bn = |0−1|/1 = 1.0 (token boundary where no syllable boundary can live).
- Suppose tokenizer T2 emits [শ][ান][্ত] with a boundary between ন and ্ ⇒ that boundary
  ∉ B_G ⇒ GTAD = 1/2, category CB.

The example shows all three quantities doing independent work — exactly the argument for the
triple over a single scalar.

### C.6 Analysis plan (ties Part A to Part C)

1. **Descriptive:** distribution of (GTAD, STAD_bn, ρ) across tokenizers (Llama-3, TigerLLM,
   GPT-4o/tiktoken, Gemma, ByT5-as-control) on the 3,000-word G2P set; violation-type
   decomposition (C.2); by etymology class.
2. **Probing:** A vs. M vs. CB probe performance per layer per task, one-sided t-tests with
   the same significance reporting as Liao & Shi Table 3; random-embedding controls.
3. **Performance:** zero-shot accuracy/PER on the same splits (their Fig. 3 analogue).
4. **Delimiter intervention:** re-run rhyme probing with akshara-delimited input
   (শা/ন্ত style) vs. codepoint-delimited vs. original — the abugida version of their
   slash experiment, with the added question of *which* granularity of splitting helps.
5. **Regression:** per-word probe residuals ~ GTAD + STAD_bn + ρ + log-freq + etym class,
   to show misalignment predicts difficulty beyond frequency.

---

## Part D — Milestones & division of labor (suggested)

1. **M1 (wks 1–3):** Lexicon acquisition + licensing audit; akshara segmenter + phonemic
   syllabifier implemented and validated (B.4 validation numbers in hand).
2. **M2 (wks 3–6):** Tasks 1–2 datasets frozen; GTAD/STAD/ρ computed for all tokenizers;
   descriptive stats (first figures of the paper).
3. **M3 (wks 5–8):** Tasks 3–4 built (rhyme mining + schwa alignment); annotation pass.
4. **M4 (wks 7–10):** Probing harness (fork liaodisen/Tokenization-Phonology, swap data
   loaders); hidden-state extraction on Kaggle T4s (batch + fp16; 8B models fit for forward
   passes with 4-bit if needed).
5. **M5 (wks 9–12):** Zero-shot evaluations (closed + open models); human baselines;
   regression analysis; writing.

Risks: lexicon coverage of rare words (mitigate: annotate the gap set); orthography–phoneme
aligner ambiguity (mitigate: restrict Tasks to aligner-confident items, report coverage);
closed-model API cost (mitigate: cap at ~1k items/task for closed models, full sets for open).
