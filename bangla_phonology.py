# -*- coding: utf-8 -*-
"""
bangla_phonology.py — Reference implementation for BanglaPhonologyBench (Spec v0.1)

Components
  1. Akshara (orthographic grapheme-cluster) segmenter        [Spec B.1]
  2. Phonemic syllabifier (nucleus / onset-max / coda)        [Spec B.2]
  3. Orthography->phoneme monotone aligner (heuristic)        [Spec C.4]
  4. Metrics: GTAD (+ decomposition), STAD_bn, rho            [Spec C.2-C.4]

Tokenizers plug in as byte-span lists, so any HF/tiktoken tokenizer can be
adapted with ~5 lines (see token_byte_spans_from_strings).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict

# ----------------------------------------------------------------------------
# 1. Unicode classes & akshara segmentation  (Spec B.1)
# ----------------------------------------------------------------------------

CONSONANTS = set("কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়")
KHANDA_TA = "\u09ce"          # ৎ  (coda-only)
INDEP_VOWELS = set("অআইঈউঊঋএঐওঔ")
MATRAS = set("\u09be\u09bf\u09c0\u09c1\u09c2\u09c3\u09c7\u09c8\u09cb\u09cc")  # া ি ী ু ূ ৃ ে ৈ ো ৌ
HASANTA = "\u09cd"            # ্
ANUSVARA = "\u0982"           # ং  -> coda /ŋ/
VISARGA = "\u0983"            # ঃ
CANDRABINDU = "\u0981"        # ঁ  -> vowel nasalization
DIACRITICS = {ANUSVARA, VISARGA, CANDRABINDU}
ZWJ, ZWNJ = "\u200d", "\u200c"


def segment_aksharas(word: str) -> List[str]:
    """Segment an NFC-normalized Bangla word into aksharas per Spec B.1.

    AKSHARA := C (H C)* M? D?   |   V D?   |   trailing C H
    ZWNJ after hasanta blocks conjunct formation (splits the cluster).
    khanda-ta and diacritics attach to the preceding akshara.
    """
    clusters: List[str] = []
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in CONSONANTS:
            j = i + 1
            # conjunct chain: (H [ZWJ]? C)*   — H+ZWNJ blocks joining
            while (j < n and word[j] == HASANTA):
                k = j + 1
                if k < n and word[k] == ZWNJ:            # explicit break
                    j = k + 1
                    break
                if k < n and word[k] == ZWJ:
                    k += 1
                if k < n and word[k] in CONSONANTS:
                    j = k + 1
                else:                                     # word-final hasanta
                    j = k
                    break
            if j <= n and j - 1 >= i and j <= n and (j < n and word[j] in MATRAS):
                j += 1
            while j < n and word[j] in DIACRITICS:
                j += 1
            clusters.append(word[i:j])
            i = j
        elif ch in INDEP_VOWELS:
            j = i + 1
            while j < n and word[j] in DIACRITICS:
                j += 1
            clusters.append(word[i:j])
            i = j
        elif ch == KHANDA_TA or ch in DIACRITICS or ch in MATRAS or ch == HASANTA:
            # stray combining mark / khanda-ta: attach to previous cluster
            if clusters:
                clusters[-1] += ch
            else:
                clusters.append(ch)
            i += 1
        else:
            # non-Bengali char: its own cluster (flag upstream via filters)
            clusters.append(ch)
            i += 1
    return clusters


def legal_gap_positions(word: str) -> Set[int]:
    """Codepoint-gap positions (1..N-1) that lie BETWEEN aksharas."""
    gaps, pos = set(), 0
    clusters = segment_aksharas(word)
    for cl in clusters[:-1]:
        pos += len(cl)
        gaps.add(pos)
    return gaps


# ----------------------------------------------------------------------------
# 2. Phonemic syllabifier  (Spec B.2)
# ----------------------------------------------------------------------------

VOWEL_BASES = set("aeiouæɔə")

def is_nucleus(p: str) -> bool:
    """Vowel phoneme? Handles nasalized (ã, precomposed or combining) & diphthongs
    pre-merged as one symbol (oi̯)."""
    import unicodedata as _ud
    base = _ud.normalize("NFD", p).replace("\u0303", "").replace("\u032f", "")
    return len(base) > 0 and base[0] in VOWEL_BASES

# Legal onset clusters beyond a single consonant (Spec B.2 whitelist).
LEGAL_ONSETS: Set[Tuple[str, ...]] = set()
for c1 in ["p", "b", "t", "t̪", "d", "d̪", "k", "g", "s", "ʃ", "m", "n", "l"]:
    for c2 in ["r", "l", "j", "w"]:
        LEGAL_ONSETS.add((c1, c2))
LEGAL_ONSETS |= {("s", "p"), ("s", "t"), ("s", "ʈ"), ("s", "k"), ("ʃ", "r"), ("s", "r")}
LEGAL_ONSETS -= {("t̪", "l"), ("d̪", "l")}   # not attested


def syllabify(phonemes: List[str],
              onsets: Set[Tuple[str, ...]] = LEGAL_ONSETS) -> List[List[str]]:
    """Nucleus marking -> onset maximization (whitelist) -> coda assignment.
    Geminates split across the boundary automatically (CC not in onset list
    unless whitelisted; identical CC never is)."""
    nuclei = [i for i, p in enumerate(phonemes) if is_nucleus(p)]
    if not nuclei:
        return [phonemes[:]] if phonemes else []
    bounds = []
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = phonemes[a + 1: b]
        k = len(cluster)
        if k > 1:
            k_try = k
            while k_try > 1 and tuple(cluster[-k_try:]) not in onsets:
                k_try -= 1
            k = k_try
        bounds.append(b - k)
    out, prev = [], 0
    for b in bounds:
        out.append(phonemes[prev:b]); prev = b
    out.append(phonemes[prev:])
    return out


# ----------------------------------------------------------------------------
# 3. Orthography -> phoneme monotone aligner  (Spec C.4, greedy heuristic)
# ----------------------------------------------------------------------------

RI_KAR = "\u09c3"   # ৃ  emits /ri/: one consonant + one vowel

def _expected_consonants(akshara: str) -> Tuple[int, int]:
    """(core, coda) consonant counts for one akshara.

    core: consonants realized BEFORE the nucleus (base/conjunct consonants,
          plus the /r/ of ri-kar ৃ).
    coda: consonants realized AFTER the nucleus — anusvara ং (/ŋ/) and
          khanda-ta ৎ (/t̪/) are phonemically syllable codas (Spec C.4), so
          the phoneme order inside the akshara is C* V coda*.
    """
    core = sum(1 for ch in akshara if ch in CONSONANTS)
    core += sum(1 for ch in akshara if ch == RI_KAR)   # ৃ emits /r/ + vowel
    coda = sum(1 for ch in akshara if ch in (ANUSVARA, KHANDA_TA))
    return core, coda

def _vowel_initial(akshara: str) -> bool:
    return len(akshara) > 0 and akshara[0] in INDEP_VOWELS


@dataclass
class Alignment:
    spans: List[Tuple[int, int]]          # phoneme [start,end) per akshara
    ok: bool
    note: str = ""


def align(word: str, phonemes: List[str]) -> Alignment:
    """Greedy monotone alignment: each akshara consumes its expected consonant
    count, then at most one following vowel (the akshara's matra/inherent vowel,
    possibly deleted).  Ambiguities are resolved by lookahead on the next
    akshara's type; failures are flagged, not silently accepted (Spec C.4:
    aligner-uncertain items go to annotation)."""
    clusters = segment_aksharas(word)
    spans: List[Tuple[int, int]] = []
    i = 0
    for ci, cl in enumerate(clusters):
        start = i
        need_core, need_coda = _expected_consonants(cl)
        got_c = 0
        if _vowel_initial(cl):
            # V (D)* akshara: vowel phoneme first, then coda consonants (ং, ৎ)
            if i < len(phonemes) and is_nucleus(phonemes[i]):
                i += 1
            else:
                return Alignment(spans, False,
                                 f"nucleus expected for independent vowel @akshara {ci} ({cl})")
            while i < len(phonemes) and got_c < need_coda and not is_nucleus(phonemes[i]):
                i += 1; got_c += 1
            spans.append((start, i))
            continue
        while i < len(phonemes) and got_c < need_core:
            if is_nucleus(phonemes[i]):
                return Alignment(spans, False,
                                 f"vowel where consonant expected @akshara {ci} ({cl})")
            i += 1; got_c += 1
        takes_vowel = False
        if i < len(phonemes) and is_nucleus(phonemes[i]):
            nxt = clusters[ci + 1] if ci + 1 < len(clusters) else None
            if nxt is None or not _vowel_initial(nxt):
                takes_vowel = True                # vowel must belong to this akshara
            # else: following independent vowel claims the nucleus
        if takes_vowel:
            i += 1
        # coda consonants (ং /ŋ/, ৎ /t̪/) come AFTER the nucleus (Spec C.4)
        got_c = 0
        while i < len(phonemes) and got_c < need_coda and not is_nucleus(phonemes[i]):
            i += 1; got_c += 1
        spans.append((start, i))
    ok = (i == len(phonemes))
    note = "" if ok else f"unconsumed phonemes: {phonemes[i:]}"
    return Alignment(spans, ok, note)


# ----------------------------------------------------------------------------
# 4. Metrics: GTAD, STAD_bn, rho  (Spec C.2 - C.4)
# ----------------------------------------------------------------------------

def token_byte_spans_from_strings(tokens: List[str]) -> List[Tuple[int, int]]:
    """Adapter: token strings -> byte [start,end) spans of the concatenation."""
    spans, pos = [], 0
    for t in tokens:
        b = len(t.encode("utf-8"))
        spans.append((pos, pos + b)); pos += b
    return spans


@dataclass
class GTADResult:
    gtad: float
    n_boundaries: int
    byte_internal: int
    matra_split: int          # boundary between consonant and dependent sign
    conjunct_split: int       # boundary inside C-H-C chain
    detail: List[str] = field(default_factory=list)


def _codepoint_byte_offsets(word: str) -> List[int]:
    offs, pos = [0], 0
    for ch in word:
        pos += len(ch.encode("utf-8")); offs.append(pos)
    return offs


def gtad(word: str, token_byte_spans: List[Tuple[int, int]]) -> GTADResult:
    """Boundary-form GTAD with violation-type decomposition (Spec C.2)."""
    offs = _codepoint_byte_offsets(word)          # legal codepoint byte boundaries
    legal_cp_gaps = legal_gap_positions(word)     # akshara gaps in codepoint units
    internal = [e for (s, e) in token_byte_spans[:-1]]  # internal boundaries only
    n = len(internal)
    if n == 0:
        return GTADResult(0.0, 0, 0, 0, 0)
    byte_i = matra_i = conj_i = 0
    detail = []
    for b in internal:
        if b not in offs:
            byte_i += 1; detail.append(f"byte-internal @byte {b}"); continue
        cp_gap = offs.index(b)                    # gap index in codepoints
        if cp_gap in legal_cp_gaps or cp_gap == 0 or cp_gap == len(word):
            continue                              # legal akshara boundary
        left, right = word[cp_gap - 1], word[cp_gap]
        if right in MATRAS or right in DIACRITICS:
            matra_i += 1; detail.append(f"matra-split before '{right}' @cp {cp_gap}")
        elif left == HASANTA or right == HASANTA or (left in CONSONANTS and word[cp_gap-1:cp_gap+1]):
            conj_i += 1; detail.append(f"conjunct-split @cp {cp_gap} ('{left}|{right}')")
        else:
            conj_i += 1; detail.append(f"cluster-internal @cp {cp_gap}")
    viol = byte_i + matra_i + conj_i
    return GTADResult(viol / n, n, byte_i, matra_i, conj_i, detail)


@dataclass
class STADResult:
    stad: Optional[float]     # None if m < 2
    rho: float                # unrepresentable syllable boundaries / all boundaries
    v_syl: List[int]
    v_tok: List[int]
    m: int


def stad_bn(word: str, phonemes: List[str],
            token_byte_spans: List[Tuple[int, int]],
            syllables: Optional[List[List[str]]] = None) -> STADResult:
    """Cluster-level STAD + rho (Spec C.3). Token boundaries not on akshara gaps
    are handled by GTAD and simply don't set bits in v_tok."""
    clusters = segment_aksharas(word)
    m = len(clusters)
    syl = syllables if syllables is not None else syllabify(phonemes)
    aln = align(word, phonemes)

    # v_syl over m-1 akshara gaps
    v_syl = [0] * max(m - 1, 0)
    n_bounds = max(len(syl) - 1, 0)
    unrep = 0
    if aln.ok and n_bounds:
        # phoneme index of each syllable boundary
        pidx, acc = [], 0
        for s in syl[:-1]:
            acc += len(s); pidx.append(acc)
        # akshara gap g corresponds to phoneme position spans[g-? ] end
        gap_phoneme_pos = {g + 1: aln.spans[g][1] for g in range(m - 1)}  # gap after cluster g
        pos_to_gap = {v: k for k, v in gap_phoneme_pos.items()}
        for pb in pidx:
            if pb in pos_to_gap:
                v_syl[pos_to_gap[pb] - 1] = 1
            else:
                unrep += 1
    rho = (unrep / n_bounds) if n_bounds else 0.0

    # v_tok over the same gaps
    offs = _codepoint_byte_offsets(word)
    legal = legal_gap_positions(word)
    # map codepoint gap -> akshara gap index
    cp_to_gap, pos = {}, 0
    for gi, cl in enumerate(clusters[:-1]):
        pos += len(cl); cp_to_gap[pos] = gi
    v_tok = [0] * max(m - 1, 0)
    for (s, e) in token_byte_spans[:-1]:
        if e in offs:
            cp = offs.index(e)
            if cp in legal:
                v_tok[cp_to_gap[cp]] = 1

    stad = None
    if m >= 2:
        stad = sum(abs(a - b) for a, b in zip(v_syl, v_tok)) / (m - 1)
    return STADResult(stad, rho, v_syl, v_tok, m)
