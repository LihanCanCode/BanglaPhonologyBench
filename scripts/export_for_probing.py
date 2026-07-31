# -*- coding: utf-8 -*-
"""Export our frozen task datasets into the CSV shapes consumed by the forked
probing harness (external/Tokenization-Phonology, Liao & Shi 2026), and split
by the A / M / CB word categories (Spec C.4) in place of their good/bad split.

Mapping to the upstream repo (see docs/probing_integration.md for the full
walkthrough — the fork itself is NOT committed here, it's a plain
`git clone`, reproducible from that doc):
  data/tasks/g2p.jsonl          -> data/probing_export/g2p_{tokenizer}_{A,M,CB}.csv
                                    columns: word, phon_vec, syllables
                                    (phon_vec = phoneme indices, padded to the
                                    dataset max length with 0 — their
                                    train_probe_g2p.py consumes this shape
                                    directly; our max is reported below vs.
                                    their English max of 8)
  data/tasks/syllable_count_word.jsonl -> reuses the same CSV (syllables col)
  data/task3a_rhyme_pairs.jsonl -> data/probing_export/rhyme_pairs.csv
                                    columns: word1, word2, label

At M4 time, copy (or symlink) data/probing_export/*.csv into the forked
repo's probing/data/ directory before running generate_embedding.py.

This script does NOT call any model / GPU code — embedding extraction
(probing/generate_embedding.py) is milestone M4 and runs on Kaggle T4s.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("data/probing_export")
TASKS = Path("data/tasks")
TOKENIZERS = ["llama3", "gpt2", "byt5", "tigerllm", "banglat5"]
M_THRESHOLD = 0.25   # Spec C.4: Syllable-misaligned category threshold


def categorize(gtad, stad):
    if gtad is None or stad is None:
        return None
    if gtad > 0:
        return "CB"
    if stad > M_THRESHOLD:
        return "M"
    return "A"


def build_phoneme_vocab(items):
    vocab = Counter()
    for it in items:
        vocab.update(it["phonemes"])
    # deterministic index assignment: most frequent first (ties broken alphabetically)
    ordered = sorted(vocab, key=lambda p: (-vocab[p], p))
    return {p: i + 1 for i, p in enumerate(ordered)}   # 0 reserved for padding


def main():
    items = [json.loads(l) for l in open(TASKS / "g2p.jsonl", encoding="utf-8")]
    pad_len = max(len(it["phonemes"]) for it in items)
    vocab = build_phoneme_vocab(items)
    print(f"[export] {len(items)} g2p items, phoneme vocab size={len(vocab)}, "
          f"pad_len={pad_len} (English arpabet reference: 8)")

    OUT.mkdir(parents=True, exist_ok=True)
    for tk in TOKENIZERS:
        rows = {"A": [], "M": [], "CB": []}
        n_missing = 0
        for it in items:
            block = it["tokenization"].get(tk)
            if not block or block.get("quarantined"):
                n_missing += 1
                continue
            cat = categorize(block["gtad"], block["stad"])
            if cat is None:
                n_missing += 1
                continue
            idx = [vocab[p] for p in it["phonemes"]] + [0] * (pad_len - len(it["phonemes"]))
            rows[cat].append({
                "word": it["orth"],
                "phon_vec": idx,
                "syllables": it["syllables_phonemic"],
                "ipa": it["ipa"],
            })
        for cat, rs in rows.items():
            path = OUT / f"g2p_{tk}_{cat}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["word", "phon_vec", "syllables", "ipa"])
                w.writeheader()
                for r in rs:
                    w.writerow(r)
            print(f"  {path.name}: {len(rs)} words")
        if n_missing:
            print(f"  ({tk}: {n_missing} items skipped — quarantined/missing tokenization)")

    with open(TASKS / "phoneme_vocab.json", "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "pad_len": pad_len}, f, ensure_ascii=False, indent=2)
    print(f"[export] wrote phoneme vocab -> data/tasks/phoneme_vocab.json")

    rhyme = [json.loads(l) for l in open("data/task3a_rhyme_pairs.jsonl", encoding="utf-8")]
    rp = OUT / "rhyme_pairs.csv"
    with open(rp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["word1", "word2", "label"])
        w.writeheader()
        for it in rhyme:
            w.writerow({"word1": it["orth1"], "word2": it["orth2"], "label": it["label"]})
    print(f"[export] wrote {rp.name}: {len(rhyme)} pairs")


if __name__ == "__main__":
    main()
