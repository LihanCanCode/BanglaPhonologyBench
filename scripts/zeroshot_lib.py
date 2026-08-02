# -*- coding: utf-8 -*-
"""M5 zero-shot prompting + parsing + scoring (Spec A.6 "Prompting": zero-shot,
Bangla-language prompts, with an English-prompt ablation).

Model-agnostic on purpose: `run_task(task, rows, generate_fn, lang="bn")` takes
any `generate_fn(prompt: str) -> str` (a real model's `.generate()` wrapped to
return decoded text, or — for local testing without a GPU — a canned/echo
function). This mirrors how `kaggle_probing_lib.py` was validated before ever
touching a real model: `tests/test_zeroshot_lib.py` exercises every
prompt/parse/score path with a mock `generate_fn`, no GPU or network needed.

Five tasks, one prompt-builder + parser + scorer each:
  g2p              -> character-level PER + exact match against gold IPA string
  syllable_count   -> exact-match accuracy against gold syllable_count
  rhyme_awareness  -> accuracy/F1 against gold binary label (Task 3a)
  rhyme_generation -> mean Success@k against gold rhyme set (Task 3b, reuses
                      src/rhyme.py's PhonologyBench-style scorer)
  schwa_deletion   -> per-position accuracy/F1 + exact-vector match (Task 4)

Schwa elicitation note: the model isn't shown IPA. It's given the word's
consonant-bearing aksharas (the orthographic grapheme clusters at
`schwa_positions`, e.g. word পটলের -> aksharas প/ট/র are schwa-eligible,
লে is not because it has an explicit matra) numbered in order, and asked
Y/N per position — "is this consonant's default vowel pronounced." This is
the only task-4 elicitation that doesn't require the model to already emit
correct IPA as a side channel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from rhyme import mean_success_at_k  # noqa: E402

GenerateFn = Callable[[str], str]


def levenshtein(a: Sequence, b: Sequence) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


# ----------------------------------------------------------------------------
# Prompt builders (bn = Bangla-language prompt, en = English ablation)
# ----------------------------------------------------------------------------

def prompt_g2p(word: str, lang: str = "bn") -> str:
    if lang == "bn":
        return (f'শব্দ: "{word}"\n'
                 "এই বাংলা শব্দটির আন্তর্জাতিক ধ্বনিমূলক বর্ণমালায় (IPA) উচ্চারণ লেখো। "
                 "শুধু IPA উচ্চারণটি লেখো, অন্য কোনো ব্যাখ্যা দিও না।")
    return (f'Word: "{word}"\n'
            "Write the IPA (International Phonetic Alphabet) pronunciation of this "
            "Bangla word. Output ONLY the IPA transcription, nothing else.")


def prompt_syllable_count(word: str, lang: str = "bn") -> str:
    if lang == "bn":
        return (f'শব্দ: "{word}"\n'
                 "এই শব্দে কয়টি অক্ষর (syllable) আছে? শুধু একটি সংখ্যা লেখো।")
    return (f'Word: "{word}"\n'
            "How many syllables does this Bangla word have? Output ONLY a single number.")


def prompt_rhyme_awareness(word1: str, word2: str, lang: str = "bn") -> str:
    if lang == "bn":
        return (f'শব্দ ১: "{word1}"\nশব্দ ২: "{word2}"\n'
                 "এই দুটি শব্দ কি একসাথে মিল (rhyme) করে? শুধু \"হ্যাঁ\" অথবা \"না\" লেখো।")
    return (f'Word 1: "{word1}"\nWord 2: "{word2}"\n'
            'Do these two Bangla words rhyme? Output ONLY "Yes" or "No".')


def prompt_rhyme_generation(word: str, k: int = 5, lang: str = "bn") -> str:
    if lang == "bn":
        return (f'শব্দ: "{word}"\n'
                 f"এই শব্দের সাথে মিল (rhyme) করে এমন {k}টি বাংলা শব্দ লেখো, কমা দিয়ে আলাদা করে। "
                 "শুধু শব্দগুলো লেখো, অন্য কিছু লিখো না।")
    return (f'Word: "{word}"\n'
            f"Write {k} Bangla words that rhyme with this word, separated by commas. "
            "Output ONLY the words.")


def prompt_schwa(word: str, clusters: Sequence[str], lang: str = "bn") -> str:
    numbered = ", ".join(f"{i+1}. {c}" for i, c in enumerate(clusters))
    if lang == "bn":
        return (f'শব্দ: "{word}"\n'
                 f"এই শব্দের নিচের ব্যঞ্জনবর্ণগুলোর প্রতিটির অন্তর্নিহিত \"অ\" স্বরধ্বনি "
                 f"উচ্চারিত হয় কিনা বলো:\n{numbered}\n"
                 "প্রতিটির জন্য Y (উচ্চারিত হয়) অথবা N (উচ্চারিত হয় না) লেখো, কমা দিয়ে "
                 f"আলাদা করে, {len(clusters)}টি উত্তর দাও। শুধু উত্তরগুলো লেখো।")
    return (f'Word: "{word}"\n'
            f"For each of the following consonant letters in this word, say whether its "
            f"default inherent vowel is pronounced:\n{numbered}\n"
            f"Answer Y (pronounced) or N (silent) for each, comma-separated, "
            f"{len(clusters)} answers total. Output ONLY the answers.")


# ----------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------

def parse_g2p(text: str) -> str:
    return text.strip().strip('"').splitlines()[0].strip() if text.strip() else ""


_INT_RE = re.compile(r"-?\d+")


def parse_syllable_count(text: str) -> Optional[int]:
    m = _INT_RE.search(text)
    return int(m.group()) if m else None


_YES_TOKENS = {"yes", "হ্যাঁ", "হ্যা", "হা", "true", "১", "1"}
_NO_TOKENS = {"no", "না", "false", "০", "0"}


def parse_yes_no(text: str) -> Optional[int]:
    t = text.strip().lower()
    first_word = re.split(r"[\s,.!।]+", t)[0] if t else ""
    if first_word in _YES_TOKENS or t.startswith("yes") or "হ্যাঁ" in t:
        return 1
    if first_word in _NO_TOKENS or t.startswith("no") or ("না" in t and "হ্যাঁ" not in t):
        return 0
    return None


def parse_word_list(text: str) -> List[str]:
    parts = re.split(r"[,\n।]+", text.strip())
    return [p.strip().strip('."\'') for p in parts if p.strip()]


def parse_schwa_answer(text: str, n_expected: int) -> List[Optional[int]]:
    parts = re.split(r"[,\n\s]+", text.strip())
    out: List[Optional[int]] = []
    for p in parts:
        pu = p.strip().upper().strip(".")
        if pu in ("Y", "YES", "1", "হ্যাঁ"):
            out.append(1)
        elif pu in ("N", "NO", "0", "না"):
            out.append(0)
        if len(out) == n_expected:
            break
    while len(out) < n_expected:
        out.append(None)
    return out


# ----------------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------------

def character_per(pred: str, gold: str) -> float:
    if not gold:
        return float("nan")
    return levenshtein(list(pred), list(gold)) / len(gold)


@dataclass
class TaskScore:
    task: str
    lang: str
    n: int
    n_parsed: int
    metrics: Dict[str, float]


def run_g2p(rows: List[dict], generate_fn: GenerateFn, lang: str = "bn") -> TaskScore:
    total_per, exact, n = 0.0, 0, 0
    for r in rows:
        pred = parse_g2p(generate_fn(prompt_g2p(r["orth"], lang)))
        gold = r["ipa"]
        total_per += character_per(pred, gold)
        exact += int(pred == gold)
        n += 1
    return TaskScore("g2p", lang, n, n, {"mean_char_PER": total_per / n, "exact_match": exact / n})


def run_syllable_count(rows: List[dict], generate_fn: GenerateFn, lang: str = "bn") -> TaskScore:
    exact, abs_err, n, parsed = 0, 0, 0, 0
    for r in rows:
        pred = parse_syllable_count(generate_fn(prompt_syllable_count(r["orth"], lang)))
        n += 1
        if pred is None:
            continue
        parsed += 1
        exact += int(pred == r["syllable_count"])
        abs_err += abs(pred - r["syllable_count"])
    return TaskScore("syllable_count", lang, n, parsed,
                      {"exact_match_acc": exact / parsed if parsed else float("nan"),
                       "MAE": abs_err / parsed if parsed else float("nan"),
                       "parse_rate": parsed / n})


def run_rhyme_awareness(rows: List[dict], generate_fn: GenerateFn, lang: str = "bn") -> TaskScore:
    tp = fp = tn = fn = parsed = 0
    n = len(rows)
    for r in rows:
        pred = parse_yes_no(generate_fn(prompt_rhyme_awareness(r["orth1"], r["orth2"], lang)))
        if pred is None:
            continue
        parsed += 1
        gold = r["label"]
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        else:
            fn += 1
    acc = (tp + tn) / parsed if parsed else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) else float("nan"))
    return TaskScore("rhyme_awareness", lang, n, parsed,
                      {"accuracy": acc, "F1": f1, "parse_rate": parsed / n})


def run_rhyme_generation(rows: List[dict], generate_fn: GenerateFn, lang: str = "bn", k: int = 5) -> TaskScore:
    all_candidates, all_gold = [], []
    for r in rows:
        candidates = parse_word_list(generate_fn(prompt_rhyme_generation(r["prompt_word"], k, lang)))
        all_candidates.append(candidates)
        all_gold.append(set(r["gold_rhymes"]))
    score = mean_success_at_k(all_candidates, all_gold, k=k)
    return TaskScore("rhyme_generation", lang, len(rows), len(rows), {f"success_at_{k}": score})


def run_schwa(rows: List[dict], generate_fn: GenerateFn, lang: str = "bn") -> TaskScore:
    exact, pos_correct, pos_total, n, parsed = 0, 0, 0, 0, 0
    for r in rows:
        clusters = [r["grapheme_clusters"][p] for p in r["schwa_positions"]]
        gold = r["schwa_vector"]
        raw = generate_fn(prompt_schwa(r["orth"], clusters, lang))
        pred = parse_schwa_answer(raw, len(gold))
        n += 1
        if any(p is None for p in pred):
            continue
        parsed += 1
        exact += int(pred == gold)
        pos_correct += sum(1 for p, g in zip(pred, gold) if p == g)
        pos_total += len(gold)
    return TaskScore("schwa_deletion", lang, n, parsed,
                      {"per_position_acc": pos_correct / pos_total if pos_total else float("nan"),
                       "exact_vector_match": exact / parsed if parsed else float("nan"),
                       "parse_rate": parsed / n})


TASK_RUNNERS = {
    "g2p": run_g2p,
    "syllable_count": run_syllable_count,
    "rhyme_awareness": run_rhyme_awareness,
    "rhyme_generation": run_rhyme_generation,
    "schwa_deletion": run_schwa,
}
