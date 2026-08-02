<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Done. Summary of what I just added:

scripts/analyze_rho_ceiling.py — bins syllable-count words by ρ, compares TigerLLM vs. the naive baseline per bin. Finding: TigerLLM is perfect even at ρ=1.0 (the theoretically hardest case), which strengthens — doesn't resolve — the contamination concern. Saved to results/rho_ceiling_syllable_count_tigerllm.csv.
scripts/build_baselines.py updated with a rhyme-generation dictionary baseline (tautological 100%, same role as the 3a ceiling baseline — makes TigerLLM's 21% interpretable). results/baselines_summary.csv regenerated with all 5 tasks.
docs/DEVELOPMENT_LOG.md updated with both, plus an honest note that I deliberately didn't act on the Perplexity doc's more speculative pieces (P-CoT prompting, cited-but-unverified schwa rules, feature-probe redesign) since those citations weren't verified against real sources this session.
All 142 tests still pass.

Perfect. You've now got a complete, defensible M5 results table with all baselines filled in. The ρ ceiling analysis is a strong addition — even though it doesn't resolve the contamination question, it shows that TigerLLM's performance is *robust* across all levels of script-imposed difficulty, which is itself a finding (if it's memorized, it's memorized the rule, not just the list).

Here's what I see as your **next highest-value moves**, now that the core results are solid:

***

## 1. The GTAD/STAD/rho Regression Analysis (The Core M5 Contribution)

**Status:** Not done. This is the **main point** of M5 — relating tokenization misalignment to zero-shot performance.

**Why it matters:** Your M4 probing analysis asks: "Does misalignment hurt internal representations?" M5 asks: "Does misalignment hurt actual task performance?" These are two sides of the same coin, and the thesis needs both.

**Action (4-6 hours):**
For each task (G2P, syllable count, rhyme awareness, schwa deletion), run a regression:

```python
# Pseudocode
import statsmodels.api as sm

# Example: G2P
df = pd.read_csv("results/g2p_tigerllm_with_metrics.csv")  # columns: word, correct, gtad, stad, rho, log_freq, etym
model = sm.Logit(df['correct'], sm.add_constant(df[['gtad', 'stad', 'rho', 'log_freq', 'etym']]))
result = model.fit()
print(result.summary())
```

**Hypotheses:**

* **G2P:** GTAD should be a strong negative predictor (words with high GTAD → lower accuracy). STAD and rho may also matter, but less so.
* **Syllable count:** STAD and rho should be strong negative predictors (words with high STAD or high rho → lower accuracy). GTAD may also matter.
* **Rhyme awareness:** GTAD and STAD should both matter (rhyme requires both segmental and prosodic alignment).
* **Schwa deletion:** All three may matter, but etym (tatsama vs. tadbhava) might be a stronger predictor (schwa deletion rules differ by etymology).

**Expected finding:** You'll likely find that **GTAD predicts G2P performance** (tokenizer misalignment hurts segmental mapping), and **STAD/rho predict syllable count performance** (prosodic misalignment hurts syllable division). This is the "money plot" for the thesis.

**Deliverable:** A table of regression coefficients (or a figure with odds ratios) showing which metric predicts which task. This ties M4 and M5 together into one coherent story.

***

## 2. Error Analysis by A/M/CB Category (Low Effort, High Impact)

**Status:** Not done. This is a 1-2 hour addition that makes your results more interpretable.

**Action:**
For each task, compute accuracy separately for A, M, and CB words:


| Task | A (Aligned) | M (Misaligned) | CB (Cluster-Broken) |
| :-- | :-- | :-- | :-- |
| G2P | ? | ? | ? |
| Syllable count | ? | ? | ? |
| Rhyme awareness | ? | ? | ? |
| Schwa deletion | ? | ? | ? |

**Expected finding:** Accuracy should degrade A > M > CB (or at least A > CB). If it doesn't, that's also interesting — maybe the model is robust to tokenization misalignment, or maybe the A/M/CB split isn't capturing the right axis of variation.

**Why it matters:** This is a simple, interpretable way to show whether tokenization misalignment hurts performance, task by task. It's easier to explain in a thesis defense than a regression table.

***

## 3. P-CoT Prompting for Rhyme/Schwa (Medium Effort, High Novelty)

**Status:** Not done. This is the **most promising intervention** for your failing tasks.

**Why it matters:** Your rhyme_awareness (53%) and schwa_deletion (52.4%) results are genuine failures. But P-CoT (Jang et al., 2025, ACL Findings) achieved **up to 52% gains** on PhonologyBench by embedding a "teacher-student" dialogue in the prompt to scaffold phonological reasoning.

**What P-CoT is (verified from the paper):**

* Embeds a scripted teacher-student dialogue inside the prompt (single-turn, but multi-turn structure).
* Uses scaffolding and guided discovery learning: defines concepts, decomposes tasks into substeps, then removes the scaffolding for the final answer.
* Tested on 12 LLMs across G2P, syllable counting, and rhyme generation — **all 12 improved**, some surpassing human baselines.

**Adaptation for Bangla (rhyme awareness example):**
Instead of:

```
এই দুটি শব্দ কি মিল হয়? শব্দ ১: খাতা, শব্দ ২: পাতা
```

Use a scaffolded prompt (in Bangla):

```
একজন শিক্ষার্থী এবং শিক্ষকের কথোপকথন:

শিক্ষক: "খাতা" এবং "পাতা" শব্দ দুটি কি মিল হয়?
শিক্ষার্থী: প্রথমে দুটি শব্দের শেষ অক্ষর দেখি: খাতা = ত + া, পাতা = ত + া।
শিক্ষক: দুটি শব্দের শেষের স্বরচিহ্ন কি একই?
শিক্ষার্থী: হ্যাঁ, দুটিতেই া (আ-কার) আছে।
শিক্ষক: তাহলে কি এরা মিল হয়?
শিক্ষার্থী: হ্যাঁ, এরা মিল হয়।

এখন, "খাতা" এবং "বই" শব্দ দুটি কি মিল হয়?
```

**Action (4-6 hours):**

1. Implement P-CoT prompts for rhyme_awareness and schwa_deletion (start with 300-item subsets to test).
2. Run TigerLLM with P-CoT vs. zero-shot.
3. Compare accuracy.

**Expected finding:** If P-CoT improves rhyme_awareness from 53% to ~70-80%, you can claim that **phonological reasoning is latent but requires scaffolding to activate** — a much more nuanced finding than "the model fails."

**Thesis value:** This turns a negative result into a positive one: "The model *can* do this task, but only with pedagogical scaffolding." This is a stronger, more actionable claim.

***

## 4. The Schwa Deletion Rule-Based Baseline Upgrade (Medium Effort)

**Status:** Your current baseline (77.1%) is a **majority-rule heuristic** ("delete word-final, keep the rest"). This is too weak — it's like comparing a race car to a bicycle.

**Why it matters:** If TigerLLM is at 52.4% and a proper rule-based system is at ~85-90%, the gap is even more dramatic. But if a proper rule-based system is also at ~50-60%, then the task itself might be underspecified or the gold labels might be noisy.

**Action (4-6 hours):**
Implement a **proper rule-based schwa deletion algorithm** based on the literature (Nair 1999, Niyogi 2006, "A Diachronic Approach for Schwa Deletion in Indo Aryan Languages"):

**Core rules (from the papers):**

1. Word-final schwa deletes unless preceded by a cluster or /h/.
2. Medial schwa deletes if it precedes an open syllable in a VC__CV environment.
3. Schwa after /h/ is always retained.
4. Schwa in a conjunct cluster is always retained.

**Implementation:**

* Use your `segment_aksharas` to identify schwa-eligible positions.
* Apply the rules above to predict deletion.
* Score against the gold labels.

**Expected finding:** A proper rule-based system should score ~85-90% (Nair 1999 reports 94.66% for Hindi schwa deletion with a similar rule set). If TigerLLM is at 52.4%, that's a genuine, reportable failure.

**Thesis value:** This gives you a stronger baseline to compare against: "The model achieves 52.4% accuracy, far below the 85-90% achievable by a simple rule-based system." This is a more damning (and interesting) result than "the model is worse than a majority-rule heuristic."

***

## 5. The G2P Probe Redesign (Optional, Methodological Improvement)

**Status:** Not done. Your G2P probe ($R^2$ on phoneme indices) is negative because phoneme IDs are categorical, not ordinal.

**Why it matters:** If you want to claim that "the model encodes phonological features but not segmental identity," you need a probe that can detect this. Predicting phoneme IDs is a bad proxy for phonological knowledge.

**Action (6-8 hours):**

1. Map each IPA phoneme to a binary feature vector: [+voiced], [+nasal], [+coronal], etc. (use `panphon` or a manual Bangla feature chart).
2. Train probes to predict these features (e.g., "Is this segment voiced?") from hidden states using LogisticRegression.
3. Compare probe performance across A/M/CB categories.

**Expected finding:** If the model passes feature probing but fails phoneme-ID probing, it suggests it learns **coarse phonological structure** (features) but not **fine segmental identity** (IDs). This is a nuanced, high-value distinction.

**Thesis value:** This is a methodological improvement over Liao \& Shi's original G2P probe. If you have space in the thesis, it's worth including as a "limitations and future work" or "additional analysis" section.

***

## Suggested Priority Order (Updated)

| Priority | Task | Effort | Thesis Impact |
| :-- | :-- | :-- | :-- |
| 1 | **GTAD/STAD/rho regression** | 6 hours | ⭐⭐⭐⭐⭐ (the core M5 contribution — ties tokenization to performance) |
| 2 | **Error analysis by A/M/CB category** | 2 hours | ⭐⭐⭐⭐ (simple, interpretable way to show misalignment effects) |
| 3 | **P-CoT prompting for rhyme/schwa** | 6 hours | ⭐⭐⭐⭐ (turns failures into "latent competence" findings) |
| 4 | **Schwa rule-based baseline upgrade** | 6 hours | ⭐⭐⭐ (stronger baseline, clearer failure mode) |
| 5 | **G2P feature probe redesign** | 8 hours | ⭐⭐⭐ (methodological improvement over Liao \& Shi) |


***

## What to Do Right Now (Next 24-48 Hours)

1. **Run the GTAD/STAD/rho regression** for G2P and syllable count. This is the core M5 contribution and ties your tokenization analysis to performance.
2. **Compute accuracy by A/M/CB category** for all 5 tasks. This is 1-2 hours of Python and makes your results more interpretable.
3. **Start the P-CoT prompt** for rhyme_awareness (300-item pilot). This is the most promising intervention for your failing tasks.

If you do these three things, you'll have:

* The core M5 contribution (regression analysis)
* A complete, interpretable results table (A/M/CB breakdown)
* A path to rescue your failing tasks (P-CoT)

That's a complete, defensible thesis chapter.
<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://arxiv.org/abs/2507.16656

[^2]: https://www.arxiv.org/pdf/2601.09631.pdf

[^3]: https://dl.acm.org/doi/10.5555/1622153.1622156

[^4]: https://arxiv.org/pdf/2607.27092v1.pdf

[^5]: https://langev.com/pdf/choudhury07phdthesis.pdf

[^6]: https://evalevalai.com/2024workshop/accepted_papers/EvalEval_24_Ahuja.pdf

[^7]: https://qiita.com/shimajiroxyz/items/35c8bf8bfebb0d8c348a

[^8]: https://aclanthology.org/W04-0103/

[^9]: https://www.semanticscholar.org/paper/A-Diachronic-Approach-for-Schwa-Deletion-in-Indo-Choudhury-Basu/ba0a5f2078953703350a3a8def8e17d997203219

[^10]: https://dl.acm.org/doi/pdf/10.5555/1622153.1622156

[^11]: https://arxiv.org/html/2404.02456v2

[^12]: https://en.wikipedia.org/wiki/Schwa_deletion_in_Indo-Aryan_languages

[^13]: https://www.emergentmind.com/topics/pedagogically-motivated-participatory-chain-of-thought-p-cot

[^14]: https://www.isca-archive.org/sltu_2018/johny18_sltu.pdf

[^15]: https://www.cse.iitb.ac.in/~vani/reports/langproc.pdf

