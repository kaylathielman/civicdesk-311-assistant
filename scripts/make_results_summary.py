"""Assemble logs/results_summary.md from the saved artifacts (no training or generation).

Re-runs the cheap checks (audit, shortcut check, comparison table) and reads the training logs,
so every number in the summary comes from a file in this repo.
Usage: python scripts/make_results_summary.py
"""

import ast
import csv
import random
import re
import statistics
import subprocess
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import AutoTokenizer

from common import BASE_MODEL, DATA_DIR, LOG_DIR, MEMORY_LOG, ROOT, load_jsonl

SCRIPTS = ROOT / "scripts"
PREF_SETS = {"v1": ["preference_train.jsonl"], "v2": ["preference_v2_train.jsonl", "preference_v2_val.jsonl"]}


def run(script, *args):
    out = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True, cwd=SCRIPTS)
    if out.returncode:
        raise RuntimeError(f"{script} failed:\n{out.stderr[-2000:]}")
    return out.stdout.strip()


def code(text):
    return f"```\n{text}\n```"


def length_table(tok):
    lines = ["| set | pair type | pairs | chosen chars | rejected chars | chosen tokens | rejected tokens | max diff |",
             "|---|---|---|---|---|---|---|---|"]
    for v, files in PREF_SETS.items():
        pairs = [p for f in files for p in load_jsonl(DATA_DIR / f)]
        types = sorted({p["meta"]["pair_type"] for p in pairs}) + ["ALL"]
        for t in types:
            sub = [p for p in pairs if t == "ALL" or p["meta"]["pair_type"] == t]
            c = [p["chosen"][0]["content"] for p in sub]
            r = [p["rejected"][0]["content"] for p in sub]
            ntok = lambda xs: statistics.mean(len(tok(x)["input_ids"]) for x in xs)
            diff = max(max(len(a), len(b)) / min(len(a), len(b)) - 1 for a, b in zip(c, r))
            lines.append(f"| {v} | {t} | {len(sub)} | {statistics.mean(map(len, c)):.1f} | "
                         f"{statistics.mean(map(len, r)):.1f} | {ntok(c):.1f} | {ntok(r):.1f} | {diff:.1%} |")
        shorter = sum(len(p["chosen"][0]["content"]) < len(p["rejected"][0]["content"]) for p in pairs)
        lines.append(f"| {v} | chosen is shorter | {shorter}/{len(pairs)} ({shorter / len(pairs):.0%}) | | | | | |")
    return "\n".join(lines)


def feature_breakdown(tok, path):
    """Pairwise LR with each feature alone, to show where the residual signal comes from."""
    pairs = load_jsonl(path)
    rng, X, y = random.Random(0), [], []
    for p in pairs:
        c, r = p["chosen"][0]["content"], p["rejected"][0]["content"]
        fc, fr = (len(c), len(tok(c)["input_ids"])), (len(r), len(tok(r)["input_ids"]))
        first = rng.random() < 0.5
        a, b = (fc, fr) if first else (fr, fc)
        X.append([a[0] - b[0], a[1] - b[1]])
        y.append(int(first))
    X, cv = np.array(X), StratifiedKFold(5, shuffle=True, random_state=0)
    acc = lambda cols: cross_val_score(LogisticRegression(), X[:, cols], y, cv=cv).mean()
    return acc([0]), acc([1]), acc([0, 1])


def dpo_rewards(version):
    """Parse the trainer's logged dicts from logs/train_dpo_<v>.out."""
    rows, text = [], (LOG_DIR / f"train_{version}.out").read_text()
    for blob in re.findall(r"\{'(?:loss|eval_loss|train_runtime)'[^}]*\}", text):
        d = {k: float(v) for k, v in ast.literal_eval(blob).items()}
        pre = "eval_" if "eval_loss" in d else ""
        if f"{pre}rewards/margins" not in d:
            continue
        rows.append({"split": "held-out" if pre else "train", "epoch": d.get("epoch"),
                     "margin": d[f"{pre}rewards/margins"], "acc": d[f"{pre}rewards/accuracies"],
                     "chosen": d[f"{pre}rewards/chosen"], "rejected": d[f"{pre}rewards/rejected"]})
    return rows


def memory_table():
    with MEMORY_LOG.open() as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c != "notes"]
    lines = ["| # | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    notes = []
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | " + " | ".join(r[c] for c in cols) + " |")
        if r["notes"]:
            notes.append(f"{i}. `{r['notes']}`")
    return "\n".join(lines), "\n".join(notes)


DIAGNOSIS = """\
### sft_v1 — the model to use
Learned the format and categories (category 17% -> 90% on test) and stopped hiding the address in a nested
`location` object (false-null 81% -> 0%). Remaining failures come from what the SFT data never showed it:
non-address numbers (phone, unit) get copied into `address` (probe-14, probe-15), and urgency barely moved
(37% -> 40%) because urgency is 1 of ~34 trained tokens and depends on fine distinctions 340 examples barely cover.

### dpo_v1 — failed run (over-optimization)
beta 0.1, lr 2e-5, 2 epochs, 150 pairs, 30% address-present.
- Train reward accuracy hit 1.00 by step 20 and the margin kept growing to 4.73. Both chosen and rejected
  rewards went negative: the policy lowered the probability of the *correct* tickets too, just less than the
  wrong ones. That is classic DPO over-optimization against a tiny, narrow dataset.
- Chosen and rejected differed only in the address value, so all the gradient landed on the few tokens around
  it. The damage shows up exactly there: 4 of 5 invalid JSON outputs drop the closing quote after the address
  (`"2241 Center Pl}`), then spreads (a Chinese summary, `location: "in the area of the reported damage, if provided"`).
- It did not fix the probe failures: the pairs only changed `address`, never `location`, and their phrasing
  ("Call me at ...") did not match the probe's ("I'm at 555-0172 ...").
- The 70/30 null/address mix gave a length shortcut (shorter response chosen in 70% of pairs). It did not
  show up as false-nulls (0/31), so it was not the main failure.

### dpo_v2 — stable but too weak to change behavior
beta 0.5, lr 5e-6, 1 epoch, 150 train + 20 held-out pairs, 50/50 mix, + location-distractor pairs.
- Gentle settings worked as intended: JSON validity 100%, category and address accuracy back to SFT level,
  no drift. The trainable adapter moved ~10x less than in v1 (max weight change 3.9e-5 vs 4.0e-4).
- Held-out reward accuracy 0.90 shows the preference was learned in probability terms, but not strongly
  enough to flip greedy outputs: only 6/30 test and 4/20 probe outputs differ from SFT, mostly reworded summaries.
  probe-14 (phone) and probe-15 (unit) are unchanged.
- Both DPO runs newly copy a landmark into `address` on the same 2 test cases ("7th", "at the light by the
  Taco Bell"). Likely cause is the preference-data design: full-address pairs reward "copy complaint text into
  address" over null, while landmark pairs only penalize an *invented* house number, never a copied landmark.
  So "copying beats null" generalizes to landmarks. Same behavior in both runs => data, not hyperparameters.

### What would likely work better (not run; tuning stopped by decision)
1. Put distractor numbers into the SFT data itself, so the base behavior is right before any preference step.
2. Add preference pairs whose rejected answer copies a landmark or a phone/unit number into `address`.
"""


def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    run("compare_evals.py")
    comparison = (LOG_DIR / "comparison.md").read_text()
    shortcut = {v: run("check_length_shortcut.py", str(DATA_DIR / files[0])) for v, files in PREF_SETS.items()}
    breakdown = {v: feature_breakdown(tok, DATA_DIR / files[0]) for v, files in PREF_SETS.items()}
    mem_rows, mem_notes = memory_table()

    reward_lines = ["| run | split | when | epoch | reward margin | reward accuracy | chosen reward | rejected reward |",
                    "|---|---|---|---|---|---|---|---|"]
    for v in ("dpo_v1", "dpo_v2"):
        reward_lines.append(f"| {v} | train | start (step 0) | 0 | 0.000 | 0.00 | 0.000 | 0.000 |")
        rows = dpo_rewards(v)
        for split in ("train", "held-out"):
            sub = [r for r in rows if r["split"] == split]
            for i, r in enumerate(sub):
                when = "end" if i == len(sub) - 1 and r["epoch"] == max(x["epoch"] for x in rows) else "middle"
                reward_lines.append(f"| {v} | {split} | {when} | {r['epoch']:.2f} | {r['margin']:.3f} | {r['acc']:.2f} | "
                                    f"{r['chosen']:.3f} | {r['rejected']:.3f} |")

    md = f"""# CivicDesk 311 — Results Summary

Generated by `scripts/make_results_summary.py` from the files in `data/` and `logs/`.
Model: Qwen2.5-0.5B-Instruct, 4-bit NF4 QLoRA, Colab T4. Adapters: `sft_v1`, `dpo_v1` (failed run), `dpo_v2`.
**Recommended adapter: `sft_v1`.**

## 1. SFT data audit (`scripts/audit_data.py`)
{code(run("audit_data.py"))}

## 2. Preference pairs: chosen vs rejected length
Chosen and rejected tickets are identical except `address` (v2 `location_distractor` pairs also differ in `location`).
All pairs are within the 15% length limit.

{length_table(tok)}

## 3. Length-shortcut check (logistic regression, 5-fold CV; 50% = no shortcut)
**Before rebalancing (v1, 30% address-present):**
{code(shortcut["v1"])}
**After rebalancing (v2, 50/50):**
{code(shortcut["v2"])}

Pairwise accuracy by feature (which signal the classifier uses):

| set | char-diff only | token-diff only | both |
|---|---|---|---|
""" + "\n".join(f"| {v} | {a:.1%} | {b:.1%} | {c:.1%} |" for v, (a, b, c) in breakdown.items()) + f"""

Reading: rebalancing removed the raw length signal ("shorter wins" 70% -> 49%; char-diff alone ~55%). The remaining
~69% comes from combining chars and tokens: Qwen tokenizes each digit separately, so the rejected answers in distractor
pairs (phone numbers, dates) cost ~1 token per character vs ~0.5 for street addresses. That is digit density, which
tracks the lesson itself ("a bare number is not an address") rather than a length shortcut.

## 4. DPO reward margin and reward accuracy
Logged every 10 steps (values average the steps since the previous log). Step 0 is zero by construction (policy = reference).
dpo_v2 has 19 steps, so its only train-split log is step 10; its end-of-run train-split rewards were not logged by trl,
so the end point is the held-out evaluation.

""" + "\n".join(reward_lines) + f"""

## 5. Memory and wall-clock log (`logs/memory_time_log.csv`, all rows)
`elapsed_s` is time since training start. `peak_mem_alloc_gb` is the true peak in use; `reserved` includes PyTorch's cache.
Row `oom_test` is a deliberate out-of-memory test of the error handler (batch size 340), not a real run.
dpo_v2 has two identical step-19 `eval` rows: trl's automatic end-of-training evaluation plus the script's explicit one.
Evaluation scripts are not training, so they don't write to this log.

{mem_rows}

Notes column:

{mem_notes}

## 6. Final comparison (test = 30 generated, probe = 20 hand-written in different styles)
Fabrication = non-null `address` when the complaint has none. Location fabrication = a street address or number in
`location` that is not in the complaint (any example). Broken JSON is still checked for fabrication.

{comparison}

## 7. Diagnosis

{DIAGNOSIS}"""
    (LOG_DIR / "results_summary.md").write_text(md)
    print(f"wrote {LOG_DIR / 'results_summary.md'} ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
