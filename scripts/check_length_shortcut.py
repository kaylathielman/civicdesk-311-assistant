"""Shortcut check: can length alone tell chosen from rejected? If so, DPO might learn "shorter wins".

Two logistic regressions on [character length, token count], 5-fold cross-validated:
  per-response - each response on its own -> chosen or rejected? (folds grouped by pair, so a pair never
                 straddles train/test)
  pairwise     - given both responses of a pair in random order -> which is chosen? (features are differences).
                 This is the comparison DPO actually makes, so it's the stricter test.
Close to 50% = length carries no signal.

Usage: python scripts/check_length_shortcut.py [data/preference_train.jsonl]
"""

import random
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from common import BASE_MODEL, DATA_DIR, load_jsonl


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / "preference_train.jsonl"
    pairs = load_jsonl(path)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    feats = lambda text: [len(text), len(tok(text)["input_ids"])]
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression())

    X, y, groups = [], [], []
    for i, p in enumerate(pairs):
        for side, label in (("chosen", 1), ("rejected", 0)):
            X.append(feats(p[side][0]["content"]))
            y.append(label)
            groups.append(i)
    acc1 = cross_val_score(clf(), np.array(X), y, groups=groups, cv=GroupKFold(5))

    rng = random.Random(0)
    Xp, yp = [], []
    for p in pairs:
        c, r = feats(p["chosen"][0]["content"]), feats(p["rejected"][0]["content"])
        first_is_chosen = rng.random() < 0.5
        a, b = (c, r) if first_is_chosen else (r, c)
        Xp.append([a[0] - b[0], a[1] - b[1]])
        yp.append(int(first_is_chosen))
    acc2 = cross_val_score(clf(), np.array(Xp), yp, cv=StratifiedKFold(5, shuffle=True, random_state=0))

    shorter = np.mean([len(p["chosen"][0]["content"]) < len(p["rejected"][0]["content"]) for p in pairs])
    print(f"Pairs: {len(pairs)}")
    print(f"  per-response logistic regression accuracy: {acc1.mean():.1%} (folds: {', '.join(f'{a:.0%}' for a in acc1)})")
    print(f"  pairwise logistic regression accuracy:     {acc2.mean():.1%} (folds: {', '.join(f'{a:.0%}' for a in acc2)})")
    print(f"  rule 'the shorter response is chosen':     {shorter:.1%}")
    print("  (50% = no length shortcut)")


if __name__ == "__main__":
    main()
