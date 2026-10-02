"""Audit the synthetic 311 dataset for diversity and address correctness.

Reports:
  1. Distinct first-three-word openers in complaints and in summaries
  2. Summary length histogram (in words)
  3. Percentage of each address case (derived from the ticket, cross-checked against meta)
  4. Address checks: every ticket address appears word-for-word in its complaint,
     landmark locations appear word-for-word, and null-address complaints contain nothing address-like
  5. Schema checks and duplicate complaints within/across splits

Exits with status 1 if any hard check fails.
Usage: python scripts/audit_data.py [files...]   (default: data/{train,val,test}.jsonl)
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FILES = [DATA_DIR / f"{s}.jsonl" for s in ("train", "val", "test")]
FIELDS = {"category", "urgency", "location", "summary", "address"}
URGENCIES = {"low", "medium", "high"}
SUFFIX = r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|ct|court|pl|place|way)"
ADDRESS_LIKE = re.compile(rf"\b\d{{1,5}}\s+(?:[nsew]\s+)?(?:\d+(?:st|nd|rd|th)|[a-z]+)(?:\s+[a-z]+){{0,3}}\s+{SUFFIX}\b", re.I)


def opener(text):
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split()[:3])


def address_case(t):
    if t["address"] is not None:
        return "full_address"
    return "no_location" if t["location"] is None else "landmark_only"


def diversity(name, texts):
    c = Counter(opener(t) for t in texts)
    print(f"  {name}: {len(c)} distinct openers across {len(texts)} rows ({len(c) / len(texts):.0%})")
    for o, n in c.most_common(5):
        print(f"      {n:>3}x  \"{o}\"")


def histogram(lengths, width=40):
    c = Counter(lengths)
    peak = max(c.values())
    for n in range(min(c), max(c) + 1):
        print(f"  {n:>3} words | {'#' * round(c[n] / peak * width):<{width}} {c[n]}")
    print(f"  min {min(lengths)}, mean {sum(lengths) / len(lengths):.1f}, max {max(lengths)}")


def main():
    files = [Path(p) for p in sys.argv[1:]] or DEFAULT_FILES
    rows = []
    for path in files:
        for line_no, line in enumerate(path.open(), 1):
            rows.append((path.stem, line_no, json.loads(line)))
    print(f"Loaded {len(rows)} rows from {', '.join(p.name for p in files)}")
    for split, n in Counter(s for s, _, _ in rows).items():
        print(f"  {split}: {n}")

    failures = []

    def fail(split, line_no, msg):
        failures.append(f"{split}:{line_no}: {msg}")

    # Schema
    for split, ln, r in rows:
        t = r["ticket"]
        if set(t) != FIELDS:
            fail(split, ln, f"ticket fields {sorted(t)} != {sorted(FIELDS)}")
        if t.get("urgency") not in URGENCIES:
            fail(split, ln, f"bad urgency {t.get('urgency')!r}")
        if not t.get("summary"):
            fail(split, ln, "empty summary")

    print("\n== 1. Opener diversity (first three words) ==")
    diversity("complaints", [r["complaint"] for _, _, r in rows])
    diversity("summaries ", [r["ticket"]["summary"] for _, _, r in rows])

    print("\n== 2. Summary length (words) ==")
    histogram([len(r["ticket"]["summary"].split()) for _, _, r in rows])

    print("\n== 3. Address cases ==")
    splits = sorted({s for s, _, _ in rows}, key=["train", "val", "test"].index)
    by_split = {s: Counter(address_case(r["ticket"]) for sp, _, r in rows if sp == s) for s in splits}
    total = Counter(address_case(r["ticket"]) for _, _, r in rows)
    print(f"  {'case':<14}{'all':>12}" + "".join(f"{s:>12}" for s in splits))
    for case in ("full_address", "no_location", "landmark_only"):
        line = f"  {case:<14}{total[case]:>4} ({total[case] / len(rows):>4.0%})"
        for s in splits:
            n = sum(by_split[s].values())
            line += f"{by_split[s][case]:>5} ({by_split[s][case] / n:>4.0%})"
        print(line)
    for split, ln, r in rows:
        if r.get("meta", {}).get("address_case") not in (None, address_case(r["ticket"])):
            fail(split, ln, "meta.address_case disagrees with ticket")

    print("\n== 4. Address grounding ==")
    n_addr = n_lm = n_null = 0
    for split, ln, r in rows:
        t, c = r["ticket"], r["complaint"]
        if t["address"] is not None:
            n_addr += 1
            if t["address"] not in c:
                fail(split, ln, f"address {t['address']!r} not found verbatim in complaint")
        else:
            n_null += 1
            if t["location"] is not None:
                n_lm += 1
                if t["location"] not in c:
                    fail(split, ln, f"landmark {t['location']!r} not found verbatim in complaint")
            m = ADDRESS_LIKE.search(c)
            if m:
                fail(split, ln, f"address is null but complaint contains address-like text {m.group()!r}")
    print(f"  {n_addr} tickets with an address: all found word-for-word in complaint? "
          f"{'YES' if not any('address ' in f and 'verbatim' in f for f in failures) else 'NO'}")
    print(f"  {n_lm} landmark locations: all found word-for-word? "
          f"{'YES' if not any('landmark' in f for f in failures) else 'NO'}")
    print(f"  {n_null} null-address complaints scanned for address-like text: "
          f"{sum('address-like' in f for f in failures)} hits")

    print("\n== 5. Duplicates ==")
    seen = {}
    dups = 0
    for split, ln, r in rows:
        key = r["complaint"].strip().lower()
        if key in seen:
            dups += 1
            fail(split, ln, f"duplicate complaint (also {seen[key]})")
        seen[key] = f"{split}:{ln}"
    print(f"  duplicate complaints (case-insensitive, within or across splits): {dups}")

    print("\n== Result ==")
    if failures:
        print(f"  FAILED: {len(failures)} problem(s)")
        for f in failures[:25]:
            print(f"   - {f}")
        sys.exit(1)
    print("  All checks passed.")


if __name__ == "__main__":
    main()
