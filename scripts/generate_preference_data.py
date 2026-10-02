"""Build chosen/rejected pairs for DPO: teach the model to leave `address` null unless a street address is given.

Pair types (chosen and rejected tickets are identical except for the `address` field):
  distractor    - no address, but phone/unit/bus/date/count/case/time numbers in the text.
                  chosen: null  | rejected: copies a distractor into address
  no_address    - no location at all.            chosen: null | rejected: invents a plausible street address
  landmark      - landmark only.                 chosen: null (landmark stays in location)
                                                 rejected: invents a house number for the landmark
  full_address  - real address present (half also contain distractors).
                  chosen: copies it exactly | rejected: null  (so the model doesn't learn "always null")
  location_distractor (v2 only) - a phone/unit number is the ONLY location-like clue ("I'm at 555-0123").
                  chosen: address AND location null | rejected: copies the number into both

Versions:
  v1 - 150 pairs, 30% address-present (dpo_v1, a documented failed run). Reproduces data/preference_train.jsonl.
  v2 - 170 pairs, 50/50 address-present vs null, + location_distractor; 20 held out as validation.
       -> data/preference_v2_train.jsonl, data/preference_v2_val.jsonl

Complaints are NEW: generated with a different seed and checked against train/val/test/probe (and v1 for v2).
Prints a length comparison.

Usage: python scripts/generate_preference_data.py --version v2
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

import generate_instruction_data as gen
from audit_data import ADDRESS_LIKE
from common import BASE_MODEL, DATA_DIR, load_jsonl, prompt_messages, ticket_json

CONFIGS = {
    "v1": dict(seed=512, n_val=0, files={"train": "preference_train.jsonl"},
               counts={"distractor": 45, "no_address": 30, "landmark": 30, "full_address": 45}),
    "v2": dict(seed=513, n_val=20, files={"train": "preference_v2_train.jsonl", "val": "preference_v2_val.jsonl"},
               counts={"distractor": 25, "location_distractor": 20, "no_address": 20, "landmark": 20,
                       "full_address": 85}),
}
ADDRESS_PRESENT_TYPES = {"full_address"}
# The probe's exact distractors: v2 targets the same failure, but must not copy the probe's strings.
PROBE_SPANS = ["555-0172", "unit 12", "apartment 4C", "if you need me"]
MAX_LEN_DIFF = 0.15
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def make_distractor(rng):
    """Return (sentence, span): a sentence with a number that is NOT an address, and the span a bad model copies."""
    kind = rng.choice(["phone", "unit", "bus", "date", "count", "case", "time"])
    n = rng.randint(2, 98)
    if kind == "phone":
        line = f"555-01{rng.randint(0, 99):02d}"
        span = rng.choice([line, f"(612) {line}", f"612-{line}", f"612.{line.replace('-', '.')}"])
        sent = rng.choice(["Call me at {x}.", "My number is {x} if you need it.", "cell {x}",
                           "You can reach me at {x}.", "Text me at {x} when it's fixed."])
    elif kind == "unit":
        span = rng.choice([f"unit {n}", f"apt {n}{rng.choice('ABCDEF')}", f"apartment {n}{rng.choice('ABCD')}",
                           f"#{n}"])
        sent = rng.choice(["I'm in {x}.", "We live in {x}.", "{x} here.", "This is the tenant in {x}."])
    elif kind == "bus":
        span = rng.choice([f"{n} bus", f"route {n}", f"bus line {n}"])
        sent = rng.choice(["I saw it from the {x} this morning.", "I noticed it riding the {x} to work.",
                           "Saw it on my way home on the {x}."])
    elif kind == "date":
        span = rng.choice([f"{rng.choice(MONTHS)} {rng.randint(1, 28)}", f"{rng.randint(1, 12)}/{rng.randint(1, 28)}",
                           f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.choice([24, 25, 26])}"])
        sent = rng.choice(["It's been like this since {x}.", "I first noticed it on {x}.",
                           "Started around {x}."])
    elif kind == "count":
        span = rng.choice([f"{n} times", f"{n} days", f"{n} kids", f"{n} weeks", f"{n} cars"])
        sent = {"times": "I've called {x} already.", "days": "It's been {x} now.",
                "kids": "There are {x} on this block.", "weeks": "Going on {x}.",
                "cars": "Watched {x} swerve today."}[span.split()[1]]
    elif kind == "case":
        span = rng.choice([f"case #{rng.randint(1000, 99999)}", f"ticket {rng.randint(100000, 999999)}",
                           f"{rng.choice([2025, 2026])}-{rng.randint(100000, 999999)}"])
        sent = rng.choice(["My case number is {x}.", "Reference {x} from last time.", "Last report was {x}."])
    else:
        span = rng.choice([f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d} pm", f"{rng.randint(1, 12)} AM",
                           f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}am"])
        sent = rng.choice(["It started at {x}.", "Still going at {x}.", "Noticed it at {x} last night."])
    return sent.format(x=span), span, kind


def make_location_distractor(rng):
    """A phone or unit number phrased as if it were the location. Returns (sentence, span)."""
    if rng.random() < 0.5:
        line = f"555-01{rng.randint(0, 99):02d}"
        span = rng.choice([line, f"(612) {line}", f"612-{line}"])
        sent = rng.choice(["I'm at {x}.", "Reach me at {x}, that's where I am.", "This is {x} calling.",
                           "You can find me at {x}.", "Number here is {x}."])
    else:
        n = rng.randint(2, 98)
        span = rng.choice([f"unit {n}", f"apt {n}{rng.choice('ABCDEF')}", f"apartment {n}{rng.choice('ABCD')}",
                           f"suite {n}"])
        sent = rng.choice(["We're in {x}.", "I live in {x}.", "It's outside {x}.", "Right by {x}.",
                           "This is {x}."])
    return sent.format(x=span), span


def add_distractors(complaint, rng, k):
    """Insert k distractor sentences at the start or end. Returns (complaint, spans, kinds)."""
    spans, kinds = [], []
    for _ in range(k):
        sent, span, kind = make_distractor(rng)
        complaint = f"{sent} {complaint}" if rng.random() < 0.4 else f"{complaint} {sent}"
        spans.append(span)
        kinds.append(kind)
    return complaint, spans, kinds


def invent_landmark_address(landmark, rng):
    """A hallucinated house number for a landmark: 'behind the Walgreens on 7th' -> '1418 7th St'."""
    num = rng.randint(100, 4999)
    m = re.search(r"\bon (\w+)$", landmark)
    if m:
        return f"{num} {m.group(1)} {rng.choice(['St', 'Ave', 'Street'])}"
    tail = re.split(r"\b(?:to|at|from|behind|outside|of|by|near|under|the)\b", landmark)[-1].strip()
    return f"{num} {tail.title()}"


def build_pair(pair_type, category, rng):
    tone = rng.choice(list(gen.TONE_STYLE))
    base_case = {"distractor": rng.choice(["no_location", "no_location", "landmark_only"]),
                 "no_address": "no_location", "landmark": "landmark_only", "full_address": "full_address",
                 "location_distractor": "no_location"}[pair_type]
    ex = gen.make_example(category, base_case, tone, rng)
    complaint, ticket, meta = ex["complaint"], ex["ticket"], {"pair_type": pair_type, "tone": tone}

    if pair_type == "location_distractor":
        sent, span = make_location_distractor(rng)
        complaint = f"{sent} {complaint}" if rng.random() < 0.4 else f"{complaint} {sent}"
        chosen, rejected = dict(ticket, address=None, location=None), dict(ticket, address=span, location=span)
        return {"prompt": prompt_messages(complaint),
                "chosen": [{"role": "assistant", "content": ticket_json(chosen)}],
                "rejected": [{"role": "assistant", "content": ticket_json(rejected)}],
                "meta": {**meta, "distractor_kinds": ["location_" + ("phone" if "555" in span else "unit")],
                         "complaint": complaint, "chosen_address": None, "rejected_address": span}}

    if pair_type == "distractor":
        complaint, spans, kinds = add_distractors(complaint, rng, rng.choice([1, 1, 2]))
        good, bad = None, rng.choice(spans)
        meta["distractor_kinds"] = kinds
    elif pair_type == "no_address":
        good, bad = None, gen.make_address(rng)
    elif pair_type == "landmark":
        good, bad = None, invent_landmark_address(ticket["location"], rng)
    else:
        if rng.random() < 0.5:
            complaint, _, kinds = add_distractors(complaint, rng, 1)
            meta["distractor_kinds"] = kinds
        good, bad = ticket["address"], None

    chosen, rejected = dict(ticket, address=good), dict(ticket, address=bad)
    return {"prompt": prompt_messages(complaint),
            "chosen": [{"role": "assistant", "content": ticket_json(chosen)}],
            "rejected": [{"role": "assistant", "content": ticket_json(rejected)}],
            "meta": {**meta, "complaint": complaint, "chosen_address": good, "rejected_address": bad}}


def norm(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=list(CONFIGS), default="v2")
    ap.add_argument("--out-dir", default=str(DATA_DIR), help="where to write (to re-check reproducibility)")
    args = ap.parse_args()
    cfg = CONFIGS[args.version]
    counts = cfg["counts"]
    rng = random.Random(cfg["seed"])

    # Everything the model is trained or evaluated on: no preference complaint may duplicate these.
    sources = ["train", "val", "test", "probe"] + (["preference_train"] if args.version != "v1" else [])
    existing = {s: {norm(r.get("complaint") or r["meta"]["complaint"]) for r in load_jsonl(DATA_DIR / f"{s}.jsonl")}
                for s in sources}
    forbidden = set().union(*existing.values())

    plan = [t for t, n in counts.items() for _ in range(n)]
    rng.shuffle(plan)
    cats = list(gen.ISSUES)
    pairs, seen, rejected_tries = [], set(), 0
    for i, pair_type in enumerate(plan):
        for _ in range(200):
            p = build_pair(pair_type, cats[i % len(cats)], rng)
            c, a, b = p["meta"]["complaint"], p["chosen"][0]["content"], p["rejected"][0]["content"]
            ok = (norm(c) not in forbidden and norm(c) not in seen
                  and max(len(a), len(b)) / min(len(a), len(b)) - 1 <= MAX_LEN_DIFF)
            gold = p["meta"]["chosen_address"]
            if pair_type == "full_address":
                ok = ok and gold in c
            else:  # the "bad" invented addresses must really be invented, and the text must contain no real address
                ok = ok and not ADDRESS_LIKE.search(c)
                if pair_type in ("no_address", "landmark"):
                    ok = ok and p["meta"]["rejected_address"] not in c
            if args.version != "v1":
                ok = ok and not any(s.lower() in c.lower() for s in PROBE_SPANS)
            if ok:
                break
            rejected_tries += 1
        else:
            raise RuntimeError(f"could not build a valid {pair_type} pair")
        seen.add(norm(c))
        p["id"] = f"pref{'' if args.version == 'v1' else args.version[1:]}-{i + 1:03d}"
        pairs.append(p)

    splits = {"train": pairs[:len(pairs) - cfg["n_val"]], "val": pairs[len(pairs) - cfg["n_val"]:]}
    out_dir = Path(args.out_dir)
    for split, fname in cfg["files"].items():
        with (out_dir / fname).open("w") as f:
            for p in splits[split]:
                f.write(json.dumps({"id": p["id"], **{k: v for k, v in p.items() if k != "id"}},
                                   ensure_ascii=False) + "\n")

    # ---- Report ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"[{args.version}] wrote " + ", ".join(f"{len(splits[s])} {s} pairs -> {fname}" for s, fname in cfg["files"].items())
          + f" ({rejected_tries} candidates discarded by checks)")
    print("\n== Pair types ==")
    print(f"  {'type':<20} {'all':>9} {'train':>6} {'val':>5}")
    for t in counts:
        n = lambda ps: sum(p["meta"]["pair_type"] == t for p in ps)
        print(f"  {t:<20} {n(pairs):>3} ({n(pairs) / len(pairs):>3.0%}) {n(splits['train']):>6} {n(splits['val']):>5}")
    present = sum(p["meta"]["pair_type"] in ADDRESS_PRESENT_TYPES for p in pairs)
    print(f"  address-present vs address-null: {present} vs {len(pairs) - present}")
    kinds = [k for p in pairs for k in p["meta"].get("distractor_kinds", [])]
    print("  distractor sentences by kind:", dict(sorted({k: kinds.count(k) for k in set(kinds)}.items())))

    print("\n== Contamination check ==")
    for s, texts in existing.items():
        print(f"  identical complaints shared with {s:<16}: {sum(norm(p['meta']['complaint']) in texts for p in pairs)}")
    if args.version != "v1":
        print(f"  pairs containing the probe's own distractor strings {PROBE_SPANS}: "
              f"{sum(any(s.lower() in p['meta']['complaint'].lower() for s in PROBE_SPANS) for p in pairs)}")

    print("\n== Length comparison: chosen vs rejected ticket ==")
    print(f"  {'pair type':<20} {'chosen chars':>13} {'rejected chars':>15} {'chosen tok':>11} {'rejected tok':>13} "
          f"{'max diff':>9}")
    for t in list(counts) + ["ALL"]:
        sub = [p for p in pairs if t == "ALL" or p["meta"]["pair_type"] == t]
        cc = [len(p["chosen"][0]["content"]) for p in sub]
        rc = [len(p["rejected"][0]["content"]) for p in sub]
        ct = [len(tok(p["chosen"][0]["content"])["input_ids"]) for p in sub]
        rt = [len(tok(p["rejected"][0]["content"])["input_ids"]) for p in sub]
        diff = max(max(a, b) / min(a, b) - 1 for a, b in zip(cc, rc))
        print(f"  {t:<20} {statistics.mean(cc):>13.1f} {statistics.mean(rc):>15.1f} {statistics.mean(ct):>11.1f} "
              f"{statistics.mean(rt):>13.1f} {diff:>8.1%}")
    shorter = sum(len(p["chosen"][0]["content"]) < len(p["rejected"][0]["content"]) for p in pairs)
    print(f"  chosen is the SHORTER response in {shorter}/{len(pairs)} pairs ({shorter / len(pairs):.0%})")


if __name__ == "__main__":
    main()
