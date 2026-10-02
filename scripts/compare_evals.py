"""Side-by-side comparison of evaluate.py results, with counts, plus the probe no-address outputs.

Reads logs/eval_<model>_<set>.jsonl (written by evaluate.py). Also writes logs/comparison.md.
Usage: python scripts/compare_evals.py [--models base sft_v1 dpo_v1] [--sets test probe]
"""

import argparse
import json

from common import LOG_DIR, load_jsonl


def counts(records):
    has = lambda r, e: any(x.startswith(e) for x in r["errors"])
    parsed = [r for r in records if r["parsed"] is not None]
    null_gold = [r for r in records if r["gold"]["address"] is None]
    addr_gold = [r for r in records if r["gold"]["address"] is not None]
    return {
        "Valid JSON": (len(parsed), len(records)),
        "Category correct": (sum(not has(r, "category") for r in parsed), len(records)),
        "Urgency correct": (sum(not has(r, "urgency") for r in parsed), len(records)),
        "Fabrication (no-address cases)": (sum(has(r, "fabricated") for r in null_gold), len(null_gold)),
        "  of which invented": (sum(has(r, "fabricated_address (invented)") for r in null_gold), len(null_gold)),
        "  of which copied non-address": (sum(has(r, "fabricated_address (misfiled)") for r in null_gold),
                                          len(null_gold)),
        "Location fabrication (all cases)": (sum(has(r, "location_fabricated") for r in records), len(records)),
        "ANY fabrication (all cases)": (sum(has(r, "fabricated") or has(r, "location_fabricated") for r in records),
                                        len(records)),
        "False-null (address cases)": (sum(has(r, "false_null") for r in addr_gold), len(addr_gold)),
        "Address exact match": (sum(r["parsed"] is not None and not has(r, "false_null")
                                    and not has(r, "address_mismatch") for r in addr_gold), len(addr_gold)),
    }


def fmt(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "sft_v1", "dpo_v1", "dpo_v2"])
    ap.add_argument("--sets", nargs="+", default=["test", "probe"])
    args = ap.parse_args()

    recs = {(m, s): load_jsonl(LOG_DIR / f"eval_{m}_{s}.jsonl") for m in args.models for s in args.sets}
    md = []
    for s in args.sets:
        table = {m: counts(recs[m, s]) for m in args.models}
        header = f"| {s} ({len(recs[args.models[0], s])} examples) | " + " | ".join(args.models) + " |"
        md += [header, "|" + "---|" * (len(args.models) + 1)]
        for metric in table[args.models[0]]:
            md.append(f"| {metric} | " + " | ".join(fmt(*table[m][metric]) for m in args.models) + " |")
        md.append("")
    print("\n".join(md))

    if "probe" in args.sets:
        out = ["\n== Probe complaints with NO address: what each model wrote ==\n"]
        first = recs[args.models[0], "probe"]
        for i, r in enumerate(first):
            if r["gold"]["address"] is not None:
                continue
            out.append(f"[{r['id']}] {r['complaint']!r}")
            for m in args.models:
                rec = recs[m, "probe"][i]
                p = rec["parsed"] or {}
                flag = "  <-- FABRICATED" if any(e.startswith(("fabricated", "location_fabricated"))
                                                 for e in rec["errors"]) else ""
                if rec["parsed"] is None:
                    out.append(f"   {m:<7} INVALID JSON: {rec['raw_output'][:160]!r}{flag}")
                    continue
                out.append(f"   {m:<7} address={p.get('address')!r}  location={p.get('location')!r}  "
                           f"category={p.get('category')!r} urgency={p.get('urgency')!r}{flag}")
            out.append("")
        print("\n".join(out))
        md += ["```"] + out + ["```"]
    (LOG_DIR / "comparison.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
