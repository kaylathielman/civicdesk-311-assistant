"""Evaluate a model (base, or base + LoRA adapter) on the test set and the hand-written probe set.

Metrics (per set):
  json_strict      - whole output is a JSON object
  json_parseable   - a JSON object can be extracted (e.g. from inside ```json fences); used for all other metrics
  category_acc / urgency_acc - exact match, over ALL examples (unparseable counts as wrong)
  fabrication_rate - gold address is null, model gave a non-null address
                     (split into: text not in complaint at all = invented, vs. copied from complaint = misfiled)
  location_fabrication - ANY example whose location contains a street address or number not in the complaint
                     (nested dict locations from the base model are flattened first)
  any_fabrication  - either of the above
  Unparseable outputs still get address/location pulled out by regex for the fabrication checks, so broken
  JSON can't hide a fabrication (it still counts as invalid JSON and wrong category/urgency).
  false_null_rate  - gold has an address, model left it null
  address_exact    - gold has an address, model's matches it (ignoring case/punctuation)

Outputs: logs/eval_<name>_<set>.jsonl (every generation) and logs/eval_<name>_metrics.json
Usage: python scripts/evaluate.py                      # base model
       python scripts/evaluate.py --adapter adapters/sft_v1
"""

import argparse
import json
import re
from pathlib import Path

import torch

from audit_data import ADDRESS_LIKE
from common import DATA_DIR, LOG_DIR, TICKET_KEYS, load_jsonl, load_model_and_tokenizer, prompt_messages

NULLISH = {"", "null", "none", "n/a", "na", "unknown", "not provided", "not specified", "not given"}


def parse_output(raw):
    text = raw.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, True
    except json.JSONDecodeError:
        pass
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj, False
        except json.JSONDecodeError:
            pass
    return None, False


def salvage_fields(raw):
    """Regex-extract address/location from broken JSON, e.g. '"address": "2241 Center Pl}'."""
    out = {}
    for key in ("address", "location"):
        m = re.search(rf'"{key}"\s*:\s*(null|"([^"}}\n]*))', raw)
        if m:
            out[key] = None if m.group(1) == "null" else m.group(2)
    return out


def flatten(value):
    if isinstance(value, dict):
        return " ".join(flatten(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(flatten(v) for v in value)
    return "" if value is None else str(value)


def location_fabrications(location, complaint):
    """Street addresses or numbers in the location that don't appear in the complaint."""
    loc, hits = flatten(location), []
    for m in ADDRESS_LIKE.finditer(loc):
        if squash(m.group()) not in squash(complaint):
            hits.append(m.group())
    complaint_numbers = set(re.findall(r"\d+", complaint))
    hits += [n for n in re.findall(r"\d+", loc) if n not in complaint_numbers and not any(n in h for h in hits)]
    return hits


def norm_addr(value):
    """Treat 'N/A', 'unknown', '' etc. as null so the base model isn't penalized for wording."""
    if value is None or not isinstance(value, str):
        return None if value is None else str(value)
    return None if value.strip().lower() in NULLISH else value.strip()


def squash(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


@torch.no_grad()
def generate(model, tok, complaints, batch_size, max_new_tokens):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(complaints), batch_size):
        prompts = [tok.apply_chat_template(prompt_messages(c), tokenize=False, add_generation_prompt=True)
                   for c in complaints[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def score(rows, outputs):
    records, m = [], dict(n=len(rows), strict=0, parseable=0, schema=0, cat=0, urg=0,
                          null_gold=0, fabricated=0, invented=0, misfiled=0, null_gold_unparseable=0,
                          addr_gold=0, false_null=0, addr_exact=0, addr_gold_unparseable=0,
                          loc_fab=0, any_fab=0, salvaged=0)
    for row, raw in zip(rows, outputs):
        gold, (pred, strict) = row["ticket"], parse_output(raw)
        m["strict"] += strict
        rec = {"id": row["id"], "complaint": row["complaint"], "gold": gold, "raw_output": raw, "parsed": pred,
               "address_case": row["meta"]["address_case"], "errors": []}
        if pred is None:
            rec["errors"].append("unparseable")
        else:
            m["parseable"] += 1
            m["schema"] += set(pred) == set(TICKET_KEYS)
            if pred.get("category") == gold["category"]:
                m["cat"] += 1
            else:
                rec["errors"].append("category")
            if str(pred.get("urgency", "")).lower() == gold["urgency"]:
                m["urg"] += 1
            else:
                rec["errors"].append("urgency")
        fields = pred if pred is not None else salvage_fields(raw)
        m["salvaged"] += pred is None and bool(fields)
        pred_addr = norm_addr(fields.get("address"))
        loc_hits = location_fabrications(fields.get("location"), row["complaint"])
        if loc_hits:
            m["loc_fab"] += 1
            rec["errors"].append(f"location_fabricated {loc_hits}")
        address_fab = False
        if gold["address"] is None:
            m["null_gold"] += 1
            if pred is None and not fields:
                m["null_gold_unparseable"] += 1
            elif pred_addr is not None:
                address_fab = True
                m["fabricated"] += 1
                kind = "misfiled" if squash(pred_addr) in squash(row["complaint"]) else "invented"
                m[kind] += 1
                rec["errors"].append(f"fabricated_address ({kind})")
        else:
            m["addr_gold"] += 1
            if pred is None:
                m["addr_gold_unparseable"] += 1
            elif pred_addr is None:
                m["false_null"] += 1
                rec["errors"].append("false_null")
            elif squash(pred_addr) == squash(gold["address"]):
                m["addr_exact"] += 1
            else:
                rec["errors"].append("address_mismatch")
        m["any_fab"] += bool(address_fab or loc_hits)
        records.append(rec)

    pct = lambda a, b: round(100 * a / b, 1) if b else None
    metrics = {
        "n": m["n"],
        "json_strict_%": pct(m["strict"], m["n"]),
        "json_parseable_%": pct(m["parseable"], m["n"]),
        "schema_exact_keys_%": pct(m["schema"], m["n"]),
        "category_acc_%": pct(m["cat"], m["n"]),
        "urgency_acc_%": pct(m["urg"], m["n"]),
        "fabrication_rate_%": pct(m["fabricated"], m["null_gold"]),
        "fabrication_counts": f"{m['fabricated']}/{m['null_gold']} (invented {m['invented']}, "
                              f"misfiled {m['misfiled']}, unparseable {m['null_gold_unparseable']})",
        "location_fabrication_%": pct(m["loc_fab"], m["n"]),
        "location_fabrication_counts": f"{m['loc_fab']}/{m['n']}",
        "any_fabrication_%": pct(m["any_fab"], m["n"]),
        "any_fabrication_counts": f"{m['any_fab']}/{m['n']}",
        "unparseable_salvaged_for_fabrication": m["salvaged"],
        "false_null_rate_%": pct(m["false_null"], m["addr_gold"]),
        "false_null_counts": f"{m['false_null']}/{m['addr_gold']} (unparseable {m['addr_gold_unparseable']})",
        "address_exact_%": pct(m["addr_exact"], m["addr_gold"]),
    }
    return metrics, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter; omit for the base model")
    ap.add_argument("--name", default=None, help="tag for output files (default: base or adapter folder name)")
    ap.add_argument("--sets", nargs="+", default=["test", "probe"])
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    name = args.name or (Path(args.adapter).name if args.adapter else "base")

    model, tok = load_model_and_tokenizer(args.adapter)
    model.eval()
    LOG_DIR.mkdir(exist_ok=True)
    all_metrics = {}
    for s in args.sets:
        rows = load_jsonl(DATA_DIR / f"{s}.jsonl")
        outputs = generate(model, tok, [r["complaint"] for r in rows], args.batch_size, args.max_new_tokens)
        metrics, records = score(rows, outputs)
        all_metrics[s] = metrics
        with (LOG_DIR / f"eval_{name}_{s}.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n== {name} on {s} ({len(rows)} examples) ==")
        for k, v in metrics.items():
            print(f"  {k:<22} {v}")
    (LOG_DIR / f"eval_{name}_metrics.json").write_text(
        json.dumps({"model": name, "adapter": args.adapter, **all_metrics}, indent=2))
    print(f"\nSaved generations and metrics to {LOG_DIR}/eval_{name}_*")


if __name__ == "__main__":
    main()
