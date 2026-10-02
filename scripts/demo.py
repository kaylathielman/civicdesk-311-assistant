"""Turn one resident complaint into a dispatch ticket.

Usage:
  python scripts/demo.py "There's a huge pothole outside 412 Elm St, my tire popped"
  python scripts/demo.py --adapter dpo_v2 "streetlight out by the library"
  python scripts/demo.py --adapter none "..."      # base model, no adapter
"""

import argparse
import json

import torch

from common import ADAPTER_DIR, load_model_and_tokenizer, prompt_messages
from evaluate import parse_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("complaint", help="the resident's complaint text (quote it)")
    ap.add_argument("--adapter", default="sft_v1", help="adapter folder in adapters/, or 'none' for the base model")
    args = ap.parse_args()

    adapter = None if args.adapter == "none" else ADAPTER_DIR / args.adapter
    if adapter and not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}")
    model, tok = load_model_and_tokenizer(adapter)
    model.eval()

    prompt = tok.apply_chat_template(prompt_messages(args.complaint), tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
    raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    ticket, _ = parse_output(raw)
    print(f"Model: base + {args.adapter}" if adapter else "Model: base (no adapter)")
    print(f"Complaint: {args.complaint}\n")
    if ticket is None:
        print("WARNING: model output is not valid JSON. Raw output:\n" + raw)
    else:
        print(json.dumps(ticket, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
