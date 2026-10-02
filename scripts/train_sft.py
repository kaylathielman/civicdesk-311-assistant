"""QLoRA supervised fine-tuning of Qwen2.5-0.5B-Instruct on complaint -> ticket pairs.

- 4-bit NF4 base with double quantization, fp16 compute
- LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o projections
- Loss only on the assistant's JSON ticket (prompt tokens masked with -100)
- Peak GPU memory every 10 steps + total wall-clock -> logs/memory_time_log.csv (OOMs logged too)

Usage: python scripts/train_sft.py [--output-name sft_v1] [--epochs 3] [--lr 2e-4] ...
"""

import argparse
import json
import math
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from common import (ADAPTER_DIR, DATA_DIR, LOG_DIR, load_jsonl, load_model_and_tokenizer, log_memory_time,
                    prompt_messages, ticket_json)

SCRIPT = "train_sft.py"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-name", default="sft_v1", help="adapter saved to adapters/<name>")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.05, help="warmup as a fraction of total steps")
    ap.add_argument("--max-length", type=int, default=None, help="override the length picked from the data")
    ap.add_argument("--seed", type=int, default=311)
    return ap.parse_args()


def tokenize_example(ex, tok):
    """Full chat sequence, with labels = -100 on every prompt token so loss is only on the ticket."""
    prompt = tok.apply_chat_template(prompt_messages(ex["complaint"]), tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(
        prompt_messages(ex["complaint"]) + [{"role": "assistant", "content": ticket_json(ex["ticket"])}],
        tokenize=False)
    assert full.startswith(prompt), "chat template changed the prompt when the answer was appended"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids, "prompt tokens differ inside the full sequence"
    return {"input_ids": full_ids, "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}


def show_masking(example, tok):
    ids, labels = example["input_ids"], example["labels"]
    n_masked = sum(l == -100 for l in labels)
    print("\n== Loss masking check (first training example) ==")
    print(f"MASKED, no loss ({n_masked} tokens):\n{tok.decode(ids[:n_masked])!r}")
    print(f"\nTRAINED ON, loss computed ({len(ids) - n_masked} tokens):\n{tok.decode(ids[n_masked:])!r}")
    print("\nTokens around the boundary (x = masked, ✓ = trained):")
    for i in range(max(0, n_masked - 6), min(len(ids), n_masked + 6)):
        print(f"  {'x' if labels[i] == -100 else '✓'}  {tok.convert_ids_to_tokens(ids[i])!r}")


def pick_max_length(lengths):
    s = sorted(lengths)
    pct = lambda p: s[min(len(s) - 1, int(p / 100 * len(s)))]
    print("\n== Token lengths (prompt + ticket) ==")
    print(f"  min {s[0]}  p50 {pct(50)}  p90 {pct(90)}  p99 {pct(99)}  max {s[-1]}")
    # Data is small and tight, so cover 100% (never truncate a ticket) rounded up to a multiple of 32.
    max_len = math.ceil(s[-1] / 32) * 32
    print(f"  -> max_length = {max_len} (longest example rounded up to a multiple of 32; nothing truncated)")
    return max_len


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
        }


class MemoryTimeCallback(TrainerCallback):
    def __init__(self, run_name, start_time):
        self.run_name, self.start_time = run_name, start_time
        self.eval_losses = []

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % 10 == 0:
            log_memory_time(self.run_name, SCRIPT, "step", self.start_time, state.global_step, state.epoch)

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        loss = metrics["eval_loss"]
        self.eval_losses.append((round(state.epoch or 0, 2), round(loss, 4)))
        print(f"\n>>> validation loss after epoch {state.epoch or 0:.2f}: {loss:.4f}\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = args.output_name

    model, tok = load_model_and_tokenizer()
    train_rows, val_rows = load_jsonl(DATA_DIR / "train.jsonl"), load_jsonl(DATA_DIR / "val.jsonl")
    train = [tokenize_example(r, tok) for r in train_rows]
    val = [tokenize_example(r, tok) for r in val_rows]
    max_length = args.max_length or pick_max_length([len(e["input_ids"]) for e in train + val])
    too_long = sum(len(e["input_ids"]) > max_length for e in train + val)
    if too_long:
        print(f"WARNING: {too_long} examples longer than max_length={max_length} will be truncated")
    for e in train + val:
        e["input_ids"], e["labels"] = e["input_ids"][:max_length], e["labels"][:max_length]
    show_masking(train[0], tok)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    for p in model.parameters():  # fp16 mixed precision needs fp32 master weights for the trainable params
        if p.requires_grad:
            p.data = p.data.float()
    trainable, total = model.get_nb_trainable_parameters()
    print(f"\n== Trainable parameters ==\n  {trainable:,} trainable of {total:,} total ({100 * trainable / total:.3f}%)")

    targs = TrainingArguments(
        output_dir=str(LOG_DIR / "trainer_tmp" / run_name),
        num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=args.warmup, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True, eval_strategy="epoch", save_strategy="no", logging_steps=5, report_to="none",
        remove_unused_columns=False, seed=args.seed,
    )
    config = {"run_name": run_name, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
              "grad_accum": args.grad_accum, "warmup": args.warmup, "max_length": max_length,
              "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": "q,k,v,o"}, "quant": "nf4+double_quant"}

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    cb = MemoryTimeCallback(run_name, start)
    trainer = Trainer(model=model, args=targs, train_dataset=Dataset.from_list(train),
                      eval_dataset=Dataset.from_list(val), data_collator=Collator(tok.pad_token_id), callbacks=[cb])

    try:
        print("\n== Validation loss before training ==")
        trainer.evaluate()
        result = trainer.train()
    except torch.OutOfMemoryError as e:
        log_memory_time(run_name, SCRIPT, "oom", start, trainer.state.global_step, trainer.state.epoch or "",
                        notes=json.dumps({"config": config, "error": str(e)[:300]}))
        print(f"\nOUT OF MEMORY at step {trainer.state.global_step}. "
              f"Peak allocated {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB. Logged to logs/memory_time_log.csv.\n"
              "Try a smaller --batch-size (with larger --grad-accum) or a smaller --max-length.")
        sys.exit(1)

    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3
    summary = {"config": config, "train_loss": round(result.training_loss, 4), "val_loss_by_epoch": cb.eval_losses,
               "steps": result.global_step}
    log_memory_time(run_name, SCRIPT, "train_end", start, result.global_step, trainer.state.epoch,
                    notes=json.dumps(summary))

    out = ADAPTER_DIR / run_name
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_info.json").write_text(json.dumps({**summary, "peak_mem_gb": round(peak, 3),
                                                     "wall_clock_s": round(elapsed, 1),
                                                     "log_history": trainer.state.log_history}, indent=2))
    print(f"\n== Done ==\n  wall-clock {elapsed / 60:.1f} min | peak GPU memory {peak:.2f} GiB | "
          f"val loss by epoch {cb.eval_losses}\n  adapter saved to {out}")


if __name__ == "__main__":
    main()
