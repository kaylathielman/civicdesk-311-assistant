"""DPO on top of the SFT adapter: prefer `address: null` over invented addresses, but keep real ones.

Reference model = the SFT adapter (frozen), NOT the raw base. trl 1.14 removed `ref_adapter_name`; instead,
when given a model that already has a LoRA adapter ("default"), DPOTrainer copies it into a frozen adapter
named "ref" and uses that for reference log-probs. This script verifies that after training: "ref" still
equals adapters/<sft>, and "default" (the trainable copy) has changed.

Logs reward margin + reward accuracy (+ peak GPU memory) every 10 steps, and total wall-clock,
to logs/memory_time_log.csv. Out-of-memory errors are logged with the config instead of crashing.

Usage (background):
  # v1 (documented failed run: over-optimized, broke JSON formatting)
  nohup python train_dpo.py --output-name dpo_v1 > ../logs/train_dpo_v1.out 2>&1 &
  # v2 (gentler, 50/50 data, held-out pairs)
  nohup python train_dpo.py --output-name dpo_v2 --beta 0.5 --lr 5e-6 --epochs 1 \
      --train-file preference_v2_train.jsonl --val-file preference_v2_val.jsonl > ../logs/train_dpo_v2.out 2>&1 &
"""

import argparse
import json
import sys
import time

import torch
from datasets import Dataset
from peft import PeftModel
from safetensors.torch import load_file
from transformers import TrainerCallback, set_seed
from trl import DPOConfig, DPOTrainer

from common import ADAPTER_DIR, DATA_DIR, LOG_DIR, load_jsonl, load_model_and_tokenizer, log_memory_time

SCRIPT = "train_dpo.py"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", default="sft_v1", help="adapters/<name> to start from and use as reference")
    ap.add_argument("--output-name", default="dpo_v1")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta: higher = stay closer to the SFT reference")
    ap.add_argument("--lr", type=float, default=2e-5, help="10x below the SFT lr; DPO is a gentle nudge")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=311)
    ap.add_argument("--train-file", default="preference_train.jsonl", help="in data/")
    ap.add_argument("--val-file", default=None, help="held-out pairs in data/; reward accuracy reported on them")
    return ap.parse_args()


class RewardMemoryCallback(TrainerCallback):
    """Every logging step (10): write reward margin/accuracy and peak memory to the CSV."""

    def __init__(self, run_name, start_time):
        self.run_name, self.start_time, self.history = run_name, start_time, []

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        row = {k.removeprefix("eval_"): round(float(v), 4) for k, v in (metrics or {}).items()
               if k in ("eval_loss", "eval_rewards/margins", "eval_rewards/accuracies")}
        self.history.append({"step": state.global_step, "split": "val", **row})
        log_memory_time(self.run_name, SCRIPT, "eval", self.start_time, state.global_step, state.epoch,
                        notes=json.dumps({"split": "val", **row}))
        print(f">>> HELD-OUT step {state.global_step}: reward margin {row['rewards/margins']:.3f}, "
              f"reward accuracy {row['rewards/accuracies']:.2f}, loss {row['loss']:.4f}", flush=True)

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "rewards/margins" not in logs or "loss" not in logs:
            return
        row = {k: round(float(logs[k]), 4) for k in ("loss", "rewards/margins", "rewards/accuracies",
                                                      "rewards/chosen", "rewards/rejected") if k in logs}
        self.history.append({"step": state.global_step, **row})
        log_memory_time(self.run_name, SCRIPT, "step", self.start_time, state.global_step, state.epoch,
                        notes=json.dumps(row))
        print(f">>> step {state.global_step}: reward margin {row['rewards/margins']:.3f}, "
              f"reward accuracy {row['rewards/accuracies']:.2f}, loss {row.get('loss', float('nan')):.4f}", flush=True)


def lora_weights(model, adapter_name):
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if f".{adapter_name}." in n}


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name, sft_path = args.output_name, ADAPTER_DIR / args.sft_adapter

    base, tok = load_model_and_tokenizer()
    model = PeftModel.from_pretrained(base, sft_path, is_trainable=True)  # adapter "default", trainable

    to_ds = lambda rows: Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])
    rows = load_jsonl(DATA_DIR / args.train_file)
    dataset = to_ds(rows)
    val_rows = load_jsonl(DATA_DIR / args.val_file) if args.val_file else None

    cfg = DPOConfig(
        output_dir=str(LOG_DIR / "trainer_tmp" / run_name), beta=args.beta, learning_rate=args.lr,
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type="cosine", warmup_steps=0.1,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True, bf16=False, max_length=512, logging_steps=10, save_strategy="no", report_to="none",
        seed=args.seed,
        eval_strategy="steps" if val_rows else "no", eval_steps=10, per_device_eval_batch_size=args.batch_size,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok,
                         eval_dataset=to_ds(val_rows) if val_rows else None)
    model = trainer.model

    # Confirm the reference is a frozen copy of the SFT adapter.
    assert "ref" in model.peft_config, "trl did not create a 'ref' adapter; reference would be the raw base model"
    for n, p in model.named_parameters():
        if ".ref." in n:
            p.requires_grad_(False)
        elif ".default." in n and "lora_" in n:
            # trl casts QLoRA adapters to bf16; T4 has no native bf16 and small DPO updates can round away in bf16.
            p.requires_grad_(True)
            p.data = p.data.float()
    sft_saved = load_file(str(sft_path / "adapter_model.safetensors"))
    ref_before, default_before = lora_weights(model, "ref"), lora_weights(model, "default")
    print(f"Reference = frozen copy of {sft_path.name} ({len(ref_before)} LoRA tensors); "
          f"trainable = second copy ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params)")

    config = {"run_name": run_name, "sft_adapter": args.sft_adapter, "beta": args.beta, "lr": args.lr,
              "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
              "n_pairs": len(rows), "train_file": args.train_file,
              "val_file": args.val_file, "n_val_pairs": len(val_rows or []), "reference": f"frozen copy of {args.sft_adapter}"}
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    cb = RewardMemoryCallback(run_name, start)
    trainer.add_callback(cb)
    try:
        result = trainer.train()
        final_val = trainer.evaluate() if val_rows else None
    except torch.OutOfMemoryError as e:
        log_memory_time(run_name, SCRIPT, "oom", start, trainer.state.global_step, trainer.state.epoch or "",
                        notes=json.dumps({"config": config, "error": str(e)[:300]}))
        print(f"OUT OF MEMORY at step {trainer.state.global_step}; logged to logs/memory_time_log.csv. "
              "Try a smaller --batch-size with a larger --grad-accum.", flush=True)
        sys.exit(1)
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 1024**3

    # Verify: reference untouched and identical to the saved SFT adapter; trainable copy moved.
    ref_after, default_after = lora_weights(model, "ref"), lora_weights(model, "default")
    ref_unchanged = all(torch.equal(ref_before[n], ref_after[n]) for n in ref_before)
    # Saved keys omit the adapter name: "...q_proj.lora_A.weight" vs in-memory "...q_proj.lora_A.ref.weight".
    ref_matches_sft = all(torch.allclose(ref_after[n], sft_saved[n.replace(".ref.", ".")].float(), atol=1e-3)
                          for n in ref_after)
    moved = max((default_after[n] - default_before[n]).abs().max().item() for n in default_before)
    print(f"Check: reference unchanged during training = {ref_unchanged}; reference == saved SFT adapter = "
          f"{ref_matches_sft}; max change in trainable adapter = {moved:.2e}", flush=True)

    summary = {"config": config, "train_loss": round(result.training_loss, 4), "steps": result.global_step,
               "final_val": {k: round(float(v), 4) for k, v in (final_val or {}).items()
                             if k.startswith("eval_rewards") or k == "eval_loss"},
               "reward_log": cb.history, "ref_unchanged": ref_unchanged, "ref_matches_sft": ref_matches_sft}
    log_memory_time(run_name, SCRIPT, "train_end", start, result.global_step, trainer.state.epoch,
                    notes=json.dumps({k: summary[k] for k in ("config", "train_loss", "steps")}))

    out = ADAPTER_DIR / run_name
    model.save_pretrained(out, selected_adapters=["default"])
    tok.save_pretrained(out)
    (out / "train_info.json").write_text(json.dumps({**summary, "peak_mem_gb": round(peak, 3),
                                                     "wall_clock_s": round(elapsed, 1)}, indent=2))
    print(f"Done: {elapsed / 60:.1f} min, peak GPU memory {peak:.2f} GiB, adapter saved to {out}", flush=True)


if __name__ == "__main__":
    main()
