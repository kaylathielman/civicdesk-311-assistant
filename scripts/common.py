"""Shared pieces for training and evaluation: prompt format, model loading, memory/time logging."""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, LOG_DIR, ADAPTER_DIR = ROOT / "data", ROOT / "logs", ROOT / "adapters"
MEMORY_LOG = LOG_DIR / "memory_time_log.csv"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

CATEGORIES = ["pothole", "streetlight", "graffiti", "missed_trash_pickup", "noise", "water_leak",
              "abandoned_vehicle", "sidewalk", "tree_damage", "animal_control"]
TICKET_KEYS = ["category", "urgency", "location", "summary", "address"]

SYSTEM_PROMPT = (
    "You are a 311 dispatch assistant. Rewrite the resident's complaint as a JSON ticket with keys: "
    f"category (one of: {', '.join(CATEGORIES)}), urgency (low, medium, or high), "
    "location (where the problem is, as described, or null), summary (one short neutral sentence), "
    "address (the street address copied exactly from the complaint, or null if none is given). "
    "Output only the JSON."
)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_messages(complaint):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": complaint}]


def ticket_json(ticket):
    return json.dumps({k: ticket[k] for k in TICKET_KEYS})


def bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16)


def load_model_and_tokenizer(adapter=None):
    """Base model in 4-bit NF4 (same as training), optionally with a LoRA adapter on top."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config(),
                                                 dtype=torch.float16, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok


MEMORY_LOG_FIELDS = ["timestamp", "run_name", "script", "event", "step", "epoch", "peak_mem_alloc_gb",
                     "peak_mem_reserved_gb", "elapsed_s", "notes"]


def log_memory_time(run_name, script, event, start_time, step="", epoch="", notes=""):
    """Append one row to logs/memory_time_log.csv (header written on first use)."""
    LOG_DIR.mkdir(exist_ok=True)
    new = not MEMORY_LOG.exists()
    with MEMORY_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MEMORY_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_name": run_name, "script": script, "event": event, "step": step,
            "epoch": f"{epoch:.2f}" if isinstance(epoch, float) else epoch,
            "peak_mem_alloc_gb": f"{torch.cuda.max_memory_allocated() / 1024**3:.3f}",
            "peak_mem_reserved_gb": f"{torch.cuda.max_memory_reserved() / 1024**3:.3f}",
            "elapsed_s": f"{time.time() - start_time:.1f}",
            "notes": notes,
        })
