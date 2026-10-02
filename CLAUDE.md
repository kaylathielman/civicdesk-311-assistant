# CivicDesk 311 Assistant

Class assignment. Fine-tune a small LLM to turn messy resident complaints into structured 311 dispatch tickets.

## Goal
Input: free-text complaint from a resident (typos, rambling, missing details).
Output: a JSON ticket with exactly these fields:
- `category` — type of issue (e.g. pothole, noise, graffiti, streetlight)
- `urgency` — priority level
- `location` — location as described by the resident
- `summary` — one-line neutral summary for dispatchers
- `address` — street address **only if the complaint states one**, otherwise `null`

The model must never invent an address. Hallucinated addresses send crews to the wrong place.

## Approach
1. **SFT with QLoRA:** base model `Qwen/Qwen2.5-0.5B-Instruct`, loaded in 4-bit NF4 via bitsandbytes; train LoRA adapters with peft + trl.
2. **DPO (later):** preference pairs where "chosen" leaves `address: null` and "rejected" invents an address, to suppress hallucinated addresses.

## Environment
- Google Colab free tier, Tesla T4 GPU (~15 GB VRAM, ~14.5 GiB usable). No bf16 on T4, so use fp16 compute.
- Project lives on Google Drive: `/content/drive/MyDrive/civicdesk`. Colab runtimes reset, so anything outside Drive is lost; reinstall packages each session.
- Key libs: torch, transformers, peft, trl, bitsandbytes, accelerate, datasets.

## Layout
- `data/` — training/eval datasets (SFT examples, DPO preference pairs)
- `scripts/` — training, evaluation, and data-prep scripts
- `adapters/` — saved LoRA adapter weights (one subfolder per run)
- `logs/` — run logs, including `memory_time_log.csv`

## Setup & data
- New Colab session: `!bash /content/drive/MyDrive/civicdesk/scripts/setup.sh` (installs pinned `requirements.txt`, checks GPU).
- `scripts/generate_instruction_data.py` (seed 311) → `data/{train,val,test}.jsonl` (340/30/30, stratified by address case: 70% full address / 15% no location / 15% landmark only). Record: `{id, complaint, ticket, meta}`; `location` is null when no location is given.
- `scripts/audit_data.py` must pass after any data change. Summaries must only state facts present in the complaint (no borrowed details), same principle as addresses.

## Training & eval (run from `scripts/`)
- `common.py` — shared system prompt, 4-bit model loading, `log_memory_time()` for the CSV log. Train and eval must use the same prompt.
- `train_sft.py --output-name sft_v1` — QLoRA SFT (prompt tokens masked). ~2.6 min, ~2.8 GiB peak on T4.
- `evaluate.py [--adapter ../adapters/X]` — test (30) + `data/probe.jsonl` (20 hand-written, different style; 10 with no address, several with phone/unit/case numbers as traps). Writes `logs/eval_<name>_*`.
- Baseline vs sft_v1 results are in `logs/eval_base_metrics.json` and `logs/eval_sft_v1_metrics.json`.
- `generate_preference_data.py` (seed 512) → `data/preference_train.jsonl` (150 DPO pairs; chosen/rejected differ only in `address`; complaints checked against train/val/test/probe). `check_length_shortcut.py` tests whether length predicts chosen.
- `train_dpo.py --output-name dpo_v1 [--beta 0.1]` — DPO on top of sft_v1. trl 1.14 has no `ref_adapter_name`: DPOTrainer auto-copies the loaded adapter into a frozen "ref" adapter; the script asserts the ref equals sft_v1 after training. trl casts QLoRA adapters to bf16, script recasts to fp32 (T4). Run with nohup, output in `logs/train_dpo_<name>.out`.
- `compare_evals.py` — side-by-side table with counts → `logs/comparison.md`.
- `evaluate.py` also flags location fabrication (street address/number in location not in complaint) and salvages address/location from broken JSON for fabrication checks.

## Run history (tuning stopped after dpo_v2, by decision on 2026-10-02)
- **sft_v1** — best overall model.
- **dpo_v1** — DOCUMENTED FAILED RUN, keep it. beta 0.1, lr 2e-5, 2 epochs, 30% address pairs. Over-optimized (train reward acc 1.0, margin 4.7): broke JSON (5/30 invalid), category 90%→67%, code-switching. Did not fix probe fabrications.
- **dpo_v2** — beta 0.5, lr 5e-6, 1 epoch, `preference_v2_*` (50/50, + location_distractor pairs, 20 held out; held-out reward acc 0.90). Kept SFT quality but changed few outputs (6/30 test); did NOT fix probe-14/15 (phone/unit copied to address); like dpo_v1 it newly copies landmarks into `address` on 2 test cases. Likely cause: full_address pairs reward "copy text into address", while landmark pairs only penalize *invented* house numbers, never a copied landmark.
- v2 shortcut check: raw length ~50%, but pairwise LR on chars+tokens is 69% via digit density (digits = 1 token each), which tracks the lesson itself, not length.
- `logs/memory_time_log.csv` has an `oom_test` row from a deliberate OOM check of the error handler (batch size 340), not a real run.

## Rules
- **Every training script must append a row to `logs/memory_time_log.csv`** recording at least: run name, script, peak GPU memory (`torch.cuda.max_memory_allocated()`, call `torch.cuda.reset_peak_memory_stats()` before training), and wall-clock training time. Write the header if the file doesn't exist.
- Keep memory within the T4 budget: small batch sizes with gradient accumulation, gradient checkpointing if needed.
- The user is new to ML fine-tuning: explain steps in plain English.
