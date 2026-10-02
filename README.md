# CivicDesk 311 Assistant

Turns messy resident complaints into structured dispatch tickets with a small, cheap model that fine-tunes on a free Colab GPU.

```
"ugh someone dumped like 6 trash bags next to the bus shelter on the corner and now theres raccoons all over it"
```
```json
{"category": "missed_trash_pickup", "urgency": "medium", "location": "next to the bus shelter on the corner",
 "summary": "Trash bag dump reported.", "address": null}
```

The rule that matters most: **the model must never invent an address.** A made-up address sends a crew to the wrong place. If the complaint has no street address, `address` must be `null`.

This is a class project and a prototype, **not production-ready**. See [Limitations](#limitations).

## What it does

| Field | Meaning |
|---|---|
| `category` | One of: pothole, streetlight, graffiti, missed_trash_pickup, noise, water_leak, abandoned_vehicle, sidewalk, tree_damage, animal_control |
| `urgency` | low / medium / high |
| `location` | Where the problem is, as the resident described it (address or landmark), or `null` |
| `summary` | One short, neutral sentence for dispatchers |
| `address` | Street address copied exactly from the complaint, or `null` |

**Model:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), loaded in 4-bit (NF4, QLoRA), with a small LoRA adapter (2.2M trainable parameters, 0.44% of the model).

## Which adapter to use: `adapters/sft_v1`

| Adapter | Status | Why |
|---|---|---|
| **`sft_v1`** | **Use this** | Best overall: 100% valid JSON, 90% category accuracy on test, copies real addresses correctly (90% test, 80% probe), never leaves a real address blank, and invents no addresses on the test set. |
| `dpo_v1` | Failed run, kept for the record | Preference tuning was too aggressive. It broke JSON formatting (5/30 invalid), dropped category accuracy to 67%, and didn't fix fabrication. |
| `dpo_v2` | Not recommended | Stable, but barely changed the model. It didn't fix the phone/unit failures and added 2 cases of copying a landmark into `address`. |

Headline numbers (test = 30 generated complaints; probe = 20 hand-written complaints in different styles):

| | base model | **sft_v1** | dpo_v1 | dpo_v2 |
|---|---|---|---|---|
| Valid JSON (test) | 30/30 | **30/30** | 25/30 | 30/30 |
| Category correct (test / probe) | 5/30 / 6/20 | **27/30 / 17/20** | 20/30 / 14/20 | 27/30 / 16/20 |
| Urgency correct (test / probe) | 11/30 / 7/20 | **12/30 / 10/20** | 12/30 / 8/20 | 12/30 / 10/20 |
| Any fabrication (test / probe) | 5/30 / 3/20 | **1/30 / 2/20** | 2/30 / 2/20 | 3/30 / 2/20 |
| Real address left blank (test+probe) | 27/31 | **0/31** | 0/31 | 0/31 |

Full results, logs and diagnosis: [`logs/results_summary.md`](logs/results_summary.md).

## Hardware budget (Colab free tier, Tesla T4, 15 GB)

| Run | Peak GPU memory | Wall-clock |
|---|---|---|
| SFT (`train_sft.py`, 340 examples × 3 epochs) | 2.79 GiB | 2.6 min |
| DPO v1 (150 pairs × 2 epochs) | 3.74 GiB | 1.0 min |
| DPO v2 (150 pairs × 1 epoch + 20 held-out) | 3.69 GiB | 0.6 min |
| Evaluation (50 complaints per model) | — (not logged; inference only) | ~1 min |

Everything fits in under 4 GiB of the T4's 14.5 GiB usable, so the full pipeline runs in about 15 minutes. Every training run appends peak memory and wall-clock time to `logs/memory_time_log.csv`, including out-of-memory failures (logged with their config instead of crashing).

## How to run (in order)

All commands assume Colab with the repo on Google Drive at `/content/drive/MyDrive/civicdesk`. Run them from `scripts/`.

```bash
# 0. Every new Colab session: install pinned packages and check the GPU
bash scripts/setup.sh
cd scripts

# 1. SFT data: 400 synthetic complaint -> ticket pairs, split 340/30/30, then audit them
python generate_instruction_data.py
python audit_data.py                       # must print "All checks passed."

# 2. Baseline: evaluate the untuned base model on test + probe
python evaluate.py

# 3. Supervised fine-tuning -> adapters/sft_v1
python train_sft.py --output-name sft_v1
python evaluate.py --adapter ../adapters/sft_v1

# 4. (Optional, experimental) Preference data + DPO. Neither run beat sft_v1.
python generate_preference_data.py --version v2
python check_length_shortcut.py ../data/preference_v2_train.jsonl
nohup python train_dpo.py --output-name dpo_v2 --beta 0.5 --lr 5e-6 --epochs 1 \
    --train-file preference_v2_train.jsonl --val-file preference_v2_val.jsonl > ../logs/train_dpo_v2.out 2>&1 &
python evaluate.py --adapter ../adapters/dpo_v2

# 5. Compare models and rebuild the results summary
python compare_evals.py                    # -> logs/comparison.md
python make_results_summary.py             # -> logs/results_summary.md

# Try it
python demo.py "The streetlight outside 2210 Fernwood Avenue has been dark for a week"
python demo.py --adapter none "..."        # base model, for comparison
```

To reproduce the failed `dpo_v1` run: `python generate_preference_data.py --version v1`, then `python train_dpo.py --output-name dpo_v1`, which uses the default settings beta 0.1, lr 2e-5 and 2 epochs.

| Script | Purpose |
|---|---|
| `setup.sh` | Installs `requirements.txt` (exact pinned versions) and checks the GPU |
| `common.py` | Shared system prompt, 4-bit model loading, memory/time logging |
| `generate_instruction_data.py` | Synthetic SFT data (seed 311) |
| `audit_data.py` | Diversity, address-case mix, address grounding and duplicate checks |
| `train_sft.py` | QLoRA SFT; loss only on the ticket tokens |
| `evaluate.py` | JSON validity, category/urgency accuracy, fabrication, false-null, address match |
| `generate_preference_data.py` | DPO chosen/rejected pairs (v1 seed 512, v2 seed 513) |
| `check_length_shortcut.py` | Checks whether response length alone predicts the preferred answer |
| `train_dpo.py` | DPO on top of `sft_v1`; the reference model is a frozen copy of `sft_v1` |
| `compare_evals.py`, `make_results_summary.py` | Reporting |
| `demo.py` | One complaint in, one ticket out |

## Data card

**Source.** All training and test data is **synthetic**, built from templates by `generate_instruction_data.py` with a fixed seed. No real resident data is used. There are 10 issue categories with 2–4 sub-issues each. Urgency comes from the sub-issue (a sparking power line is high, a flickering light is low), never from the resident's tone. Complaints are deliberately messy: four tones (angry, polite, rambling, terse), typos, slang, run-on sentences, missing punctuation and ALL CAPS.

**Address cases** (exact counts): 70% include a full street address (280), 15% have no location (60), and 15% have only a landmark like "behind the Walgreens on 7th" (60). Addresses and landmarks are never altered by the typo and slang noise, so the ticket can copy them word-for-word. Each summary is written only from facts in the complaint's wording, so the model isn't taught to add details.

**Split.** 340 train / 30 validation / 30 test, stratified so every split has all three address cases.

**Audit** (`audit_data.py`, all passing):
- every ticket address and landmark appears word-for-word in its complaint (280/280 and 60/60)
- an address detector finds nothing address-like in the 120 no-address complaints; the same detector catches 280/280 real addresses
- 0 duplicate complaints within or across splits
- 171 distinct three-word openings across 400 complaints, and 163 across summaries
- summaries are 2–17 words, with a mean of 5.4

**Probe set** (`data/probe.jsonl`). 20 complaints written by hand in styles the generator never produces: a formal letter, texts with emoji, a voicemail transcript, web-form fields, bullet points, Spanglish and questions. 10 have no address, and several include numbers that aren't addresses (phone, unit, case number, bus route, highway exit). It tests generalization beyond the templates.

**Preference data** (DPO only). New complaints from the same generator with different seeds, checked to share nothing with train/val/test/probe. Chosen and rejected tickets differ only in `address` (and in `location` for one v2 pair type), with lengths within 15%. v2 is 50/50 address-present vs address-null, with 20 pairs held out.

## Limitations

- **Template-generated data.** Training and test complaints come from the same templates, so test scores overstate real-world performance. The 20-complaint probe set is the only out-of-template check. Real 311 messages will be messier and more varied.
- **Small test sets.** 30 test and 20 probe examples: one example moves a rate by 3–5 points. Treat differences of 1–2 examples as noise, and quote counts, not just percentages.
- **Urgency is still weak.** About 40% correct on test and 50% on probe, close to the base model. The model mostly confuses neighbors (medium vs. high). Don't rely on the model's urgency for triage.
- **Known fabrication failures are not fixed.** `sft_v1` copies a phone number (`"555-0172"`) and `"unit 12 of the townhomes"` into `address` on the probe set. Both DPO runs also copy landmarks into `address` on 2 test cases. These are copied, not invented, but neither is a street address a crew can be sent to.
- **The probe set was partly used to design the v2 preference data.** No probe text was reused, but the phone/unit pairs were designed after seeing probe failures, so the probe isn't fully unseen for those cases. v2 didn't fix them, so no reported result is inflated by this.
- **Category set is fixed.** Complaints outside the 10 categories (or in other languages) can produce made-up labels such as `"hoyo"` or `"graffio"`. Downstream systems should validate `category` against the allowed list.
- **Recommendation for any real use:** validate the JSON and category, treat `address` as a suggestion a human confirms, and test on real (consented, anonymized) 311 data before deployment.

## Repo layout

```
data/       SFT splits, probe set, preference pairs (all small JSONL)
scripts/    everything above
adapters/   sft_v1 (recommended), dpo_v1 (failed run), dpo_v2
logs/       memory_time_log.csv, eval outputs, comparison.md, results_summary.md, DPO training output
```
