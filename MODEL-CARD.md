# Model card: RunoffLab garden planner

Status 2026-10-17: **final for the models listed.** The model was chosen on validation only. Held-out was run once per model, following the plan written in `coordination/claude.md` before any held-out outcome was seen.

## Task
- **Input:** a text prompt (`runofflab-planner-prompt-5`) containing:
  - the storm, the ground assumptions and the garden assumptions;
  - the target's area, elevation and upstream area;
  - the limits (number of gardens, treated area);
  - a table of 6–28 candidate 16 × 16 m garden sites with simple terrain features. These come from D8 routing; no simulation output is included.
- **Output:** exactly `{"selected_patch_ids": [...]}`.
- **In the app:**
  - every answer is parsed strictly and checked against the same constraints as manual designs;
  - it is then **rerun through the water model**, so the model never reports an effect itself.
- **Supported scope:** solver li-1.0.0, all ponded water counted (threshold 0), dry starting surface. Outside this, the app and server refuse to ask the model (PROMPT-CX-001).

## Training data
- **Size:** 4,000 training rows from 25 terrain families: 15 real NJ LiDAR crops (2,700 rows) and 10 explicitly synthetic (1,300 rows). Validation has 420 rows from 7 families; held-out has 420 rows from 7 families.
- **Separation:**
  - splits are whole families, fixed before generation;
  - the Mercer showcase site is excluded;
  - Codex audited split separation independently (`coordination/codex/final-assembly-audit.md`).
- **Labels:** the best design found by a simulator search (greedy plus swap/add/remove passes) on 8 m grids. These are numerical experiments, not observed flooding.
- **Files and hashes:** `models/data/v2/{train,validation,heldout}.jsonl`. SHA-256 values are in `EVIDENCE.md` and `evidence/dataset-v2-full.json` (train `b3a72d48…`, validation `4b0e4a28…`, held-out `0ef0d30f…`).

## Models

The three completed Adaption runs use LoRA rank 16, alpha 32, on the attention and MLP projections, with completion-only loss and batch 8. They measured 0 credits. Each was merged into full weights for local MLX serving (`scripts/merge_lora.py`).

| Model | Base (license) | Training | Run record |
|---|---|---|---|
| **Adaption 4B-e6 (selected)** | Qwen/Qwen3.5-4B (Apache-2.0) | 6 epochs, learning rate 2e-4. 126 packed optimizer steps; eval loss 0.231. | `models/runs/v2-qwen35-4b-e6`, run `596d3089`; served from `models/checkpoints/merged-v2-qwen35-4b-e6` |
| Adaption 4B-b8 | Qwen/Qwen3.5-4B | 3 epochs, learning rate 1e-4. 63 steps; eval loss 0.282 → 0.237. | `models/runs/v2-qwen35-4b-b8`, run `36d41213` |
| Adaption 9B | Qwen/Qwen3.5-9B (Apache-2.0) | 3 epochs, learning rate 1e-4. 63 steps; eval loss 0.251. bf16 does not fit in 18 GB, so the merged model was converted to MLX 8-bit (group size 64). That quantized artifact is what was evaluated. | `models/runs/v2-qwen35-9b-retry`, run `680fa48f` |
| Local 0.8B | Qwen/Qwen3.5-0.8B (Apache-2.0) | MLX LoRA on the last 8 layers (rank 8). 8,000 iterations (2 epochs), learning rate 1e-4, Adam. Completion-only loss with a chunked gated-delta backward. 2.6 h on an M3 Pro, peak 3.7 GB. Validation loss 2.29 → 0.349. | `models/checkpoints/local-v2-08b/run.json` |

- **Failed runs:** all measured 0 credits.
  - Two launches at batch 16 (4B `d78d5440`, 9B `5ea9b226`) were rejected at once by the provider ("batch size is higher than 8").
  - The first 9B run at batch 8 (`5ab6f696`) and a 0.8B pilot (`085a08e3`) failed on the provider side with no error message.
- **Serving settings:** local serving is greedy decoding (temperature 0), thinking off, at most 96 new tokens. The longest answer in any evaluation was 17 tokens, so no local answer was cut off (`evidence/local-truncation-audit.json`).

## How results are measured
- **Captured improvement** = (no gardens − planner) / (no gardens − search label). It is averaged over the cases where the label improves on no gardens: 409 of 420 on validation and 415 of 420 on held-out.
- **Invalid answers** count as no gardens, so they score 0.
- **Intervals:** 95% family-cluster bootstrap with 2,000 draws.
- **Paired comparisons** use the same cases and the same family draws. "Better" or "worse" is claimed only when the paired interval excludes 0.
- **Best of 2** is a planner that runs the simulator on the model's design and on the D8 heuristic's design and keeps the better one. It costs 2 solver runs. It was added after validation and before held-out (EVAL-PLAN-002).

## Validation results (used for selection)

| Planner | Valid | Captured (95% CI) | Best of 2 |
|---|---|---|---|
| **Adaption 4B-e6** | 100% | **0.635** (0.576–0.699) | 0.713 |
| Adaption 9B, 8-bit | 100% | 0.614 (0.563–0.664) | 0.697 |
| Adaption 4B-b8 | 100% | 0.608 (0.560–0.656) | 0.693 |
| Local 0.8B | 99.8% | 0.269 (0.227–0.312) | 0.621 |
| D8 heuristic (no model) | — | 0.577 | — |
| Greedy simulator search, about 59 runs | — | 0.994 (0.992–0.997) | — |

4B-e6 had the highest validation score and was selected before any held-out run. One of its 420 validation requests timed out (an infrastructure failure, not an invalid answer), so its validation numbers cover 419 cases (408 qualifying). Infrastructure failures are now retried once at the end of a run; held-out had none.

## Held-out results (420 cases, 7 families, run once)

Full table: `evidence/heldout-results.md`, generated by `scripts/collect_heldout.ts`. Every run computed the same baselines, which the script checks.

| Planner | Valid (first try) | Captured (95% CI) | vs D8 heuristic, paired | Time per proposal |
|---|---|---|---|---|
| **Adaption 4B-e6 (selected)** | 100% (100%) | **0.618** (0.574–0.675) | +0.051 (−0.009 to +0.141), **not significant** | 3.0 s |
| **4B-e6 + heuristic, best of 2** | 100% | **0.693** (0.652–0.734) | **+0.126 (+0.079 to +0.191), better** | 3.0 s + 2 runs |
| Adaption 4B-b8 | 100% (100%) | 0.611 (0.571–0.655) | +0.045, not significant | 2.9 s |
| Adaption 9B, 8-bit | 99.8% (99.8%) | 0.584 (0.548–0.628) | +0.017, not significant | 4.9 s |
| Local 0.8B | 98.3% (97.9%) | 0.280 (0.238–0.314) | −0.287, worse | 0.7 s |
| Untuned Qwen3.5-397B-A17B, thinking off | 91.7% (84.3%) | 0.457 (0.422–0.491) | −0.109, worse | hosted |
| Untuned DeepSeek-V3.2 | 86.7% (81.0%) | 0.355 (0.321–0.386) | −0.211, worse | hosted |
| Untuned Qwen3.5-9B, thinking off | 69.3% (51.7%) | 0.345 (0.305–0.380) | −0.222, worse | hosted |
| Untuned Qwen3.5-4B, thinking off | 77.9% (72.1%) | 0.285 (0.246–0.321) | −0.282, worse | hosted |
| D8 heuristic (no model) | — | 0.567 (0.525–0.608) | — | 1 run |
| Random design | — | 0.199 (0.148–0.252) | — | 1 run |
| Greedy simulator search | — | 0.993 (0.990–0.996) | — | about 55 runs |

**What this shows**
- **Training worked.** The same Qwen3.5-4B went from 0.285 untuned to 0.618 trained, and from 78% valid answers to 100%. The trained 4B also beats the untuned 397B model (0.457), which has about 100 times as many total parameters (17B active per token).
- **The model alone is about level with the D8 heuristic, not better.** Its captured improvement is higher by 0.051, but the interval includes 0. On raw volume it leaves on average 0.096 m³ more water than the heuristic.
- **Checking both is better than either.** The model and the heuristic make different mistakes. Simulating both designs and keeping the better one beats the heuristic on all three measures: +0.126 captured improvement, 0.326 m³ less water on average, and a higher relative reduction. This is the main result.
- **Greedy search is still best,** at 0.993. It uses about 55 simulations per proposal instead of 2.
- **The selection held up.** 4B-e6 is also the best trained model on held-out. Every held-out value is within 0.03 of its validation value. The 9B and 4B-b8 swap places, but by less than their intervals.
- **Real versus synthetic families:** 0.646 on real families versus 0.581 on synthetic ones for 4B-e6.

## Real site check (out of distribution)

The showcase site is never in training. It is a 4 m grid with real lawns, buildings and a lake, which differs from the 8 m training crops. It is tested on 3 targets × 4 storms (`evidence/planner-eval/site-adaption-qwen35-4b-e6.json`).

| Planner | Captured (12 scenarios) | Notes |
|---|---|---|
| Adaption 4B-e6 | 0.906 | 12 of 12 valid on the first try. It gives the same designs as 4B-b8. |
| D8 heuristic | 0.964 | Finds the best design in 8 of 12. |
| Best of 2 | 0.964 | Derived from the same simulated runs; the heuristic's design wins or ties every time. |
| Greedy | 1.000 | 73 simulations per target, measured live. |

- **The model is below the heuristic on the showcase.** It matches the heuristic on the East and North parking targets. On the South loop it picks G8 instead of G11, which captures 0.83 instead of 1.0.
- **Speed, measured live on the default storm:** the model takes 4.2 s per target (3.4 s answer plus 0.8 s verification run). Live greedy search takes 52.2 s (73 simulations).

## Limitations and failure modes
- **Training domain:** trained on 8 m crops with random elliptical "lawns" and no buildings. The 4 m showcase with real lawns is out of distribution, and the model does worse than the heuristic there.
- **Storm sensitivity:** on the showcase, the model gives the same design for all four storms at each target.
- **Features:** the D8 features miss two-dimensional flow. This is one reason the model, like the heuristic, falls short of greedy.
- **Budget errors:** untuned models most often break the area budget. None of the trained 4B models did on validation or held-out.
- **Hardware:** live model proposals need a local MLX server on Apple silicon.
  - The static public build can only show the saved, labelled answers in `public/data/saved-proposals.json`: 9 answers from 4B-e6, each rerun live in the app and tied to its exact prompt by a signature.
  - The app's **Compare saved model + downhill** runs best of 2 on these saved answers. On the showcase, the downhill design wins on the South loop. On East and North parking, both pick the same gardens.
- **Not engineering advice:** there is no field data. A model proposal is a suggestion for an experiment, not a design recommendation.
