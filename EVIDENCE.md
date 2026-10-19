# Verification evidence

Updated 2026-10-17 by Claude Code. It records observed results, with the raw files they come from. It is not a completion certificate; open items are listed at the end. Superseded October 16 results (the dw-1.0.0, 256 m-domain build) are kept in `evidence/investigation-2026-10-17/` for comparison.

**Environment:** Apple M3 Pro (12 cores, 18 GB), macOS, Node 22.23.2, Python 3.12 (`.venv`).

## Software checks

| Check | Command | Result |
|---|---|---|
| Core physics, validation, planner and scenario | `npx vitest run tests/core.test.ts` | 113 passed, 2 todo |
| Training-data construction and dataset consistency | `npx vitest run tests/training.test.ts` | 9 passed |
| Evaluation metrics (scoring, bootstrap, pairing, coverage, timing, truncation) | `npx vitest run tests/evaluation.test.ts` | 10 passed |
| Saved model + downhill comparison (worker flow, rejections, and the real solver on the shipped saved answers) | `npx vitest run tests/saved-hybrid.test.ts tests/saved-hybrid-real.test.ts` | 18 passed |
| Types | `npx tsc --noEmit` | 0 errors |
| Browser journeys (real worker, Chrome) | `npx playwright test` | 6 passed, 2026-10-18 04:15 EDT |

The browser journeys cover: baseline, garden toggle by keyboard, same-storm comparison, scrubbing, export and reimport, cancel, stale inputs, an invalid custom target on the building and on the lake, the provider-unavailable message, rejection of a model proposal that names an unknown garden (G99) followed by a valid one (G7) that runs under the model's name, phone width with reduced motion, and greedy search with cancel, and the saved model + downhill comparison (two live runs, the kept design labelled, export of both components).

## Physics on the shipped configuration

Default storm: 60 mm/h for 20 min, then 10 min dry. Computed domain: 384 m at 4 m cells (9,216 cells). Solver li-1.0.0. Raw results: `evidence/numerical-results.json`, from `npm run evidence`.

**Conservation.** Across all 24 runs in the file, the largest relative water-balance residual is **9.1e-11** (tolerance 1e-3). The balance is: rain + initial storage = surface + soil + garden storage + edge outflow + lake + residual.

**Independent formulations.** Results for the south loop target:

| | li-1.0.0 (local inertial) | dw-1.0.0 (diffusive wave) | Difference |
|---|---|---|---|
| Baseline peak ponded volume | 23.511 m³ | 23.561 m³ | 0.21% |
| Effect of the 3-garden design | −1.042 m³ | −1.046 m³ | 0.4% |
| Steps / wall time | 1,145 / 0.67 s | 71,710 / 67.1 s | li is about 100× faster |

On the earlier 256 m site, tightening dw's head regularization to 1 mm reproduced li to about 1e-6 relative (`evidence/investigation-2026-10-17/inertial/dw-eps.json`).

**Formulation note (NUM-CX-001):** li-1.0.0 is the Bates, Horritt & Fewtrell (2010) local inertial update with a fully implicit friction term: |q_new| sits in the friction denominator and is solved exactly as a quadratic. This differs from their literal semi-implicit equation, which uses |q_old|. The semi-implicit variant changed the 256 m site objective by about 1% (prototype evidence above). Codex's independent small-grid review found no arithmetic or accounting defect (`coordination/codex/numerical-review.md`).

**Timestep refinement.** Halving the stability coefficient (α 0.7 → 0.35) changes the baseline objective by 1.3e-5 relative, and the garden effect by 7.8e-6 (target: under 5%).

**Domain size.** Peak ponded volume at the south loop, for the same design, at each computed domain size:

| Domain | Baseline | Garden effect | Baseline change vs 512 m |
|---|---|---|---|
| 320 m | 23.496 m³ | 1.0420 m³ | 0.066% |
| 384 m (shipped) | 23.511 m³ | 1.0418 m³ | 0.0005% |
| 448 m | 23.511 m³ | 1.0418 m³ | 0.0003% |
| 512 m | 23.511 m³ | 1.0418 m³ | — |

Before the domain was enlarged, the 256 m crop under-counted this target by about 18% at 8 m (6.47 against 7.93 m³; the converged value is about 22% higher than the crop), because runoff entered from up to 32 m past its west edge (`evidence/investigation-2026-10-17/domain`).

**Grid resolution.** Identical physical target and gardens (snapped to even 4 m indices). All grids sample the same nominal 10 ft (about 3.05 m) DEM product. The 2 m grid is a separate bilinear service resampling; it adds no independently measured terrain detail. This is combined grid/terrain-sampling sensitivity:

| Cells | Baseline | Garden effect | Peak depth |
|---|---|---|---|
| 8 m | 22.249 m³ | 0.991 m³ | 5.2 cm |
| 4 m (shipped) | 22.898 m³ | 1.031 m³ | 22.3 cm |
| 2 m | 21.438 m³ | 1.055 m³ | 20.1 cm |

Going from 4 m to 2 m changes the absolute ponded volume by **6.4%**, which is above the 5% aim, and changes the garden effect by **2.4%**. For this storm and garden set, the computed garden effect is less sensitive than the absolute volume. These differences do not establish absolute predictive accuracy or ranking stability for other designs. 8 m smooths away the ponding depth (5 cm against about 20 cm), which is why training data on 8 m grids is never compared directly with the 4 m site.

**Garden effect per target.** Greedy search over 25 lawn candidates, up to 3 gardens (73 solver runs each):

| Target | Baseline peak | Best 3 gardens | Change | Held in gardens at end |
|---|---|---|---|---|
| South loop | 23.51 m³ | 22.47 m³ (G12 + G11 + G7) | −1.04 m³ (−4.4%) | 8.3 m³ |
| East parking | 22.25 m³ | 19.40 m³ (G2 + G3 + G4) | −2.85 m³ (−12.8%) | 56.9 m³ |
| North parking | 13.72 m³ | 13.20 m³ (G4 + G7 + G5) | −0.52 m³ (−3.8%) | 6.5 m³ |

The effect depends on how much runoff crosses lawn upstream of each target. Targets were chosen by a fixed rule (most baseline ponding on pavement) before any garden was simulated (`evidence/target-selection.json`).

## Planner dataset (v2)

- **Size:** 4,840 simulator-labelled cases, 0 rejected (`evidence/dataset-v2-full.json`). Splits by whole family, fixed before generation:
  - train: 4,000 rows, 25 families (15 real NJ LiDAR, 10 synthetic); SHA-256 `b3a72d485b2f079113a8693f6ba1e5f8442dd1098c0192078b371a116aa670de`;
  - validation: 420 rows, 7 families (4 real, 3 synthetic); SHA-256 `4b0e4a287c8108da8a0cbe67e755e16b6419de0736ef0aafd2d9337cf3321cef`;
  - held-out: 420 rows, 7 families (4 real, 3 synthetic); SHA-256 `0ef0d30ffdfd5b15805e2185a93171a6a206ef60239385f9cbee1bbbbd82db3a`.

  The Mercer showcase is excluded.
- **Labels:** the best design found by greedy plus swap/add/remove search, about 89 solver runs per case. The search beat plain greedy in 403 cases (8.3%). 54 labels are "no gardens".
- **Checks:** no prompt is shared across splits (SHA-256), and every planned case is present (assembly refuses otherwise). Every stored prompt equals a fresh rebuild. Codex's independent split audit found no family or terrain overlap and no label leakage (`coordination/codex/split-audit.md`).
- **Prompt version:** `runofflab-planner-prompt-5`. Columns are defined in the prompt. The routed-area feature counts each source cell once (FEAT-CX-001).

## Planner results

**How results are measured.**
- Captured improvement = (no gardens − method) / (no gardens − search label), averaged over the cases where the label improves on no gardens.
- Invalid answers count as 0.
- Intervals are a 95% family-cluster bootstrap. Paired comparisons use the same cases and the same family draws.
- Models were selected on validation only. Held-out was run once per model, under the plan predeclared in `coordination/claude.md`.

Raw per-case reports are in `evidence/planner-eval/`. The held-out table is `evidence/heldout-results.md`; model details are in `MODEL-CARD.md`.

**Untuned models, small validation probes** (used to fix settings before held-out; the settings differ, so these rows are not directly comparable with each other or with held-out):

| Model | Cases | Prompt | Token cap | Valid | Captured |
|---|---|---|---|---|---|
| DeepSeek-V3.2 | 40 | v4 | 768 | 0.93 | 0.34 |
| Qwen3.5-9B, thinking off | 40 | v4 | 256 | 0.80 | 0.34 |
| Qwen3.5-4B, thinking off | 40 | v4 | 256 | 0.88 | 0.24 |
| Qwen3.5-397B-A17B, thinking off | 39 | v5 | 256 | 0.82 | 0.35 |
| Kimi-K2.6, thinking off | 39 | v5 | 256 | 0.59 | 0.25 |

- At a 768-token cap, reasoning models with hidden thinking returned empty or truncated answers (EVAL-001). Those runs measure the harness, not the models, and are excluded.
- GLM-5.3 ignores the thinking-off flag, so it is excluded.
- Kimi-K2.6 answered with thinking off but was not carried to held-out. It had the lowest validity of the probes (0.59).
- The held-out comparators were fixed before any held-out run: DeepSeek-V3.2, and Qwen3.5-397B, 9B and 4B.

**Held-out: 420 cases from 7 families, 415 qualifying, complete coverage, 0 infrastructure failures.**

| Planner | Valid | Captured (95% CI) | vs D8 heuristic, paired | Solver runs |
|---|---|---|---|---|
| Random feasible design | — | 0.199 (0.148–0.252) | — | 1 |
| D8 heuristic (most routed upstream area) | — | 0.567 (0.525–0.608) | — | 1 |
| Greedy (simulator marginal benefit) | — | 0.993 (0.990–0.996) | — | about 55 |
| **Trained Adaption Qwen3.5-4B, 6 epochs (4B-e6, selected)** | 100% | **0.618** (0.574–0.675) | +0.051 (−0.009 to +0.141), not significant | 1 |
| **Same model + heuristic, best of 2** | 100% | **0.693** (0.652–0.734) | **+0.126 (+0.079 to +0.191), better** | 2 |
| Trained Adaption Qwen3.5-4B, 3 epochs (4B-b8) | 100% | 0.611 (0.571–0.655) | +0.045 (−0.005 to +0.111), not significant | 1 |
| Trained Adaption Qwen3.5-9B, 8-bit | 99.8% | 0.584 (0.548–0.628) | +0.017, not significant | 1 |
| Trained local Qwen3.5-0.8B | 98.3% | 0.280 (0.238–0.314) | −0.287, worse | 1 |
| Untuned Qwen3.5-397B-A17B, thinking off | 91.7% | 0.457 (0.422–0.491) | −0.109, worse | 1 |
| Untuned DeepSeek-V3.2 | 86.7% | 0.355 (0.321–0.386) | −0.211, worse | 1 |
| Untuned Qwen3.5-9B, thinking off | 69.3% | 0.345 (0.305–0.380) | −0.222, worse | 1 |
| Untuned Qwen3.5-4B, thinking off | 77.9% | 0.285 (0.246–0.321) | −0.282, worse | 1 |

- **Training effect.** On the same base model, captured improvement went from 0.285 to 0.618 and validity from 77.9% to 100%.
- **On raw volume,** the selected model alone leaves on average 0.096 m³ more water than the heuristic, so it is about level with it. Best of 2 leaves 0.326 m³ less.
- **Validation agrees** with held-out to within 0.03 for every trained model; the selected model scored 0.635 on validation.

**Real showcase site, out of distribution** (3 targets × 4 storms, `evidence/planner-eval/site-*.json`):

| Planner | Captured |
|---|---|
| Trained 4B | 0.906 (12 of 12 valid) |
| D8 heuristic | 0.964 |
| Best of 2 | 0.964 |
| Greedy | 1.000 |
| Random | 0.142 |

- **Speed, measured live on the default storm** (`defaultStormMeanEndToEndSeconds`): the model's answer plus one verification run takes 4.2 s per target (3.4 s answer, 0.8 s verification). Live greedy takes 52.2 s, with 73 executed simulations.
- **Here the model is below the heuristic.** On the South loop it picks one different lawn (G8 instead of G11).

**Local truncation (EVAL-CX-009).** Every stored local answer is at most 17 tokens, against a 96-token cap, and every one parses as JSON (`evidence/local-truncation-audit.json`). The server now reports a stop reason. Switching it to streaming generation left all 12 checked answers byte-identical (`evidence/server-stream-equivalence.json`).

## Providers

- **Adaption:**
  - Proven path: upload → AutoScientist LoRA training → download (zstd tar) → merge into full weights (`scripts/merge_lora.py`) → MLX serving.
  - 9 AutoScientist launches, all measuring 0 credits.
  - 5 succeeded: the three v2 planners, 4B-b8 (`36d41213`), 4B-e6 (`596d3089`) and 9B (`680fa48f`); and two diagnostic runs on the superseded 1,100-row dw-1 dataset (`b69c8a8b`, `e278da87`).
  - 4 failed. Two were rejected at launch for asking for batch 16 when the limit is 8 (`d78d5440`, `5ea9b226`). Two failed on the provider side with no error message (9B `5ab6f696`, 0.8B pilot `085a08e3`).
  - Credits: 2,848 still available; every run measured 0 credits.
  - Run records are in `models/runs/`.
- **Featherless:** $2.02 of $23.25 spent (22 logged entries, costed at catalog prices). Every request is logged in `models/spend.jsonl`. No payment method or top-up was added.

## Limits of the recorded evidence

- These results document the original local verification; they do not certify a particular competition submission or video.
- No field calibration exists. A modelled reduction is not evidence of avoided flooding.
