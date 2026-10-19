# RunoffLab

<img src="brand/runofflab-logo.png" alt="RunoffLab" width="440">

Run a hypothetical storm over public LiDAR terrain at Mercer County Park, New Jersey. Place rain gardens on lawn, repeat the same storm, and compare the computed ponding on a paved target. A water balance tracks rainfall, surface water, soil, garden storage, edge outflow and drainage into the lake.

Built by **Rishik Rontala**. Rishik chose the concept, wrote the requirements, supplied source leads, set the budget and visual direction, and directed the project. **AI coding assistants wrote the implementation and documentation.** The detailed contribution record is in [AI-USE.md](AI-USE.md).

Live app: [rishikrrontala-bot.github.io/runofflab](https://rishikrrontala-bot.github.io/runofflab/).

Source: [the repository's `source` branch](https://github.com/rishikrrontala-bot/runofflab/tree/source). The `main` branch contains the compiled public website.

This is an educational experiment with stated assumptions. It is not a flood forecast or an engineering design. No field calibration exists, and a modelled reduction is not evidence of avoided flooding or damage. See [LIMITATIONS.md](LIMITATIONS.md).

## Run locally

Use Node.js 22.12 or newer. The original project was checked with Node 22.23.2.

```sh
git clone --branch source https://github.com/rishikrrontala-bot/runofflab.git
cd runofflab
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. The simulation, manual placement, greedy search and saved-model comparison run without an account, API key or backend.

1. Choose a storm and click **Run baseline**.
2. Select lawn gardens, then click **Compare same storm**.
3. Scrub both histories at the same simulated second and inspect the water balance.
4. Try **Evaluate greedy proposal** or **Compare saved model + downhill**.
5. Export a scenario JSON and reimport it to reproduce the experiment.

The default scenario is 60 mm/h for 20 minutes followed by 10 minutes of drainage. The default budget is three 16 × 16 m gardens, totalling 768 m². Controls support keyboard interaction. The flat map is the phone default and the fallback when WebGL is unavailable.

## Verify

```sh
npm test
npm run typecheck
npm run build
```

The compact prepared terrain-family grids are included, so the training-family tests run without downloading data. Two dataset consistency checks are skipped when the optional generated training-case directory is absent. The existing core suite also contains two explicit todo cases.

Browser tests require the development server on port 5173. The supplied Playwright configuration uses Google Chrome at its standard macOS path. On another system, change `launchOptions.executablePath` in `playwright.config.ts` to the installed browser path, or remove that option and install Playwright's Chromium with `npx playwright install chromium`. Then run:

```sh
npm run test:browser
```

`npm run evidence` regenerates the numerical evidence, including slower reference-solver and resolution checks. Its prepared refinement grids are included. It overwrites `evidence/numerical-results.json`, so retain the recorded file separately if comparing runs.

## How it works

| Part | Source | Purpose |
|---|---|---|
| Water model | `src/core/solver.ts` | Local inertial shallow-water formulation, with an independently implemented diffusive-wave reference. Shared-face transfers and donor limits preserve the water balance. |
| Real site | `src/core/site.ts`, `public/data/` | A 384 m computational domain at 4 m cells: a 256 m study site plus a 64 m buffer on each side. Includes a lake sink, two masked buildings, aerial-derived surface classes, three targets and 25 candidate gardens. |
| Gardens | `src/core/solver.ts` | Bermed storage basins with up to 15 cm of runoff storage and finite infiltration. Ground elevation is not reshaped. |
| Worker and interface | `src/workers/`, `src/App.tsx`, `src/components/` | Computation independent of rendering, synchronized histories, cancellation, water accounting, 2D/3D views and scenario export. |
| Planners | `src/core/planner.ts`, `src/core/savedHybrid.ts` | Strict proposal validation, a downhill routing heuristic, greedy simulator search, and two-run comparison of a saved trained-model answer with the heuristic. |
| Data and training | `src/core/trainingCases.ts`, `scripts/` | Deterministic case construction, terrain-family splits, simulator labels, Adaption orchestration, local MLX training and model evaluation. |

Every proposed design is validated and simulated before an effect is reported. The language model selects garden IDs; it does not produce the displayed water-volume result. Greedy adds one garden at a time by testing feasible additions; it is not an exhaustive search over every combination or a proof of a global optimum.

## Trained planner and recorded evidence

Qwen3.5-4B was fine-tuned through Adaption AutoScientist on 4,000 simulator-labelled cases from 25 terrain families. Validation and held-out sets use different whole families, and the Mercer showcase was excluded from training.

On 415 qualifying cases from seven held-out families:

| Planner | Fraction of best-known improvement captured |
|---|---:|
| Greedy simulator search | 0.993 |
| Trained model and downhill rule, simulate both and keep the better | 0.693 |
| Trained model alone | 0.618 |
| Downhill rule alone | 0.567 |
| Same 4B model before training | 0.285 |

The trained model's advantage over the downhill rule alone is not statistically significant. Comparing both designs is significantly better than the rule across the held-out evaluation. On the real showcase, the model scores below the rule: 0.906 versus 0.964. These values measure captured improvement relative to bounded simulator-search labels, not field accuracy or a percentage reduction in real flooding.

Full definitions, confidence intervals and limitations are in [MODEL-CARD.md](MODEL-CARD.md), [EVIDENCE.md](EVIDENCE.md) and [the held-out results](evidence/heldout-results.md). Recorded numerical results and compact evaluation summaries are included under `evidence/`.

## Optional live model server

The live model path requires Apple silicon, MLX / MLX-LM and a compatible merged trained checkpoint. **Model weights are not included in this source export.** The original selected checkpoint was `models/checkpoints/merged-v2-qwen35-4b-e6`.

With such a checkpoint available, the original serving flow is:

```sh
npm run build
npm run serve
python3 scripts/model_server.py --model /path/to/merged-checkpoint
```

Run the app server and model server in separate terminals. They bind to loopback ports 4173 and 8787 respectively. The app server checks proposals and permits at most one repair request. An invalid model answer is reported as invalid; another planner is never substituted under the model's name.

Without weights or servers, the app can use nine recorded, labelled model answers in `public/data/saved-proposals.json`. Each is tied to the exact model prompt by a signature and is simulated live. **Compare saved model + downhill** performs exactly two water simulations and identifies which design was kept, including when both chose the same gardens. Saved answers are not live inference.

## Build a static site

```sh
npm run build:public
```

Publish the contents of `dist-public/` to a static host. Live model requests are disabled in that build; saved answers and all water simulations still work in the browser. The Vite base is relative, so the build supports a GitHub Pages project path.

`scripts/publish-site.sh` is the original helper for staging a separate repository containing only the compiled site. Without `--publish` it stages locally. With `--publish` it creates a public GitHub repository and enables Pages; it is not needed to run or build this source export.

## Data provenance and export scope

Terrain is the NJOGIS/NJDEP nominal 10 ft bare-earth DEM, a 2014–2019 LiDAR composite in the 2021 catalog edition. It was resampled to UTM 18N, and US survey feet were converted to metres NAVD88. Lawn and pavement are colour classifications of NJ 2015 orthoimagery, not surveyed ground truth. Source URLs, processing assumptions and checksums are retained in `data/manifest.json`, `data/families-manifest.json` and `public/data/manifest.json`. The original DEM item metadata, including its use limitations, is preserved in `data/source-terms/nj-dem-item.json`.

This export includes the application source, runtime data, prepared terrain-family grids, evidence refinement grids, scripts, tests and compact evidence. It omits credentials, dependency directories, model weights, generated training cases and SFT JSONL rows, API caches, provider run logs, raw download files, historical investigation artifacts and an unrelated project. The scripts preserve the training and evaluation methods; rerunning training requires regenerating the optional data and obtaining the relevant models and provider access. `scripts/prepare_data.py --fetch` can reacquire the raw site exports; `scripts/fetch_families.py` and `scripts/prepare_families.py` prepare the family grids.

The evidence and model card retain their original measurement dates. References to omitted historical artifacts describe the original working project, not files bundled here. `SOURCE-EXPORT.json` lists the copied files and omissions. No terrain presented as measured, training label or reported numerical result was invented by a language model.

## Credits

Rishik Rontala: product concept, requirements, source leads, budget, visual direction and project direction. Codex and Claude Code: implementation and documentation, as recorded in [AI-USE.md](AI-USE.md).

Methods include the local inertial shallow-water formulation of Bates, Horritt and Fewtrell (2010) and Manning-type diffusive-wave flux. Libraries and tools include React, TypeScript, Vite, Three.js, Vitest, Playwright, MLX / MLX-LM, Transformers, PEFT, NumPy, Pillow and the Adaption SDK. Fonts are Archivo and JetBrains Mono through Fontsource. Terrain and imagery are credited to NJOGIS / NJDEP and the NJ imagery partners. Each external dependency and dataset retains its own license or terms.
