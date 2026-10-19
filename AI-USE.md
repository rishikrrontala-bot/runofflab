# Contributions and AI support

This is a factual record. It separates product direction from code authorship and is updated when contributions change.

**Source-export note, October 18, 2026:** Codex prepared submission materials, packaged this source snapshot and adapted its README for independent setup. Application source files were copied without changes. Model weights, generated training rows, provider logs and credentials are omitted; see `SOURCE-EXPORT.json` and `README.md`. The contribution record below describes the original project implementation.

## Submission materials, October 18–19, 2026

At Rishik's request, Codex completed the Devpost draft, prepared and verified the public source export, captured actual public-app runs and saved simulation histories, and assembled the demo video, evidence chart and captions. OpenAI image generation created the RunoffLab logo and submission cover. ElevenLabs generated a stock, third-person synthetic voiceover; it does not impersonate Rishik. These presentation assets do not change the application, measured terrain or simulator results.

## Who did what

**Rishik Rontala** chose RunoffLab, wrote the product concept and the detailed requirements (the master brief), supplied source leads, set the budget and spending authorization, chose the visual direction, provided the provider accounts and API keys, and directed the work session by session ("continue building and train the model", "use lots of data", "use the best models"). These are substantive product and project contributions.

**AI coding assistants wrote the code and documentation:**

- **Codex (OpenAI), October 15–16, 2026:** first implementation: diffusive-wave solver, worker, interface, terrain acquisition and preparation, original planner/dataset pipeline, local MLX tooling, first evidence and documents. **October 17:** independent read-only audits of split separation, numerics, evaluation metrics, the validation and held-out reports, and the saved proposals, with its own check scripts and suggested patches in `coordination/codex/`. It also wrote the "Compare saved model + downhill" patch (`src/core/savedHybrid.ts` and its worker, app and test changes). Claude Code reviewed, applied and revised that patch on October 18: tie wording, labels, prompt signatures, and real-solver and browser tests.
- **Claude Code (Anthropic, Claude Opus 5.5), October 16–18, 2026:** investigation of solver speed, domain size, spatial refinement, the demo effect, local training memory and code defects (with sub-agents); the local inertial solver (li-1.0.0) and storage-based rain gardens; the 384 m computational domain, water-body sinks, aerial surface classification, building footprints and rule-based targets and candidates; the 23 real-terrain training families and dataset v2; the shared planner prompt; the Adaption orchestration and the v2 training runs; LoRA merging and the local MLX model server; planner evaluation on validation, held-out and the real site; the held-out collection and an independent recomputation (`scripts/recompute_heldout.py`); saved model proposals in the app; the chunked training path; tests; this record and the other documents.

No student-authored code changes have been identified in this workspace. Generated code must not be described as written by Rishik. Explaining generated code is not the same as having written it.

## Data, models and external work

- **Real terrain and imagery:** NJOGIS/NJDEP 10 ft DEM (2014–2019 LiDAR composite) and NJ 2015 orthoimagery; see `data/manifest.json` and `data/families-manifest.json`. AI did not generate any terrain presented as measured.
- **Synthetic training families:** analytic terrain in `src/core/trainingCases.ts`, explicitly labelled synthetic.
- **Training labels:** the best design found by a simulator-driven search on each case. They are numerical experiments, not observed flood outcomes, and no language model wrote them.
- **Published methods implemented:** local inertial formulation of the shallow-water equations (Bates, Horritt & Fewtrell 2010) and Manning-type diffusive wave flux; chunk-parallel gated delta rule for training memory.
- **Provider use and cost:** every Adaption and Featherless request is logged in `models/spend.jsonl`; run records are under `models/runs/`.
- **Libraries:** React, TypeScript, Vite, Three.js, Vitest, Playwright, MLX / MLX-LM, Transformers, PEFT, NumPy, Pillow, zstandard, the Adaption Python SDK; fonts from Fontsource. Each keeps its own license.

## Still to confirm with Rishik

Submission answers about inspiration, personal learning and any code he writes or changes himself must come from him and match what happened. No personal history, user adoption, field accuracy or trained-model advantage may be invented.
