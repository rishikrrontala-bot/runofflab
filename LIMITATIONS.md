# Limitations

RunoffLab is a hypothetical rainfall experiment, not a flood forecast or an engineering design. No observed local flood depths, drain inventory or field infiltration measurements validate its output. All numbers below refer to retained runs in `evidence/`.

## Terrain and surfaces
- **Terrain source:** the NJ statewide bare-earth 10 ft composite (2014–2019 LiDAR), exported at 4 m. Exact per-pixel collection dates are not available from the service.
- **Not represented:** curbs, inlets, pipes, culverts and changes since collection.
- **Buildings:** two footprints were traced from the 2015 aerial and masked. They get no rain and water cannot flow through them; roof drainage is not modelled. Smaller structures, trees and vehicles are treated as ground.
- **Pavement and lawn:** classified from the colour of a leaf-off April 2015 aerial (`scripts/prepare_data.py`). This is a reference classification, not a surface survey. "Eligible lawn" is not evidence of suitable soil, utilities clearance or permission to build.
- **Lake:** hydro-flattened lake cells are an open sink at fixed level. Lake level change and backwater are not modelled.

## Domain and boundaries
- **Domain size:** the computed domain is 384 m: the 256 m study site plus a 64 m buffer. Domain tests showed the south loop target receives runoff from up to 32 m beyond the west edge of the 256 m crop, and that a 320 m domain already matched the 512 m result to 0.003% at 8 m (`evidence/investigation-2026-10-17/domain`). On the final 4 m configuration, the shipped 384 m domain is within 0.0005% of a 512 m domain (`evidence/numerical-results.json`, `domain`).
- **Edges:** open edges drain one way to a zero-depth ghost cell at the edge elevation. Nothing flows in through an edge.

## Physics
- **Not full shallow-water:** li-1.0.0 is the local inertial approximation. It omits convective acceleration. Friction is Manning's, with one uniform roughness value.
- **Uniform assumptions:** infiltration and soil storage are uniform user assumptions with a finite soil reservoir. There is no evaporation, deep percolation or soil-water return.
- **Rainfall:** constant intensity. No design-storm hyetograph or return period is claimed.
- **Independent check:** the dw-1.0.0 reference reproduces li to about 1e-6 (relative) on the 256 m site when its head regularization is tightened to 1 mm. At its default 1 cm it differs by 0.7%.
- **Grid resolution:** absolute ponded volume changes 6.4% from 4 m to 2 m, while the garden effect changes 2.4% (`evidence/numerical-results.json`, `spatial`). These are observed grid/terrain-sampling sensitivities for this scenario, not estimates or bounds on absolute predictive error. An earlier threshold-based objective (water above 2 cm) changed 15%, which is why the objective is now total ponded volume.

## Rain gardens
- **Representation:** each garden is a bermed basin that stores up to 15 cm of runoff that reaches it, then infiltrates it at 25 mm/h into 0.15 m of soil. Excavation, berms and outlets are not modelled. On this site's slopes (median about 5% across a garden), a level basin would need about 1.2 m of cut, so the storage representation is an idealization.
- **Size of effect:** the effect depends on how much runoff actually crosses lawn upstream of a target. On the default storm, with up to three gardens, the modelled peak reduction ranges from about 4% (north parking 3.8%, south loop 4.4%) to about 13% (east parking row, 12.8%). Across the other site storms, the best designs found range from 3.6% to 37% (east parking, 100 mm/h short burst). This does not show that runoff is permanently removed or that flooding would be avoided.

## Planner and training data
- **Labels:** each label is the best design found by a bounded simulator search (greedy plus swap/add/remove passes), not a proven optimum. The search beat plain greedy in 403 of 4,840 final cases (8.3%).
- **Training terrain:** the dataset has 23 real NJ LiDAR crops plus 16 explicitly synthetic terrains, run on 8 m grids. The model trained on 15 real and 10 synthetic. Validation and held-out each use 4 real and 3 synthetic. Buildings are not represented in them, and the eligible areas are random ellipses rather than mapped lawns.
- **Showcase excluded:** the showcase site (4 m, real lawn and buildings) is never in training, so performance on it is an out-of-distribution check, not a held-out family.
- **Features:** the inputs a model sees use strictly downhill D8 routing, which misses flow the 2D model finds.
- **What a model proposal is:** a suggestion that is validated and then rerun. The app never reports a model's claimed effect, only the simulated one. Comparisons are in `evidence/planner-eval/`, `evidence/heldout-results.md` and `MODEL-CARD.md`.
- **How good the trained model is:**
  - On held-out families, the trained 4B alone is about level with the simple D8 heuristic. It scores higher (0.618 vs 0.567), but the paired interval includes 0, and it leaves slightly more water on average.
  - Only "best of 2" (simulate both the model's design and the heuristic's, keep the better) is shown to beat the heuristic.
  - Greedy simulator search is better than both (0.993), at about 55 simulations per proposal.
- **On the showcase site** (out of distribution), the trained model is below the heuristic: 0.906 vs 0.964 captured. It gives the same design for every storm at a given target.
- **Small held-out set:** 7 held-out families. The intervals resample whole families, so they are wide, and a different set of families could change the ranking between close methods.

## Serving and providers
- **Where the trained model runs:** on this Mac, with MLX, through local servers (`scripts/model_server.py`, `scripts/server.ts`). A static public deployment cannot run it, so the deployed app offers greedy and stored, clearly labelled results only.
- **Provider records:** Adaption run IDs, downloads and measured credit changes are in `models/runs/` and `models/spend.jsonl`. Hosted untuned-model requests are logged with token counts and catalog-price cost estimates.
