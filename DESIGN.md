# RunoffLab design world

Pinned by the user's master brief. Mode: Operate. An environmental analysis instrument used at a desk or on a phone in daylight. Bone #F4F1EA, ink #100F0D, terra #DA532C for actions; restrained blue/cyan reserved for computed water. Thin rules; square controls; no card grid, ornamental particles, or decorative terrain.

Archivo provides the clear grotesk voice and a broad, heavy wordmark. JetBrains Mono marks measurements, units and map labels. Self-host fonts. Controls and axes remain legible at small sizes. The experiment occupies most of the desktop viewport: a compact left rail, large terrain/map, and narrow accounting column. Mobile prioritizes a flat map above controls and results. Every panel can collapse.

Signature interaction: compute a baseline, place legal rain gardens, then scrub both physical histories at the same simulated second. Water geometry and color derive from solver output. Playback steps through saved states only. Results retain their actual run configuration. A fixed legend and text alternative explain depth.

Terrain is a measured DEM with optional 3× labeled vertical exaggeration. Camera initialized synchronously. 2D remains available without WebGL. Keyboard candidate buttons and named targets avoid canvas-only operation. Primary actions have visible focus, progress/cancel, unavailable/error/stale states. Reduced motion disables auto playback; no audio.

## As built (Oct 17)

**Domain and site.** Both views draw the whole 384 m computed domain (96 × 96 cells at 4 m) over the matching domain aerial. The 256 m study site is an ink outline on a bone halo with a "256 M STUDY SITE" chip; everything outside it sits under a 42% bone veil and is labeled "COMPUTED BUFFER", so the buffer reads as simulated but secondary. The flat map, the 3D terrain texture and the legend share one painter (`src/components/mapLayers.tsx`), so the two views cannot drift apart. The flat canvas renders at its displayed size × devicePixelRatio for crisp lines and labels; the 3D view is a smooth corner heightfield draped with a 1536 px texture of the same layers, plus one water quad per cell whose buffers are allocated once and rewritten in place.

**Water and masks.** Computed depth uses a cyan-to-deep-blue scale (square-root spacing, dry below 2 mm, faint for thin films, opaque from about 5 cm). The lake (hydro-flattened sink cells) is hatched navy, never on the depth scale, with a "LAKE · OPEN SINK" chip. Building footprints are hatched ink and called masked. Gardens are colour only — the ground is never lowered. Solid terra marks gardens that are part of the run shown; light terra marks gardens that are selected but not in that run (the baseline map). The intervention map title states how much water the gardens hold at the displayed time against their total storage.

**Candidates.** About 25 lattice gardens (G1…G25, north to south) are dashed bone squares on the map and a five-column grid of square mono toggle buttons (`aria-pressed`, focus and hover highlight the matching square). IDs print inside squares when a square is at least ~22 CSS px. The list is recomputed when the target changes; gardens that are no longer candidates are dropped with a visible notice. Custom targets are validated before they apply; a building, lake or garden overlap leaves the target unchanged and says why.

**Results.** The primary metric is "Peak ponded water in target" (with "above X cm" when a count threshold is set), baseline large in mono, change in terra with a percentage only when the baseline is at least 0.05 m³. A three-column table gives peak target depth, whole-domain peak water and water held in gardens at the end. The accounting table, at the displayed time, has rows Rain input, Initial storage, Surface water, Soil, Held in gardens, Boundary outflow, Into lake and Residual, and its note states the balance identity.

**Planning.** Greedy runs in the worker, reports designs evaluated against its upper bound with a time estimate, can be canceled, and replays the chosen design once so it can be scrubbed; its method label names the simulation-call count. Model proposals are validated before anything runs, and an invalid proposal is reported, never replaced.

Reviewed with Playwright screenshots at 1440 and 390 px (`.impeccable/review/`). Impeccable's direction roll ran with key 46519bd2; the explicit user-pinned instrument world takes precedence.
