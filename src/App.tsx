import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  SOLVERS, cells, defaultConfig, patchArea, validateConfig, validatePatches, validateRegion,
  type Config, type Frame, type Grid, type Patch, type Region, type Run, type SolverId,
} from './core/types';
import { GARDEN, candidates, preparedSite, targets } from './core/site';
import { importScenario, scenario, summary } from './core/scenario';
import { parseProposal, plannerUnsupported } from './core/planner';
import { checkSavedInput, hybridOutcome, hybridRows, sameScenarioConfig, type SavedHybridEvidence, type SavedModelProposal } from './core/savedHybrid';
import type { Reply, Request } from './workers/simulation.worker';
import MapView from './components/MapView';
import Chart from './components/Chart';
import CandidateGrid from './components/CandidateGrid';
import { AccountingTable, ResultsPanel } from './components/ResultsPanel';
import { DEPTH_STOPS, depthRGB } from './components/mapLayers';
import { fmt, modelMethodLabel, SOLVER_NAMES, solverLabel } from './components/format';

type Busy = 'baseline' | 'comparison' | 'greedy' | 'saved-hybrid' | null;
/** Static public build (npm run build:public): there is no local model server behind it, so live requests are off. */
const PUBLIC_BUILD = import.meta.env.VITE_PUBLIC_BUILD === '1';
type Search = { evaluated: number; upperBound: number; elapsedMs: number };

const STORAGE_KEY = 'runofflab-v2';
const NO_GARDENS: Patch[] = [];
const CUSTOM_TARGET_CELLS = 8;
/** Simulated seconds per real second during playback. */
const PLAYBACK_SPEED = 90;

const configKey = (c: Config) => JSON.stringify(c);
/** A trained model's recorded answer for one exact scenario (public/data/saved-proposals.json). */
type SavedProposal = SavedModelProposal;
const designKey = (p: Patch[]) => p.map(q => `${q.id}@${q.x},${q.y},${q.width},${q.height}`).sort().join('|');
const regionKey = (r: Region) => `${r.x},${r.y},${r.width},${r.height}`;
const sameGeometry = (a: Region, b: Region) => a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;
const overlaps = (a: Region, b: Region) =>
  a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height;
const ids = (p: Patch[]) => p.map(q => q.id).join(', ');
const minutes = (ms: number) => (ms < 90_000 ? `${Math.max(1, Math.round(ms / 1000))} s` : `${fmt(ms / 60_000, 1)} min`);

/** Candidate lattices depend only on the target; compute each once. */
const candidateCache = new Map<string, Patch[]>();
function candidatesFor(grid: Grid, roi: Region): Patch[] {
  const key = `${grid.id}:${regionKey(roi)}`;
  let list = candidateCache.get(key);
  if (!list) { list = candidates(grid, roi); candidateCache.set(key, list); }
  return list;
}

/** The saved physical state at or before simulated time t. No interpolated hydrological claims. */
function atTime(run: Run | undefined, t: number): Frame | undefined {
  if (!run) return undefined;
  for (let i = run.frames.length - 1; i >= 0; i--) if (run.frames[i].t <= t) return run.frames[i];
  return run.frames[0];
}

function download(name: string, value: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Plain-language reason a region cannot be the target. */
function regionProblem(grid: Grid, region: Region): string | null {
  try {
    validateRegion(grid, region);
    return null;
  } catch (error) {
    const message = error instanceof Error ? error.message : '';
    if (/building|nodata/i.test(message)) return 'it would include a masked building footprint';
    if (/water-body/i.test(message)) return 'it would include lake cells, which are an open sink';
    if (/outside/i.test(message)) return 'it would extend past the computed domain';
    return message || 'the region is invalid';
  }
}

export default function App() {
  const grid = useMemo(preparedSite, []);
  const [config, setConfig] = useState<Config>(() => ({ ...defaultConfig(), roi: targets[0] }));
  const latestConfig = useRef(config);
  useEffect(() => { latestConfig.current = config; }, [config]);
  const available = useMemo(() => candidatesFor(grid, config.roi), [grid, config.roi]);

  const [patches, setPatches] = useState<Patch[]>([]);
  const [history, setHistory] = useState<Patch[][]>([]);
  const [baseline, setBaseline] = useState<Run>();
  const [comparison, setComparison] = useState<Run>();
  const [busy, setBusy] = useState<Busy>(null);
  const [progress, setProgress] = useState(0);
  const [live, setLive] = useState<Frame>();
  const [search, setSearch] = useState<Search>();
  const [replayPatches, setReplayPatches] = useState<Patch[]>();
  const [status, setStatus] = useState('Ready. Set a storm and run the baseline.');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [mode, setMode] = useState<'2d' | '3d'>(() => (typeof window !== 'undefined' && window.innerWidth < 800 ? '2d' : '3d'));
  const [mapAction, setMapAction] = useState<'garden' | 'target'>('garden');
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [resetView, setResetView] = useState(0);
  const [providerPending, setProviderPending] = useState(false);
  const [showCompare, setShowCompare] = useState(true);
  const [highlight, setHighlight] = useState<string>();
  const [saved, setSaved] = useState<SavedProposal[]>([]);
  /** Evidence belongs to the completed comparison, so failed/canceled jobs retain the previous completed result. */
  const [savedHybrid, setSavedHybrid] = useState<SavedHybridEvidence>();

  const worker = useRef<Worker | null>(null);
  const sequence = useRef(0);
  const providerAbort = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const reducedMotion = useRef(typeof window !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches);

  useEffect(() => () => { worker.current?.terminate(); providerAbort.current?.abort(); }, []);
  useEffect(() => {
    // Optional: recorded trained-model answers, so a static deployment can show them (labelled as saved, not live).
    fetch('./data/saved-proposals.json').then(r => r.ok ? r.json() : null)
      .then(v => { if (v && Array.isArray(v.entries)) setSaved(v.entries); }).catch(() => {});
  }, []);

  const observationSeconds = (baseline ? baseline.config.rainMinutes + baseline.config.drainMinutes : config.rainMinutes + config.drainMinutes) * 60;

  useEffect(() => {
    if (!playing) return;
    let previous = performance.now();
    const timer = setInterval(() => {
      const now = performance.now(), elapsed = (now - previous) / 1000;
      previous = now;
      setTime(t => {
        const next = t + elapsed * PLAYBACK_SPEED;
        if (next >= observationSeconds) { setPlaying(false); return observationSeconds; }
        return next;
      });
    }, 80);
    return () => clearInterval(timer);
  }, [playing, observationSeconds]);

  /* ------------------------------------------------------------ derived state */

  const locked = !!busy || providerPending;
  const staleBase = !!baseline && configKey(baseline.config) !== configKey(config);
  const staleCompare = !!comparison && (configKey(comparison.config) !== configKey(config) || designKey(comparison.patches) !== designKey(patches));
  const completeCompare = !!comparison && !staleBase && !staleCompare;
  const baseFrame = atTime(baseline, time), compareFrame = atTime(comparison, time);
  const replaying = busy === 'comparison' || (busy === 'greedy' && !!replayPatches);
  const baseDepth = busy === 'baseline' ? live?.h : baseFrame?.h;
  const compareDepth = replaying ? live?.h : compareFrame?.h;
  const interventionPatches = busy === 'comparison' ? patches : replaying ? replayPatches! : comparison?.patches ?? NO_GARDENS;
  const interventionFrame = replaying ? live : compareFrame;
  const showIntervention = showCompare && (!!comparison || replaying);
  const area = patchArea(grid, patches);
  const searchBound = 1 + available.length * config.maxPatches;
  // Search runs keep no saved states, so they run about 30% faster than the streamed baseline (measured).
  const perRunMs = baseline && !staleBase ? baseline.wallMs * 0.7 : 1400;
  const targetIndex = targets.findIndex(t => t.name === config.roi.name && sameGeometry(t, config.roi));
  const gardenCapacity = interventionPatches.reduce((sum, p) => sum + p.pondingDepthM * cells(grid, p).length * grid.dx * grid.dy, 0);
  const gardenSize = GARDEN.sizeM;
  const gardenStorage = GARDEN.pondingDepthM * gardenSize * gardenSize;

  /* ------------------------------------------------------------ editing */

  const update = <K extends keyof Config>(key: K, value: Config[K]) => {
    setPlaying(false);
    setConfig(c => ({ ...c, [key]: value }));
  };

  const record = (next: Patch[]) => {
    validatePatches(grid, config, next);
    setHistory(h => [...h, patches]);
    setPatches(next);
    setError('');
  };

  const toggle = (p: Patch) => {
    const selected = patches.some(q => q.id === p.id);
    try {
      record(selected ? patches.filter(q => q.id !== p.id) : [...patches, p]);
      setNotice('');
    } catch (e) {
      setError(`Can't add ${p.id}: ${(e as Error).message}.`);
    }
  };

  /** Applies a validated target, recomputes candidates and keeps only gardens that are still candidates. */
  const applyTarget = (roi: Region) => {
    const next = candidatesFor(grid, roi);
    const kept: Patch[] = [], dropped: Patch[] = [];
    let renamed = false;
    for (const p of patches) {
      const match = next.find(q => sameGeometry(q, p));
      if (match) { kept.push(match); renamed ||= match.id !== p.id; } else dropped.push(p);
    }
    setPlaying(false);
    setConfig(c => ({ ...c, roi }));
    setPatches(kept);
    setHistory([]);
    setError('');
    const parts = [`Target set to ${roi.name}. ${next.length} candidates for this target.`];
    if (dropped.length) parts.push(`Removed ${ids(dropped)}: no longer ${dropped.length === 1 ? 'a candidate' : 'candidates'} for this target.`);
    if (renamed) parts.push(`Kept gardens were renumbered: ${ids(kept)}.`);
    setNotice(parts.join(' '));
  };

  const chooseCustomTarget = (x: number, y: number) => {
    const half = CUSTOM_TARGET_CELLS / 2;
    const region: Region = {
      x: Math.min(grid.nx - CUSTOM_TARGET_CELLS, Math.max(0, x - half)),
      y: Math.min(grid.ny - CUSTOM_TARGET_CELLS, Math.max(0, y - half)),
      width: CUSTOM_TARGET_CELLS, height: CUSTOM_TARGET_CELLS, name: 'Custom target',
    };
    const problem = regionProblem(grid, region);
    const covered = patches.filter(p => overlaps(p, region));
    if (problem || covered.length) {
      const reason = problem ?? `it would cover selected garden${covered.length > 1 ? 's' : ''} ${ids(covered)}; remove ${covered.length > 1 ? 'them' : 'it'} first`;
      setNotice('');
      setError(`Target not moved: ${reason}. The target is still ${config.roi.name}.`);
      return;
    }
    applyTarget(region);
  };

  const onMapCell = (x: number, y: number) => {
    if (locked) return;
    if (mapAction === 'target') { chooseCustomTarget(x, y); return; }
    const hit = available.find(q => x >= q.x && x < q.x + q.width && y >= q.y && y < q.y + q.height);
    if (hit) toggle(hit);
    else setError('No candidate there. Click a dashed lawn square; other cells are not eligible for a garden.');
  };

  /* ------------------------------------------------------------ computation */

  const cancel = () => {
    worker.current?.terminate();
    worker.current = null;
    sequence.current++;
    setBusy(null); setLive(undefined); setSearch(undefined); setReplayPatches(undefined); setProgress(0);
    setStatus('Canceled. Completed results are kept.');
  };

  const start = (kind: Exclude<Busy, null>, override?: Patch[], method = `Manual design`, savedProposal?: SavedModelProposal) => {
    if (busy) return;
    const design = kind === 'baseline' ? [] : override ?? patches;
    try {
      validateConfig(grid, config);
      validatePatches(grid, config, design);
      if (kind === 'saved-hybrid') {
        if (!savedProposal || !sameScenarioConfig(savedProposal.config, config)) throw Error('A saved proposal for this exact scenario is required');
        const unsupported = plannerUnsupported(config);
        if (unsupported) throw Error(unsupported);
      }
    } catch (e) {
      setError(`Cannot run: ${(e as Error).message}.`);
      return;
    }
    setError(''); setNotice(''); setPlaying(false); setProgress(0); setLive(undefined); setSearch(undefined); setReplayPatches(undefined);
    setBusy(kind);
    const id = ++sequence.current;
    const w = new Worker(new URL('./workers/simulation.worker.ts', import.meta.url), { type: 'module' });
    worker.current = w;
    setStatus(kind === 'greedy'
      ? `Greedy search: testing gardens with the simulator (up to ${searchBound} runs)…`
      : kind === 'saved-hybrid' ? 'Checking saved model and downhill designs · 2 live simulations…'
      : 'Simulating saved water states…');

    const finish = () => { setBusy(null); setLive(undefined); setSearch(undefined); setReplayPatches(undefined); w.terminate(); worker.current = null; };
    const fail = (message: string) => {
      setError(message);
      setStatus('Run failed. Previous completed results are kept.');
      finish();
    };
    w.onerror = () => fail('The simulation worker failed to load or compute. Reload the page and try again.');
    w.onmessage = (event: MessageEvent<Reply>) => {
      const m = event.data;
      if (m.id !== sequence.current) return;
      switch (m.type) {
        case 'frame':
          setProgress(m.fraction); setLive(m.frame); setTime(m.frame.t);
          return;
        case 'hybrid-progress':
          setProgress(m.fraction);
          setStatus(m.component === 'saved-model'
            ? 'Checking saved model design · live simulation 1 of 2. Model output is recorded, not live inference. Previous completed results stay visible until selection.'
            : 'Checking downhill design · live simulation 2 of 2. Previous completed results stay visible until selection.');
          return;
        case 'search': {
          setSearch(m);
          setProgress(m.evaluated / m.upperBound);
          const left = m.elapsedMs / m.evaluated * (m.upperBound - m.evaluated);
          setStatus(`Greedy search: evaluated ${m.evaluated} of up to ${m.upperBound} designs · about ${minutes(left)} left at most.`);
          return;
        }
        case 'replay':
          setReplayPatches(m.patches); setProgress(0);
          setStatus(`Greedy chose ${ids(m.patches) || 'no garden'} after ${m.simulationCalls} runs (${minutes(m.searchMs)}). Replaying it to save its history…`);
          return;
        case 'error':
          fail(m.error);
          return;
        case 'complete': {
          if (kind === 'saved-hybrid' && !m.savedHybrid) { fail('The worker did not return the saved-model comparison evidence.'); return; }
          const run = m.run;
          if (kind === 'baseline') setBaseline(run);
          else {
            setComparison(run);
            setSavedHybrid(m.savedHybrid);
            if (kind === 'greedy' || kind === 'saved-hybrid') { setHistory(h => [...h, patches]); setPatches(run.patches); }
          }
          setProgress(1);
          setTime(run.frames.at(-1)?.t ?? 0);
          setStatus(kind === 'greedy'
            ? `Complete · greedy evaluated ${m.simulationCalls} designs in ${minutes(m.searchMs ?? 0)} and chose ${ids(run.patches) || 'no garden'}; that design was replayed to show its history.`
            : kind === 'saved-hybrid' && m.savedHybrid
              ? `Complete · ${hybridOutcome(m.savedHybrid)} · 2 live simulations in ${fmt(m.savedHybrid.wallMs / 1000, 2)} s. The kept design's history was retained; no replay.`
            : `Complete · ${fmt(run.wallMs / 1000, 2)} s computation · 1 simulation.`);
          finish();
        }
      }
    };
    const request: Request = {
      id, grid, config: structuredClone(config), patches: design, candidates: available,
      mode: kind === 'greedy' ? 'greedy' : kind === 'saved-hybrid' ? 'saved-hybrid' : 'run',
      method: kind === 'baseline' ? 'No intervention' : method,
      savedProposal: kind === 'saved-hybrid' ? structuredClone(savedProposal) : undefined,
    };
    w.postMessage(request);
  };

  const requestModel = async () => {
    setProviderPending(true); setError(''); setNotice('');
    setStatus('Requesting a model proposal…');
    const abort = new AbortController();
    providerAbort.current = abort;
    const configAtRequest = structuredClone(config), candidatesAtRequest = available;
    try {
      const response = await fetch('/api/planner', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: configAtRequest, patches: [] }),
        signal: AbortSignal.any([abort.signal, AbortSignal.timeout(25_000)]),
      });
      if (!response.headers.get('content-type')?.includes('application/json')) {
        throw Error('Provider unavailable. This build has no running model endpoint. Greedy planning remains available.');
      }
      const value = await response.json();
      if (!response.ok) throw Error(typeof value?.error === 'string' ? value.error : 'Provider unavailable.');
      let accepted: Patch[];
      try {
        accepted = parseProposal(value.proposal, grid, configAtRequest, candidatesAtRequest);
      } catch (e) {
        throw Error(`Model proposal rejected: ${(e as Error).message}.`);
      }
      if (configKey(configAtRequest) !== configKey(latestConfig.current)) throw Error('The scenario changed during the model request; request again.');
      record(accepted);
      setProviderPending(false);
      const repaired = typeof value.requests === 'number' && value.requests > 1 ? ' · accepted after one repair request' : '';
      start('comparison', accepted, modelMethodLabel(value.method, value.model) + repaired);
    } catch (e) {
      const name = e instanceof Error ? e.name : '';
      setError(name === 'TimeoutError'
        ? 'Provider unavailable: the model did not answer within 25 s. Greedy planning remains available.'
        : e instanceof Error && name !== 'AbortError' ? e.message : 'Provider unavailable.');
      setStatus('Model proposal was not applied. No other method was substituted under its name.');
    } finally {
      setProviderPending(false);
      providerAbort.current = null;
    }
  };

  const savedMatch = useMemo(() => saved.find(e => sameScenarioConfig(e.config, config)), [saved, config]);
  const modelScope = plannerUnsupported(config);
  const useSavedProposal = () => {
    if (!savedMatch) return;
    try {
      checkSavedInput(grid, config, available, savedMatch);
      const accepted = parseProposal(savedMatch.proposal, grid, config, available);
      record(accepted);
      const when = new Date(savedMatch.recordedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      start('comparison', accepted, `Saved output (not live) · ${modelMethodLabel(savedMatch.method, savedMatch.model)} · recorded ${when}`);
    } catch (e) { setError(`Saved proposal rejected: ${(e as Error).message}.`); }
  };

  const compareSavedAndDownhill = () => {
    if (!savedMatch) return;
    try {
      // Reject invalid or mismatched recorded output before any worker work; the worker independently checks it again.
      checkSavedInput(grid, config, available, savedMatch);
      const accepted = parseProposal(savedMatch.proposal, grid, config, available);
      start('saved-hybrid', accepted, 'Saved model answer + downhill rule', savedMatch);
    } catch (e) { setError(`Saved proposal rejected: ${(e as Error).message}.`); }
  };

  /* ------------------------------------------------------------ save & restore */

  const exportScenario = () => {
    try {
      download('runofflab-scenario-v2.json', scenario(grid, config, patches));
    } catch (e) { setError(`Export failed: ${(e as Error).message}.`); }
  };

  const saveOnDevice = () => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(scenario(grid, config, patches)));
      setStatus('Scenario saved on this device.');
    } catch (e) { setError(`Save failed: ${(e as Error).message}.`); }
  };

  const load = (text: string) => {
    try {
      const s = importScenario(text, grid);
      setConfig(s.config); setPatches(s.patches); setHistory([]); setPlaying(false); setError(''); setNotice('');
      setStatus('Scenario restored. Run it to reproduce the computed result.');
    } catch (e) { setError(`Import failed: ${(e as Error).message}.`); }
  };

  const restoreSaved = () => {
    let text: string | null = null;
    try { text = localStorage.getItem(STORAGE_KEY); } catch { /* storage blocked */ }
    if (text) load(text); else setError('No saved scenario on this device.');
  };

  const exportResults = () => {
    try {
      download('runofflab-results-v2.json', {
        version: 2,
        scenario: scenario(grid, config, patches),
        stale: { baseline: staleBase, comparison: staleCompare },
        baseline: baseline ? summary(baseline) : null,
        comparison: comparison ? { ...summary(comparison), savedModelAndDownhill: savedHybrid ?? null } : null,
        tolerance: { absoluteM3: 1e-6, relative: 1e-3 },
      });
    } catch (e) { setError(`Export failed: ${(e as Error).message}.`); }
  };

  const onHighlight = useCallback((id?: string) => setHighlight(id), []);

  /* ------------------------------------------------------------ view */

  const numeric = (id: string, label: string, key: keyof Config, min: number, max: number, step = 1, scale = 1) => {
    const value = config[key] as number;
    return (
      <label className="number-field" htmlFor={id}>
        <span>{label}</span>
        <input
          id={id} type="number" min={min} max={max} step={step}
          value={Number.isFinite(value) ? Math.round(value * scale * 1e6) / 1e6 : ''}
          disabled={locked}
          onChange={e => update(key, (e.target.valueAsNumber / scale) as never)}
        />
      </label>
    );
  };

  const gardenTitle = (() => {
    if (staleCompare) return 'Stale design';
    if (busy === 'greedy' && !replayPatches) return 'Searching';
    if (busy === 'saved-hybrid') return 'Previous result · checking 2 designs';
    if (!interventionFrame) return comparison?.method ?? '';
    const held = interventionFrame.balance.garden;
    return `${replaying ? 'Computing · ' : ''}gardens hold ${fmt(held)} of ${fmt(gardenCapacity, 0)} m³`;
  })();

  return (
    <>
      <a className="skip-link" href="#experiment">Skip to experiment</a>
      <header className="topbar">
        <a className="wordmark" href="#">RunoffLab<span>®</span></a>
        <p>A local experiment in making room for rain.</p>
        <span className="model-label">Solver {solverLabel(config.solver)}</span>
      </header>

      <main id="experiment">
        <div className="site-heading">
          <div>
            <h1>One place.<br /><span>A different response.</span></h1>
            <p>Mercer County Park · west picnic loop, New Jersey</p>
          </div>
          <div className="site-spec">
            <p>384 m computed domain · 256 m study site · 4 m cells · 2014–2019 LiDAR composite</p>
            <a href="#assumptions">Sources &amp; assumptions</a>
          </div>
        </div>

        <div className="workspace">
          <aside className="controls" aria-label="Experiment controls">
            <Panel title="Rain scenario">
              {numeric('rain', 'Intensity · mm/hour', 'rainMmH', 0, 300)}
              {numeric('duration', 'Rain duration · min', 'rainMinutes', 1, 180)}
              {numeric('drain', 'After rain · min', 'drainMinutes', 0, 180)}
              <p className="total-rain">{fmt(config.rainMmH * config.rainMinutes / 60)} <span>mm total rain</span></p>
              <p className="hint">Constant hypothetical rain. No return period or recorded-storm claim.</p>
              <button type="button" className="primary" disabled={locked} onClick={() => start('baseline')}>
                {baseline ? 'Rerun baseline' : 'Run baseline'} <span aria-hidden="true">↗</span>
              </button>
            </Panel>

            <Panel title="Rain gardens">
              <p className="hint">
                Each garden is a {gardenSize} × {gardenSize} m bermed basin. It stores up to {GARDEN.pondingDepthM * 100} cm of
                runoff (about {fmt(gardenStorage, 0)} m³) and infiltrates {GARDEN.infiltrationMmH} mm/h into {GARDEN.soilCapacityM} m
                of soil. The ground is not reshaped. Candidates are aerial-classified lawn inside the study site, not
                evidence of suitable soil or permission to build.
              </p>
              <CandidateGrid candidates={available} selected={patches} disabled={locked} onToggle={toggle} onHighlight={onHighlight} />
              <p className="budget-readout">
                <span className="mono">{area} / {config.areaBudgetM2} m²</span>
                <span>{patches.length} of {config.maxPatches} gardens{patches.length ? ` · ${ids(patches)}` : ''}</span>
              </p>
              {numeric('budget', 'Area budget · m²', 'areaBudgetM2', 0, 2048, 16)}
              {numeric('count', 'Maximum gardens', 'maxPatches', 0, 8)}
              <div className="button-row">
                <button
                  type="button"
                  disabled={!history.length || locked}
                  onClick={() => { setPatches(history.at(-1)!); setHistory(h => h.slice(0, -1)); setError(''); }}
                >Undo</button>
                <button type="button" disabled={!patches.length || locked} onClick={() => record([])}>Remove all</button>
              </div>
              <button
                type="button"
                className="secondary"
                disabled={!baseline || staleBase || !patches.length || locked}
                onClick={() => start('comparison', undefined, `Manual design · ${ids(patches)}`)}
              >Compare same storm</button>
            </Panel>

            <Panel title="Physical assumptions" open={false}>
              {numeric('infiltration', 'Ground infiltration · mm/h', 'infiltrationMmH', 0, 100)}
              {numeric('capacity', 'Ground soil storage · m', 'soilCapacityM', 0, 1, 0.01)}
              {numeric('roughness', 'Manning roughness · s/m⅓', 'roughness', 0.015, 0.3, 0.005)}
              {numeric('threshold', 'Count water above · cm', 'thresholdM', 0, 100, 0.5, 100)}
              <label htmlFor="boundary">Domain edges
                <select id="boundary" value={config.boundary} disabled={locked} onChange={e => update('boundary', e.target.value as Config['boundary'])}>
                  <option value="open">Free-draining edges</option>
                  <option value="closed">Closed edges</option>
                </select>
              </label>
              <label htmlFor="solver">Solver
                <select id="solver" value={config.solver} disabled={locked} onChange={e => update('solver', e.target.value as SolverId)}>
                  {SOLVERS.map(s => <option key={s} value={s}>{SOLVER_NAMES[s]} ({s})</option>)}
                </select>
              </label>
              <p className="hint">
                Finite soil reservoir; dry initial surface and soil. No pipes or culverts. The count threshold is for
                analysis, not a safety level.
              </p>
            </Panel>
          </aside>

          <section className="terrain-workspace" aria-label="Terrain experiment">
            <div className="map-toolbar">
              <div className="segmented" role="group" aria-label="Map view">
                <button type="button" aria-pressed={mode === '3d'} onClick={() => setMode('3d')}>3D terrain</button>
                <button type="button" aria-pressed={mode === '2d'} onClick={() => setMode('2d')}>Flat map</button>
              </div>
              {mode === '3d' && <button type="button" onClick={() => setResetView(r => r + 1)}>Reset view</button>}
              <label className="inline-check">
                <input type="checkbox" checked={showCompare} onChange={e => setShowCompare(e.target.checked)} />
                Show gardens side by side
              </label>
            </div>

            <div className={`maps ${showIntervention ? 'paired' : ''}`}>
              <div className="map-item">
                <div className="map-title">
                  <strong>Baseline</strong>
                  <span>{staleBase ? 'Stale settings' : busy === 'baseline' ? 'Computing' : baseline ? 'No gardens' : 'Ready to compute'}</span>
                </div>
                <MapView
                  grid={grid} depth={baseDepth} selected={patches} gardens="planned" candidates={available} roi={config.roi}
                  highlight={highlight} mode={mode} reset={resetView} title="Baseline" onCell={onMapCell}
                />
              </div>
              {showIntervention && (
                <div className="map-item">
                  <div className="map-title">
                    <strong>With gardens</strong>
                    <span>{gardenTitle}</span>
                  </div>
                  <MapView
                    grid={grid} depth={compareDepth} selected={interventionPatches} gardens="computed" candidates={available} roi={config.roi}
                    highlight={highlight} mode={mode} reset={resetView} title="With gardens" onCell={onMapCell}
                  />
                </div>
              )}
            </div>

            <div className="map-description">
              <span className="legend-title">Water depth · m</span>
              <div className="legend">
                {DEPTH_STOPS.map(h => (
                  <span key={h}><i style={{ background: `rgb(${depthRGB(h).join(',')})` }} />{h === 0.2 ? '≥ ' : ''}{h}</span>
                ))}
              </div>
              <div className="legend legend-keys">
                <span><i className="key-lake" />Lake · open sink</span>
                <span><i className="key-building" />Building · masked</span>
                <span><i className="key-candidate" />Candidate</span>
                <span><i className="key-garden" />Garden in run</span>
                <span><i className="key-planned" />Selected, not in run</span>
                <span><i className="key-target" />Target</span>
                <span><i className="key-buffer" />Computed buffer</span>
              </div>
            </div>

            <div className="target-row">
              <label htmlFor="target">Inspect target
                <select
                  id="target"
                  disabled={locked}
                  value={targetIndex}
                  onChange={e => applyTarget(targets[Number(e.target.value)])}
                >
                  <option value={-1} disabled>Custom target</option>
                  {targets.map((t, i) => <option key={t.name} value={i}>{t.name}</option>)}
                </select>
              </label>
              <label htmlFor="map-action">Flat-map click
                <select id="map-action" value={mapAction} onChange={e => setMapAction(e.target.value as 'garden' | 'target')}>
                  <option value="garden">Toggle garden</option>
                  <option value="target">Set a 32 m custom target</option>
                </select>
              </label>
            </div>
            {notice && <p className="notice" role="status">{notice}</p>}
            <p className="map-text">
              North is up. The ink square is the 256 m study site; the pale band around it is computed buffer,
              simulated so that run-on from outside reaches the site. The hatched lake is a fixed-level open sink:
              water that reaches it leaves the surface and is counted as “Into lake”. Two building footprints are masked
              (no rain, no flow through). Terra squares are gardens: solid where the run includes them, light where
              they are only selected. Surface classes come from the aerial; present-day drainage is unverified.
            </p>

            <div className="run-status" role="status" aria-live="polite">
              <span>{status}</span>
              {busy && (
                <>
                  <progress value={progress} max={1} aria-label={busy === 'greedy' && search && !replayPatches ? 'Designs evaluated' : 'Simulation progress'} />
                  <button type="button" onClick={cancel}>Cancel run</button>
                </>
              )}
            </div>
            {error && <p className="error" role="alert">{error}</p>}
            {(staleBase || staleCompare) && (
              <p className="stale">Inputs changed. Previous results stay visible; rerun before interpreting a comparison.</p>
            )}

            <div className="playback">
              <button
                type="button"
                disabled={!baseline || !!busy}
                onClick={() => {
                  if (reducedMotion.current) { setStatus('Reduced motion is on. Use the time slider to step through saved states.'); return; }
                  if (time >= observationSeconds) setTime(0);
                  setPlaying(v => !v);
                }}
              >{playing ? 'Pause playback' : 'Play history'}</button>
              <button type="button" disabled={!baseline || !!busy} onClick={() => { setPlaying(false); setTime(0); }}>Restart</button>
              <label htmlFor="time">Simulated time <output htmlFor="time">{fmt(time / 60, 1)} / {fmt(observationSeconds / 60, 0)} min</output></label>
              <input
                id="time" type="range" min={0} max={observationSeconds} step={1} value={Math.min(time, observationSeconds)}
                disabled={!baseline || !!busy}
                onChange={e => { setPlaying(false); setTime(Number(e.target.value)); }}
              />
            </div>
            <Chart baseline={baseline} comparison={completeCompare ? comparison : undefined} time={time} />
          </section>

          <aside className="results" aria-label="Measured results">
            <Panel title="Inside the model">
              <ResultsPanel baseline={baseline} comparison={completeCompare ? comparison : undefined} />
            </Panel>
            <Panel title="Water accounting">
              <AccountingTable baseFrame={baseFrame} compareFrame={compareFrame} comparison={completeCompare ? comparison : undefined} />
            </Panel>
            <Panel title="Find a placement">
              <p className="hint">
                Greedy adds the garden that most lowers the peak, one at a time, testing every feasible option with the
                simulator: up to {searchBound} runs here (about {minutes(perRunMs * searchBound)}). It returns no garden
                if none helps.
              </p>
              <button type="button" className="secondary" disabled={!baseline || staleBase || locked} onClick={() => start('greedy')}>
                Evaluate greedy proposal
              </button>
              {busy === 'greedy' && search && !replayPatches && (
                <p className="search-count" aria-live="off"><strong>{search.evaluated}</strong> / {search.upperBound} designs evaluated</p>
              )}
              <button type="button" disabled={PUBLIC_BUILD || !baseline || staleBase || locked || !!modelScope} onClick={requestModel}
                aria-describedby={PUBLIC_BUILD ? 'public-model-note' : undefined}>
                {providerPending ? 'Model pending…' : 'Request model proposal'}
              </button>
              {PUBLIC_BUILD && (
                <p className="hint" id="public-model-note" role="note">
                  Live model requests are off in this public version: the trained model runs on a local computer, not on this website.
                  The trained model's recorded answers are below for the recorded scenarios, and every design is still simulated live here.
                </p>
              )}
              {savedMatch && (
                <>
                  <button type="button" disabled={!baseline || staleBase || locked || !!modelScope} onClick={useSavedProposal}>
                    Use saved model proposal
                  </button>
                  <button type="button" disabled={!baseline || staleBase || locked || !!modelScope} onClick={compareSavedAndDownhill}>
                    Compare saved model + downhill
                  </button>
                  <p className="hint">The model answer is saved, not live. The comparison also takes the downhill rule's design (the gardens that most upstream ground drains through, from D8 flow paths, with no simulation), runs both designs now, and keeps the one with less peak ponded water in the target: exactly two simulations.</p>
                </>
              )}
              {modelScope && <p className="hint" role="note">{modelScope}</p>}
              <p className="method">{comparison?.method ?? 'No proposal evaluated'}</p>
              {savedHybrid && (
                <>
                  <p className="hint">
                    {staleBase || staleCompare ? 'Previous comparison · stale settings or gardens. ' : ''}
                    Saved answer (not live) from {modelMethodLabel(savedHybrid.source.method, savedHybrid.source.model)}
                    {savedHybrid.source.checkpoint ? ` · checkpoint ${savedHybrid.source.checkpoint.split('/').pop()}` : ''}, recorded {new Date(savedHybrid.source.recordedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}.
                    {' '}Both designs were simulated live under the same settings.
                  </p>
                  <table className="metrics-table" aria-label="Saved model and downhill comparison">
                    <thead><tr><th scope="col">Design</th><th scope="col">Gardens</th><th scope="col">Target peak · m³</th></tr></thead>
                    <tbody>
                      {hybridRows(savedHybrid).map(r => (
                        <tr key={r.component}><th scope="row">{r.label}</th><td>{r.gardens.join(', ') || 'none'}</td><td>{fmt(r.objectiveM3, 3)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="method">
                    Result: {hybridOutcome(savedHybrid)} · {savedHybrid.executedSimulations} live simulations.
                    {' '}Selection uses unrounded peaks; on an exact tie the saved model's design is kept. The kept design's history is shown, with no third simulation.
                  </p>
                </>
              )}
              <p className="hint">
                The model sees the same scenario and candidates and must return valid garden IDs. Its name is shown
                only after a validated response; an invalid proposal is reported and never replaced by another method.
                {savedMatch ? ' A saved proposal is a recorded answer from the trained model for exactly this scenario; the garden design is still rerun live.'
                  : saved.length && !modelScope ? ` No saved model answer for these exact settings. Saved answers exist for ${saved.length} recorded scenarios: each built-in target with the default storm, 100 mm/h for 10 min, or 35 mm/h for 45 min, and default garden limits.` : ''}
              </p>
            </Panel>
            <Panel title="Save & reproduce" open={false}>
              <button type="button" onClick={saveOnDevice} disabled={!!busy}>Save on this device</button>
              <button type="button" onClick={restoreSaved} disabled={!!busy}>Restore saved</button>
              <button type="button" onClick={exportScenario} disabled={!!busy}>Export scenario JSON</button>
              <button type="button" onClick={() => fileInput.current?.click()} disabled={!!busy}>Import scenario JSON</button>
              <input
                className="sr-only" type="file" accept=".json,application/json" ref={fileInput} tabIndex={-1} aria-hidden="true"
                onChange={async e => {
                  const file = e.target.files?.[0];
                  if (file) {
                    if (file.size > 100_000) setError('Import failed: scenario exceeds 100 kB.');
                    else load(await file.text());
                    e.target.value = '';
                  }
                }}
              />
              <button type="button" disabled={!baseline} onClick={exportResults}>Export result summary</button>
              <p className="hint">Version 2 settings, solver and gardens; reimport and rerun. Terrain identity is checked. No account or upload.</p>
            </Panel>
          </aside>
        </div>

        <section className="assumptions" id="assumptions">
          <h2>A scenario, with its assumptions in view.</h2>
          <div>
            <p>
              Measured bare-earth terrain; hypothetical rain, soil and gardens. The default solver is a 2D local-inertial
              shallow-water approximation (Bates et al. 2010); a diffusive-wave solver is kept as a reference. Both
              conserve water exactly. The model omits pipes, curbs and culverts, has no site calibration and no observed
              flood validation. A result is not a flood warning or an engineering design.
            </p>
            <p>
              Terrain: NJOGIS / NJDEP statewide 10 ft DEM, 2014–2019 source collections, 2021 catalog edition, resampled to
              4 m cells in UTM 18N, elevations in meters NAVD88, with no pit filling. The 384 m computed domain is the 256 m
              study site plus a 64 m buffer so that run-on from outside the site is simulated. The hydro-flattened lake is a
              fixed-level open sink. Building footprints traced from the 2015 aerial are masked; roof runoff is not modeled.
              Lawn and pavement come from an aerial colour classification. Open edges are zero-depth outlets at edge elevation.
            </p>
            <p>
              <a href="https://www.nj.gov/njgin/edata/elevation/" target="_blank" rel="noreferrer">NJ elevation source</a>
              {' · '}<a href="./data/park-domain-aerial.jpg" target="_blank" rel="noreferrer">Domain aerial reference</a>
              {' · '}<a href="./data/manifest.json" target="_blank" rel="noreferrer">Data manifest</a>
            </p>
          </div>
        </section>
      </main>

      <footer>
        <strong>Built by Rishik Rontala</strong>
        <span>AI-assisted implementation · contribution record in source</span>
        <span>RunoffLab / a reproducible experiment</span>
      </footer>
    </>
  );
}

function Panel({ title, open = true, children }: { title: string; open?: boolean; children: ReactNode }) {
  return (
    <details open={open}>
      <summary>{title}</summary>
      <div className="panel-body">{children}</div>
    </details>
  );
}
