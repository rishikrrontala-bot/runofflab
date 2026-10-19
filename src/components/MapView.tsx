import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentType } from 'react';
import type { Grid, Patch, Region } from '../core/types';
import {
  paintBase, paintOverlay, paintWater, useAerial, useMapFont, waterScratch,
  VERTICAL_EXAGGERATION, type Overlay, type WaterScratch,
} from './mapLayers';
import type { TerrainProps } from './TerrainMap';

/** The 3D view is a separate chunk; every map instance shares one load. */
let terrainModule: Promise<ComponentType<TerrainProps>> | undefined;
function useTerrainMap(wanted: boolean, onFail: () => void) {
  const [component, setComponent] = useState<ComponentType<TerrainProps>>();
  useEffect(() => {
    if (!wanted || component) return;
    let alive = true;
    terrainModule ??= import('./TerrainMap').then(m => m.default);
    terrainModule.then(c => alive && setComponent(() => c), () => { terrainModule = undefined; if (alive) onFail(); });
    return () => { alive = false; };
  }, [wanted, component, onFail]);
  return component;
}

interface Props {
  grid: Grid;
  depth?: Float32Array;
  selected: Patch[];
  /** Whether the selected gardens are part of the run shown on this map. */
  gardens: Overlay['gardens'];
  candidates: Patch[];
  roi: Region;
  highlight?: string;
  mode: '2d' | '3d';
  reset: number;
  title: string;
  onCell?: (x: number, y: number) => void;
}

export default function MapView(props: Props) {
  const [webglFailed, setWebglFailed] = useState(false);
  const aerial = useAerial();
  const fontReady = useMapFont();
  const { candidates, selected, roi, highlight, gardens } = props;
  // fontReady is a dependency so labels repaint once the mono font has loaded.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const overlay = useMemo<Overlay>(() => ({ candidates, selected, roi, highlight, gardens }), [candidates, selected, roi, highlight, gardens, fontReady]);
  const flat = props.mode === '2d' || webglFailed;
  const failTo2d = useMemo(() => () => setWebglFailed(true), []);
  const TerrainMap = useTerrainMap(!flat, failTo2d);

  return (
    <div className={`map-shell ${flat ? 'is-flat' : 'is-3d'}`} data-aerial={aerial ? 'loaded' : 'pending'}>
      {flat
        ? <FlatMap grid={props.grid} aerial={aerial} depth={props.depth} overlay={overlay} title={props.title} onCell={props.onCell} />
        : TerrainMap
          ? <TerrainMap grid={props.grid} aerial={aerial} depth={props.depth} overlay={overlay} title={props.title} reset={props.reset} onFail={failTo2d} />
          : <div className="three-map three-loading">Loading 3D terrain…</div>}
      <span className="map-caption">
        {flat ? 'North up · 4 m cells' : `3D terrain · height ×${VERTICAL_EXAGGERATION} · drag to orbit`}
        {webglFailed ? ' · WebGL unavailable; flat view shown' : ''}
      </span>
    </div>
  );
}

/* ---------------------------------------------------------------- flat map */

interface FlatProps {
  grid: Grid;
  aerial?: HTMLImageElement;
  depth?: Float32Array;
  overlay: Overlay;
  title: string;
  onCell?: (x: number, y: number) => void;
}

function FlatMap({ grid, aerial, depth, overlay, title, onCell }: FlatProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [css, setCss] = useState(0);
  const layers = useRef<{ base?: HTMLCanvasElement; over?: HTMLCanvasElement; water?: WaterScratch }>({});

  // Canvas resolution follows its displayed size, so lines and labels stay crisp at any width.
  useLayoutEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const measure = () => setCss(Math.round(el.getBoundingClientRect().width));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const dpr = typeof window === 'undefined' ? 1 : Math.min(2, window.devicePixelRatio || 1);
  const size = Math.max(1, Math.round(css * dpr));

  useEffect(() => {
    if (!css) return;
    const base = layers.current.base ?? document.createElement('canvas');
    base.width = size; base.height = size;
    const ctx = base.getContext('2d');
    if (ctx) paintBase(ctx, size, grid, aerial);
    layers.current.base = base;
  }, [grid, aerial, size, css]);

  useEffect(() => {
    if (!css) return;
    const over = layers.current.over ?? document.createElement('canvas');
    over.width = size; over.height = size;
    const ctx = over.getContext('2d');
    // Garden IDs only when a 16 m square is at least ~22 CSS px wide.
    const cellCss = css / grid.nx;
    if (ctx) paintOverlay(ctx, size, grid, overlay, { unit: dpr, ids: cellCss * 4 >= 22, scaleBar: true });
    layers.current.over = over;
  }, [grid, overlay, size, css, dpr]);

  useEffect(() => {
    const el = canvas.current, { base, over } = layers.current;
    if (!el || !css || !base || !over) return;
    if (el.width !== size) { el.width = size; el.height = size; }
    const ctx = el.getContext('2d');
    if (!ctx) return;
    layers.current.water ??= waterScratch(grid);
    ctx.drawImage(base, 0, 0);
    if (layers.current.water) paintWater(ctx, size, grid, depth, layers.current.water);
    ctx.drawImage(over, 0, 0);
  }, [grid, depth, overlay, aerial, size, css]);

  const label = `${title} flat map, north up. Ink outline: 256 m study site; outside it is computed buffer. `
    + `Dashed squares are ${overlay.candidates.length} candidate gardens; terra squares are `
    + `${overlay.gardens === 'computed' ? 'the gardens in this run' : 'selected gardens, not part of this run'} `
    + `(${overlay.selected.map(p => p.id).join(', ') || 'none'}). Dashed ink rectangle: ${overlay.roi.name}. `
    + 'Hatched navy: lake (open sink). Black: masked buildings. Use the garden buttons to select by keyboard.';

  return (
    <div className="flat-wrap">
      <canvas
        ref={canvas}
        className="flat-map"
        role="img"
        aria-label={label}
        onClick={event => {
          if (!onCell) return;
          const r = event.currentTarget.getBoundingClientRect();
          const x = Math.floor((event.clientX - r.left) / r.width * grid.nx);
          const y = Math.floor((event.clientY - r.top) / r.height * grid.ny);
          if (x >= 0 && y >= 0 && x < grid.nx && y < grid.ny) onCell(x, y);
        }}
      />
    </div>
  );
}
