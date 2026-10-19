import { useEffect, useState } from 'react';
import type { Grid, Patch, Region } from '../core/types';

/**
 * Shared map drawing for the flat map and the 3D terrain texture: aerial base, computed-buffer veil,
 * water depth, lake and building masks, outlines and labels. Everything is drawn in grid-cell units
 * scaled to the canvas, so both views show the same thing.
 */

export const AERIAL_URL = './data/park-domain-aerial.jpg';
export const VERTICAL_EXAGGERATION = 3;

const INK = '#100F0D';
const BONE = '#F4F1EA';
const TERRA_DARK = '#B83F1C';
const LAKE = 'rgba(22, 46, 70, 0.72)';

/** Depth (m) below which a cell is drawn dry. Thinner films are still counted in every result. */
export const WET_M = 0.002;
/** Legend stops (m). */
export const DEPTH_STOPS = [0.005, 0.02, 0.05, 0.1, 0.2];

/** Cyan to blue over 0–0.2 m (square-root spacing so shallow sheet flow and deep ponding both read). */
export function depthRGB(h: number): [number, number, number] {
  const a = Math.min(1, Math.sqrt(Math.max(0, h) / 0.2));
  return [Math.round(110 - 92 * a), Math.round(214 - 130 * a), Math.round(232 - 76 * a)];
}

/** Opacity: transparent when dry, faint (0.18) for a 2 mm film, opaque enough to read from 5 cm. */
export function depthAlpha(h: number): number {
  if (!(h >= WET_M)) return 0;
  return 0.18 + 0.7 * Math.pow(Math.min(1, (h - WET_M) / 0.048), 0.7);
}

/** Lookup table (RGBA bytes) in 0.5 mm steps, so per-frame drawing allocates nothing. */
const LUT_STEP_M = 0.0005;
const LUT_SIZE = 512;
export const depthLUT = (() => {
  const lut = new Uint8ClampedArray(LUT_SIZE * 4);
  for (let k = 0; k < LUT_SIZE; k++) {
    const h = k * LUT_STEP_M, [r, g, b] = depthRGB(h);
    lut.set([r, g, b, Math.round(depthAlpha(h) * 255)], k * 4);
  }
  return lut;
})();
export const lutIndex = (h: number) => (h > 0 ? Math.min(LUT_SIZE - 1, Math.round(h / LUT_STEP_M)) * 4 : 0);

/** One shared aerial image for every map instance. */
let aerialLoad: Promise<HTMLImageElement> | undefined;
export function useAerial(): HTMLImageElement | undefined {
  const [image, setImage] = useState<HTMLImageElement>();
  useEffect(() => {
    let alive = true;
    aerialLoad ??= new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Aerial failed to load'));
      img.src = AERIAL_URL;
    });
    aerialLoad.then(img => alive && setImage(img), () => undefined);
    return () => { alive = false; };
  }, []);
  return image;
}

/** True once the map label font is available; canvases repaint then (canvas text never waits for fonts). */
let fontLoad: Promise<unknown> | undefined;
export function useMapFont(): boolean {
  const [ready, setReady] = useState(false);
  useEffect(() => {
    let alive = true;
    if (typeof document === 'undefined' || !document.fonts) { setReady(true); return; }
    fontLoad ??= document.fonts.load("500 12px 'JetBrains Mono'").catch(() => undefined);
    fontLoad.then(() => alive && setReady(true));
    return () => { alive = false; };
  }, []);
  return ready;
}

export interface Overlay {
  candidates: Patch[];
  selected: Patch[];
  roi: Region;
  highlight?: string;
  /** 'computed': gardens are part of the run shown (solid terra). 'planned': selected but not in this run. */
  gardens: 'computed' | 'planned';
}

/** Drawing scale: px = canvas pixels per grid cell; unit = canvas pixels per CSS pixel (line widths, text). */
interface Scale { px: number; unit: number }

const fullSite = (g: Grid): Region => g.site ?? { x: 0, y: 0, width: g.nx, height: g.ny, name: 'Site' };

/** Aerial (or an elevation tint if the image is unavailable), with the buffer outside the study site veiled. */
export function paintBase(ctx: CanvasRenderingContext2D, size: number, g: Grid, aerial?: HTMLImageElement) {
  const px = size / g.nx;
  ctx.clearRect(0, 0, size, size);
  if (aerial) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(aerial, 0, 0, size, size);
  } else {
    let low = Infinity, high = -Infinity;
    for (const z of g.z) { low = Math.min(low, z); high = Math.max(high, z); }
    for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) {
      const v = (g.z[y * g.nx + x] - low) / (high - low || 1);
      ctx.fillStyle = `rgb(${Math.round(176 + v * 60)},${Math.round(180 + v * 52)},${Math.round(156 + v * 52)})`;
      ctx.fillRect(x * px, y * px, px + 0.5, px + 0.5);
    }
  }
  // Computed buffer: the solver runs here too, but it is outside the study site.
  const s = fullSite(g);
  ctx.fillStyle = 'rgba(244, 241, 234, 0.42)';
  ctx.beginPath();
  ctx.rect(0, 0, size, size);
  ctx.rect(s.x * px, s.y * px, s.width * px, s.height * px);
  ctx.fill('evenodd');
}

/** Reusable one-pixel-per-cell scratch canvas for water depth. */
export interface WaterScratch { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D; image: ImageData }
export function waterScratch(g: Grid): WaterScratch | undefined {
  const canvas = document.createElement('canvas');
  canvas.width = g.nx; canvas.height = g.ny;
  const ctx = canvas.getContext('2d');
  return ctx ? { canvas, ctx, image: ctx.createImageData(g.nx, g.ny) } : undefined;
}

export function paintWater(ctx: CanvasRenderingContext2D, size: number, g: Grid, depth: Float32Array | undefined, scratch: WaterScratch) {
  if (!depth || depth.length !== g.nx * g.ny) return;
  const data = scratch.image.data;
  for (let i = 0; i < depth.length; i++) {
    const k = g.mask[i] && !g.sink?.[i] ? lutIndex(depth[i]) : 0;
    data[i * 4] = depthLUT[k]; data[i * 4 + 1] = depthLUT[k + 1]; data[i * 4 + 2] = depthLUT[k + 2]; data[i * 4 + 3] = depthLUT[k + 3];
  }
  scratch.ctx.putImageData(scratch.image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch.canvas, 0, 0, size, size);
  ctx.imageSmoothingEnabled = true;
}

function cellPath(g: Grid, test: (i: number) => boolean, px: number): Path2D {
  const path = new Path2D();
  for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) if (test(y * g.nx + x)) path.rect(x * px, y * px, px, px);
  return path;
}

function hatch(ctx: CanvasRenderingContext2D, spacing: number, color: string, width: number): CanvasPattern | null {
  const tile = document.createElement('canvas');
  const n = Math.max(4, Math.round(spacing));
  tile.width = n; tile.height = n;
  const t = tile.getContext('2d');
  if (!t) return null;
  t.strokeStyle = color; t.lineWidth = width;
  t.beginPath();
  t.moveTo(-1, n + 1); t.lineTo(n + 1, -1);
  t.moveTo(-1, 1); t.lineTo(1, -1);
  t.moveTo(n - 1, n + 1); t.lineTo(n + 1, n - 1);
  t.stroke();
  return ctx.createPattern(tile, 'repeat');
}

function rect(ctx: CanvasRenderingContext2D, r: Region, s: Scale) {
  ctx.strokeRect(r.x * s.px, r.y * s.px, r.width * s.px, r.height * s.px);
}

/** A small uppercase label on a bone chip. (x, y) is the chip's top-left corner in canvas pixels. */
function chip(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, s: Scale, size: number, align: 'left' | 'right' = 'left') {
  ctx.font = `500 ${size * s.unit}px 'JetBrains Mono', ui-monospace, monospace`;
  const pad = 3 * s.unit, w = ctx.measureText(text).width + pad * 2, h = size * s.unit + pad * 1.6;
  const left = align === 'right' ? x - w : x;
  ctx.fillStyle = 'rgba(244, 241, 234, 0.9)';
  ctx.fillRect(left, y, w, h);
  ctx.fillStyle = INK;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  ctx.fillText(text, left + pad, y + h / 2 + 0.5 * s.unit);
  return h;
}

export interface OverlayOptions {
  /** Canvas pixels per CSS pixel (or per nominal pixel for the 3D texture). */
  unit: number;
  /** Draw garden IDs inside candidate squares (only when they are large enough to read). */
  ids: boolean;
  /** Draw the scale bar (flat map only). */
  scaleBar: boolean;
}

/** Masks, outlines and labels, drawn above water. */
export function paintOverlay(ctx: CanvasRenderingContext2D, size: number, g: Grid, overlay: Overlay, options: OverlayOptions) {
  const s: Scale = { px: size / g.nx, unit: options.unit }, u = options.unit;
  ctx.clearRect(0, 0, size, size);
  ctx.setLineDash([]);
  ctx.lineJoin = 'miter';

  // Lake: hydro-flattened water body, a fixed-level open sink. Hatched navy, distinct from computed depth.
  if (g.sink?.some(Boolean)) {
    const lake = cellPath(g, i => !!g.sink?.[i], s.px);
    ctx.fillStyle = LAKE;
    ctx.fill(lake);
    const pattern = hatch(ctx, 7 * u, 'rgba(244, 241, 234, 0.45)', Math.max(1, 0.9 * u));
    if (pattern) { ctx.fillStyle = pattern; ctx.fill(lake); }
  }

  // Building footprints: masked from the computation (no rain, no flow through).
  const masked = cellPath(g, i => !g.mask[i], s.px);
  ctx.fillStyle = 'rgba(16, 15, 13, 0.86)';
  ctx.fill(masked);
  const cross = hatch(ctx, 5 * u, 'rgba(244, 241, 234, 0.35)', Math.max(1, 0.8 * u));
  if (cross) { ctx.fillStyle = cross; ctx.fill(masked); }

  // Study-site outline.
  const site = fullSite(g);
  ctx.strokeStyle = BONE; ctx.lineWidth = 4.5 * u; rect(ctx, site, s);
  ctx.strokeStyle = INK; ctx.lineWidth = 2 * u; rect(ctx, site, s);

  // Candidate gardens: dashed bone squares; selected gardens filled terra.
  const selected = new Set(overlay.selected.map(p => p.id));
  for (const p of overlay.candidates) {
    if (selected.has(p.id)) continue;
    const lit = p.id === overlay.highlight;
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(16, 15, 13, 0.4)'; ctx.lineWidth = (lit ? 4 : 2.6) * u; rect(ctx, p, s);
    ctx.setLineDash(lit ? [] : [3.5 * u, 2.5 * u]);
    ctx.strokeStyle = BONE; ctx.lineWidth = (lit ? 2.2 : 1.3) * u; rect(ctx, p, s);
  }
  ctx.setLineDash([]);
  for (const p of overlay.selected) {
    ctx.fillStyle = overlay.gardens === 'computed' ? 'rgba(218, 83, 44, 0.6)' : 'rgba(218, 83, 44, 0.28)';
    ctx.fillRect(p.x * s.px, p.y * s.px, p.width * s.px, p.height * s.px);
    ctx.strokeStyle = p.id === overlay.highlight ? BONE : TERRA_DARK;
    ctx.lineWidth = 2.2 * u; rect(ctx, p, s);
  }
  if (options.ids) {
    ctx.font = `500 ${9 * u}px 'JetBrains Mono', ui-monospace, monospace`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
    for (const p of overlay.candidates) {
      const cx = (p.x + p.width / 2) * s.px, cy = (p.y + p.height / 2) * s.px;
      ctx.strokeStyle = selected.has(p.id) ? TERRA_DARK : 'rgba(16, 15, 13, 0.75)';
      ctx.lineWidth = 2.6 * u;
      ctx.strokeText(p.id, cx, cy);
      ctx.fillStyle = BONE;
      ctx.fillText(p.id, cx, cy);
    }
    ctx.lineJoin = 'miter';
  }

  // Inspected target: ink dashes on a bone halo.
  const r = overlay.roi;
  ctx.strokeStyle = BONE; ctx.lineWidth = 5 * u; rect(ctx, r, s);
  ctx.setLineDash([6 * u, 3.5 * u]);
  ctx.strokeStyle = INK; ctx.lineWidth = 2.4 * u; rect(ctx, r, s);
  ctx.setLineDash([]);

  // Labels.
  const label = 9;
  const chipH = (label + 4.8) * u, below = (r.y + r.height) * s.px + 4 * u;
  // Below the target (usually pavement, where no candidate sits); above it near the bottom edge.
  chip(ctx, 'TARGET', r.x * s.px, below + chipH < size ? below : r.y * s.px - chipH - 4 * u, s, label);
  chip(ctx, '256 M STUDY SITE', site.x * s.px, Math.max(2 * u, site.y * s.px - chipH - 3 * u), s, label);
  chip(ctx, 'COMPUTED BUFFER', 5 * u, 5 * u, s, label);
  if (g.sink?.some(Boolean)) chip(ctx, 'LAKE · OPEN SINK', size - 5 * u, 5 * u, s, label, 'right');

  if (options.scaleBar) {
    const bar = 50 / g.dx * s.px, x = size - bar - 8 * u, y = size - 9 * u;
    ctx.fillStyle = 'rgba(244, 241, 234, 0.9)';
    ctx.fillRect(x - 5 * u, y - 15 * u, bar + 10 * u, 20 * u);
    ctx.fillStyle = INK;
    ctx.fillRect(x, y - 1.5 * u, bar, 3 * u);
    ctx.font = `500 ${9 * u}px 'JetBrains Mono', ui-monospace, monospace`;
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('50 m', x, y - 5 * u);
  }
}

