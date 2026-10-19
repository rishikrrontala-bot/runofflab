import type { SolverId } from '../core/types';

/** Fixed-decimal number for measurements; an em dash when there is nothing computed. */
export function fmt(value: number | undefined | null, decimals = 1): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—';
  const rounded = Number(value.toFixed(decimals));
  return (Object.is(rounded, -0) ? 0 : rounded).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/** Signed change with a true minus sign. */
export function signed(value: number, decimals = 1): string {
  const text = fmt(Math.abs(value), decimals);
  if (Number(text.replace(/,/g, '')) === 0) return fmt(0, decimals);
  return `${value < 0 ? '−' : '+'}${text}`;
}

/** The objective's label. With a threshold the metric only counts water deeper than it. */
export function metricLabel(thresholdM: number, base = 'Peak ponded water in target'): string {
  return thresholdM > 0 ? `${base} above ${Math.round(thresholdM * 1000) / 10} cm` : base;
}

export const SOLVER_NAMES: Record<SolverId, string> = {
  'li-1.0.0': 'local inertial',
  'dw-1.0.0': 'diffusive wave (reference)',
};

export const solverLabel = (id: SolverId) => `${id} · ${SOLVER_NAMES[id] ?? 'unknown'}`;

/** Planner method names returned by the local model endpoint. Unknown labels are shown as sent. */
export function modelMethodLabel(method: unknown, model: unknown): string {
  const names: Record<string, string> = {
    'trained-local': 'Trained local model',
    'trained-adaption': 'Trained model (Adaption LoRA, run locally)',
    'untuned-local': 'Untuned local model',
    'untuned-hosted': 'Untuned hosted model',
  };
  const name = typeof method === 'string' ? names[method] ?? method : 'Model';
  return typeof model === 'string' && model ? `${name} · ${model}` : name;
}
