import type { Accounting, Frame, Run } from '../core/types';
import { fmt, metricLabel, signed, solverLabel } from './format';

/** Below this baseline volume a percentage change says more about rounding than about the design. */
const PERCENT_FLOOR_M3 = 0.05;

interface Props {
  baseline?: Run;
  /** Only passed when the comparison was computed with the current settings and the same storm. */
  comparison?: Run;
  baseFrame?: Frame;
  compareFrame?: Frame;
}

const ACCOUNTING_ROWS: [label: string, key: keyof Accounting][] = [
  ['Rain input', 'rain'],
  ['Initial storage', 'initial'],
  ['Surface water', 'surface'],
  ['Soil', 'soil'],
  ['Held in gardens', 'garden'],
  ['Boundary outflow', 'outflow'],
  ['Into lake', 'waterBody'],
  ['Residual', 'residual'],
];

export function ResultsPanel({ baseline, comparison }: Props) {
  const threshold = baseline?.config.thresholdM ?? 0;
  const change = baseline && comparison ? comparison.objective - baseline.objective : undefined;
  const percent = change === undefined || !baseline
    ? undefined
    : baseline.objective < PERCENT_FLOOR_M3 ? null : change / baseline.objective * 100;

  return (
    <>
      <p className="hint">Same storm, same target, same observation window. A negative change means less water ponded in the target.</p>
      <div className="result-feature">
        <span className="result-label">{metricLabel(threshold)}</span>
        <strong>{fmt(baseline?.objective)} <small>m³</small></strong>
        <span className="result-sub">Baseline · no gardens</span>
      </div>
      <div className="delta" aria-live="polite">
        <strong>{change === undefined ? '—' : signed(change, Math.abs(change) < 0.1 ? 2 : 1)} <small>m³</small></strong>
        <span>
          {change === undefined
            ? 'Change with gardens appears after a same-storm comparison.'
            : <>With gardens {fmt(comparison!.objective)} m³ · {percent === null
                ? `no % shown (baseline under ${PERCENT_FLOOR_M3} m³)`
                : `${signed(percent!, 1)}%`}</>}
        </span>
      </div>
      <table className="metrics-table">
        <thead>
          <tr><th scope="col"><span className="sr-only">Measure</span></th><th scope="col">Baseline</th><th scope="col">Gardens</th></tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Peak target depth · cm</th>
            <td>{fmt(baseline && baseline.peakRoiDepth * 100)}</td>
            <td>{fmt(comparison && comparison.peakRoiDepth * 100)}</td>
          </tr>
          {threshold > 0 && (
            <tr>
              <th scope="row">Peak total water in target · m³</th>
              <td>{fmt(baseline?.peakRoiVolume)}</td>
              <td>{fmt(comparison?.peakRoiVolume)}</td>
            </tr>
          )}
          <tr>
            <th scope="row">Whole-domain peak water · m³</th>
            <td>{fmt(baseline?.peakDomainVolume, 0)}</td>
            <td>{fmt(comparison?.peakDomainVolume, 0)}</td>
          </tr>
          <tr>
            <th scope="row">Held in gardens at end · m³</th>
            <td>{baseline ? '0.0' : '—'}</td>
            <td>{fmt(comparison?.balance.garden)}</td>
          </tr>
        </tbody>
      </table>
      {baseline && <p className="hint mono-hint">Solver {solverLabel(baseline.config.solver)} · {baseline.steps.toLocaleString('en-US')} steps · smallest stable step {fmt(baseline.minStableDt, 2)} s</p>}
    </>
  );
}

export function AccountingTable({ baseFrame, compareFrame, comparison }: Props) {
  const balance = baseFrame?.balance, other = comparison ? compareFrame?.balance : undefined;
  return (
    <>
      <p className="hint">At the displayed simulated time · m³</p>
      <table className="accounting">
        <thead>
          <tr>
            <th scope="col">Component</th>
            <th scope="col">Baseline</th>
            {other && <th scope="col">Gardens</th>}
          </tr>
        </thead>
        <tbody>
          {ACCOUNTING_ROWS.map(([label, key]) => {
            const decimals = key === 'residual' ? 5 : 1;
            return (
              <tr key={key} className={key === 'residual' ? 'residual' : undefined}>
                <th scope="row">{label}</th>
                <td>{fmt(balance?.[key] as number | undefined, decimals)}</td>
                {other && <td>{fmt(other[key] as number | undefined, decimals)}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="hint">
        Rain + initial = surface + soil + gardens + outflow + lake + residual. Garden water is counted once: it leaves
        the surface when a basin captures it and moves to soil as it infiltrates. Lower outflow can mean delayed release.
      </p>
    </>
  );
}
