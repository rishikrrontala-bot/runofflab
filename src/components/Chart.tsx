import type { Run } from '../core/types';
import { metricLabel } from './format';

interface Props {
  baseline?: Run;
  comparison?: Run;
  time: number;
}

const LEFT = 44, RIGHT = 536, TOP = 25, BOTTOM = 156;

/** Ponded water in the target over simulated time (the same quantity the primary metric takes the peak of). */
export default function Chart({ baseline, comparison, time }: Props) {
  const data = baseline?.frames ?? [], other = comparison?.frames ?? [];
  const maxT = data.at(-1)?.t || 1;
  const maxY = Math.max(0.01, ...data.map(f => f.roiExcess), ...other.map(f => f.roiExcess));
  const x = (t: number) => LEFT + t / maxT * (RIGHT - LEFT);
  const y = (v: number) => BOTTOM - v / maxY * (BOTTOM - TOP);
  const path = (run: Run) => run.frames.map((f, i) => `${i ? 'L' : 'M'}${x(f.t).toFixed(1)},${y(f.roiExcess).toFixed(1)}`).join(' ');
  const threshold = baseline?.config.thresholdM ?? 0;
  const describe = (run?: Run) => (run ? `${run.objective.toFixed(1)} cubic meters` : 'not computed');

  return (
    <figure className="chart">
      <figcaption>
        {metricLabel(threshold, 'Ponded water in target')}
        <span>m³ / simulated minutes</span>
      </figcaption>
      <svg
        viewBox="0 0 560 192"
        role="img"
        aria-label={`Target ponded water over time. Baseline peak ${describe(baseline)}. With gardens peak ${describe(comparison)}.`}
      >
        {[0, 0.5, 1].map(t => (
          <g key={t}>
            <line x1={LEFT} y1={y(t * maxY)} x2={RIGHT} y2={y(t * maxY)} stroke="#CECBBF" />
            <text x={LEFT - 8} y={y(t * maxY) + 4} textAnchor="end">{(t * maxY).toFixed(1)}</text>
          </g>
        ))}
        {baseline && <path d={path(baseline)} fill="none" stroke="#100F0D" strokeWidth="2" strokeDasharray="5 4" />}
        {comparison && <path d={path(comparison)} fill="none" stroke="#B83F1C" strokeWidth="2.5" />}
        {baseline && <line x1={x(time)} x2={x(time)} y1={TOP} y2={BOTTOM + 1} stroke="#2A7A8F" />}
        {[0, 0.5, 1].map(t => (
          <text key={t} x={LEFT + t * (RIGHT - LEFT)} y="181" textAnchor="middle">{Math.round(t * maxT / 60)}</text>
        ))}
      </svg>
      <div className="chart-key">
        <span>┄ Baseline</span>
        <span>━ With gardens</span>
      </div>
      {baseline && (
        <details>
          <summary>Read time series</summary>
          <div className="table-scroll">
            <table>
              <thead>
                <tr><th>Minute</th><th>Baseline m³</th><th>With gardens m³</th></tr>
              </thead>
              <tbody>
                {data.map((f, i) => (
                  <tr key={f.t}>
                    <td>{(f.t / 60).toFixed(1)}</td>
                    <td>{f.roiExcess.toFixed(2)}</td>
                    <td>{other[i]?.roiExcess.toFixed(2) ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </figure>
  );
}
