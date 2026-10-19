import type { Patch } from '../core/types';

interface Props {
  candidates: Patch[];
  selected: Patch[];
  disabled: boolean;
  onToggle: (patch: Patch) => void;
  onHighlight: (id?: string) => void;
}

/** Keyboard-operable toggle for every candidate garden, in the same north-to-south ID order as the map. */
export default function CandidateGrid({ candidates, selected, disabled, onToggle, onHighlight }: Props) {
  const chosen = new Set(selected.map(p => p.id));
  return (
    <div className="candidates" role="group" aria-label="Rain-garden candidates" onMouseLeave={() => onHighlight(undefined)}>
      {candidates.map(p => (
        <button
          key={p.id}
          type="button"
          disabled={disabled}
          aria-pressed={chosen.has(p.id)}
          aria-label={`Garden ${p.id}`}
          onClick={() => onToggle(p)}
          onMouseEnter={() => onHighlight(p.id)}
          onFocus={() => onHighlight(p.id)}
          onBlur={() => onHighlight(undefined)}
        >
          {p.id}
        </button>
      ))}
    </div>
  );
}
