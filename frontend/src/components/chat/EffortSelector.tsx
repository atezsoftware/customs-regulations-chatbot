import type {ChatEffortLevel} from '../../types';

const EFFORT_LEVELS: {value: ChatEffortLevel; label: string}[] = [
  {value: 'low', label: 'Fast'},
  {value: 'medium', label: 'Normal'},
  {value: 'high', label: 'Thorough'},
];

export function EffortSelector({effort, onChange, disabled}: {
  effort: ChatEffortLevel;
  onChange: (effort: ChatEffortLevel) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex items-center gap-0.5 rounded-lg border border-slate-200 p-0.5">
      {EFFORT_LEVELS.map(level => (
        <button
          key={level.value}
          type="button"
          disabled={disabled}
          onClick={() => onChange(level.value)}
          className={`rounded px-2 py-1 text-xs disabled:cursor-not-allowed ${
            effort === level.value
              ? 'bg-slate-800 text-white'
              : 'text-slate-600 hover:bg-slate-50'
          }`}
        >
          {level.label}
        </button>
      ))}
    </div>
  );
}
