interface Props<T extends string> {
  label: string;
  value: T;
  options: [T, string][];
  onChange: (v: T) => void;
}

/** A row of mutually exclusive buttons that switches between input methods. */
export function Segmented<T extends string>({ label, value, options, onChange }: Props<T>) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={v} type="button" role="radio" aria-checked={value === v} onClick={() => onChange(v)}>
          {text}
        </button>
      ))}
    </div>
  );
}
