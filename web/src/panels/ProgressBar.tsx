import type { ProgressInfo } from "../api";

interface Props {
  progress: ProgressInfo;
}

export default function ProgressBar({ progress }: Props) {
  const { phase, pct, current, total, elapsed, eta, speed } = progress;

  if (!phase && pct >= 100) return null;

  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-3">
      {/* Phase label + counts */}
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-zinc-200 truncate mr-4">{phase || "Processing..."}</span>
        <span className="font-mono text-zinc-400 shrink-0">
          {current.toLocaleString()} / {total.toLocaleString()}
        </span>
      </div>

      {/* Bar */}
      <div className="h-3 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>

      {/* Stats row */}
      <div className="flex items-center justify-between text-xs text-zinc-500">
        <div className="flex items-center gap-3">
          <span className="font-mono text-indigo-400 text-sm font-semibold">{pct}%</span>
          {elapsed && <span>Elapsed: {elapsed}</span>}
        </div>
        <div className="flex items-center gap-3">
          {speed && <span>{speed}</span>}
          {eta && eta !== "?" && (
            <span className="bg-zinc-800 text-zinc-300 px-2 py-0.5 rounded font-mono">
              ETA {eta}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
