import { useEffect, useRef } from "react";

interface Props {
  logs: { tag: string; line: string }[];
  onClear: () => void;
}

const TAG_COLORS: Record<string, string> = {
  serve: "text-emerald-400",
  client: "text-sky-400",
  eval: "text-amber-400",
  perf: "text-purple-400",
  guardrails: "text-pink-400",
};

export default function LogPanel({ logs, onClear }: Props) {
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <aside className="w-80 border-l border-zinc-800 flex flex-col shrink-0">
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <span className="text-xs font-medium text-zinc-400 uppercase tracking-wider">Live Logs</span>
        <button onClick={onClear} className="text-xs text-zinc-500 hover:text-zinc-300">Clear</button>
      </div>
      <div className="flex-1 overflow-y-auto p-3 font-mono text-xs space-y-0.5">
        {logs.length === 0 && <p className="text-zinc-600 italic">No logs yet. Run a task to see output.</p>}
        {logs.map((l, i) => (
          <div key={i} className="flex gap-2">
            <span className={`shrink-0 ${TAG_COLORS[l.tag] || "text-zinc-500"}`}>[{l.tag}]</span>
            <span className="text-zinc-300 break-all">{l.line}</span>
          </div>
        ))}
        <div ref={bottom} />
      </div>
    </aside>
  );
}
