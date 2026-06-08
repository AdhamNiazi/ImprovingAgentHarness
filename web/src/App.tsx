import { useState, useEffect, useCallback } from "react";
import { connectWs, type WsMsg, type ProgressInfo } from "./api";
import StatusPanel from "./panels/StatusPanel";
import PlaygroundPanel from "./panels/PlaygroundPanel";
import EvalPanel from "./panels/EvalPanel";
import PerfPanel from "./panels/PerfPanel";
import GuardrailsPanel from "./panels/GuardrailsPanel";
import ImprovePanel from "./panels/ImprovePanel";
import HistoryPanel from "./panels/HistoryPanel";
import LogPanel from "./panels/LogPanel";

const TABS = ["Status", "Playground", "Evaluation", "Improve", "Performance", "Guardrails", "History"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [tab, setTab] = useState<Tab>("Status");
  const [logs, setLogs] = useState<{ tag: string; line: string }[]>([]);
  const [running, setRunning] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);

  useEffect(() => {
    const ws = connectWs((msg: WsMsg) => {
      if (msg.type === "log") {
        setLogs((prev) => [...prev.slice(-500), { tag: msg.tag, line: msg.line }]);
      } else if (msg.type === "progress") {
        setProgress({ tag: msg.tag, phase: msg.phase, pct: msg.pct, current: msg.current, total: msg.total, elapsed: msg.elapsed, eta: msg.eta, speed: msg.speed });
      } else if (msg.type === "start") {
        setRunning(msg.tag);
        setProgress(null);
      } else if (msg.type === "done") {
        setRunning(null);
        setProgress(null);
      }
    });
    return () => ws.close();
  }, []);

  const clearLogs = useCallback(() => setLogs([]), []);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-sm">
            LM
          </div>
          <h1 className="text-lg font-semibold tracking-tight">LLM Eval Dashboard</h1>
        </div>
        {running && (
          <div className="flex items-center gap-3">
            {progress && progress.tag === running && progress.phase && (
              <div className="flex items-center gap-2 text-xs text-zinc-400">
                <span className="text-zinc-500">{progress.phase}</span>
                <span className="font-mono">{progress.current}/{progress.total}</span>
                {progress.eta && <span className="text-zinc-500">ETA {progress.eta}</span>}
              </div>
            )}
            <span className="text-xs bg-amber-500/20 text-amber-400 px-3 py-1 rounded-full animate-pulse">
              Running: {running}
            </span>
          </div>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar nav */}
        <nav className="w-52 border-r border-zinc-800 p-3 flex flex-col gap-1 shrink-0">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                tab === t
                  ? "bg-indigo-600/20 text-indigo-400 font-medium"
                  : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              }`}
            >
              {t}
            </button>
          ))}
        </nav>

        {/* Main content */}
        <main className="flex-1 overflow-y-auto p-6">
          {tab === "Status" && <StatusPanel />}
          {tab === "Playground" && <PlaygroundPanel />}
          {tab === "Evaluation" && <EvalPanel running={running} progress={running === "eval" ? progress : null} />}
          {tab === "Improve" && <ImprovePanel running={running} progress={running?.startsWith("improve") ? progress : null} />}
          {tab === "Performance" && <PerfPanel running={running} progress={running === "perf" ? progress : null} />}
          {tab === "Guardrails" && <GuardrailsPanel running={running} progress={running === "guardrails" ? progress : null} />}
          {tab === "History" && <HistoryPanel />}
        </main>

        {/* Live log sidebar */}
        <LogPanel logs={logs} onClear={clearLogs} />
      </div>
    </div>
  );
}
