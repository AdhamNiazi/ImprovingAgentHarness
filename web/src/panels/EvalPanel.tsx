import { useState, useEffect } from "react";
import { api, type ProgressInfo } from "../api";
import ProgressBar from "./ProgressBar";

const AVAILABLE_TASKS = [
  { id: "python_logic", label: "Python Logic (custom)", checked: true },
  { id: "mmlu_abstract_algebra", label: "MMLU (Abstract Algebra)", checked: true },
  { id: "hellaswag", label: "HellaSwag (slow - uses loglikelihood)", checked: false },
];

interface Props {
  running: string | null;
  progress: ProgressInfo | null;
}

export default function EvalPanel({ running, progress }: Props) {
  const [tasks, setTasks] = useState(AVAILABLE_TASKS.map((t) => ({ ...t })));
  const [limit, setLimit] = useState(20);
  const [launching, setLaunching] = useState(false);
  const [results, setResults] = useState<Record<string, Record<string, number | string>> | null>(null);

  const fetchResults = async () => {
    try {
      const data = await api.evalResults();
      if (data.results) setResults(data.results);
    } catch { /* ignore */ }
  };

  useEffect(() => { fetchResults(); }, []);
  useEffect(() => { if (running === null) fetchResults(); }, [running]);

  const toggle = (id: string) =>
    setTasks((prev) => prev.map((t) => (t.id === id ? { ...t, checked: !t.checked } : t)));

  const runEval = async () => {
    const selected = tasks.filter((t) => t.checked).map((t) => t.id);
    if (selected.length === 0) return;
    setLaunching(true);
    await api.evalRun(selected, limit);
    setLaunching(false);
    fetchResults();
  };

  const isRunning = running === "eval" || launching;

  return (
    <div className="space-y-6 max-w-4xl">
      <h2 className="text-xl font-semibold">Evaluation</h2>
      <p className="text-sm text-zinc-400">
        Run benchmarks using lm-evaluation-harness with the custom ollama wrapper.
      </p>

      {/* Config */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-4">
        <h3 className="text-sm font-medium text-zinc-300">Tasks</h3>
        <div className="flex flex-wrap gap-3">
          {tasks.map((t) => (
            <label key={t.id} className="flex items-center gap-2 text-sm cursor-pointer select-none">
              <input type="checkbox" checked={t.checked} onChange={() => toggle(t.id)}
                className="accent-indigo-500" />
              {t.label}
            </label>
          ))}
        </div>

        <div className="flex items-center gap-4">
          <label className="text-sm text-zinc-400 flex items-center gap-2">
            Limit per task
            <input type="number" value={limit} onChange={(e) => setLimit(+e.target.value)}
              className="w-20 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-sm" />
          </label>

          <button onClick={runEval} disabled={isRunning || tasks.every((t) => !t.checked)}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors ml-auto">
            {isRunning ? "Running..." : "Run Evaluation"}
          </button>
        </div>
      </div>

      {/* Progress bar */}
      {isRunning && progress && progress.phase && (
        <ProgressBar progress={progress} />
      )}

      {/* Results table */}
      {results && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-zinc-800/50">
              <tr>
                <th className="text-left px-4 py-3 font-medium text-zinc-400">Task</th>
                <th className="text-left px-4 py-3 font-medium text-zinc-400">Metric</th>
                <th className="text-right px-4 py-3 font-medium text-zinc-400">Score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {Object.entries(results).map(([task, metrics]) =>
                Object.entries(metrics)
                  .filter(([k, v]) => typeof v === "number" && !k.includes("stderr") && k !== "alias")
                  .map(([metric, value]) => (
                    <tr key={`${task}-${metric}`} className="hover:bg-zinc-800/40">
                      <td className="px-4 py-2.5 font-mono">{task}</td>
                      <td className="px-4 py-2.5 text-zinc-400">{metric.replace(",none", "")}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-indigo-400">
                        {(value as number).toFixed(4)}
                      </td>
                    </tr>
                  ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
