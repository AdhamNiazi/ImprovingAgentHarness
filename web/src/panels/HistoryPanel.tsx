import { useState, useEffect } from "react";
import { api } from "../api";
import SamplesViewer from "./SamplesViewer";

interface Run {
  id: number;
  kind: string;
  model: string;
  started_at: string;
  finished_at: string | null;
  exit_code: number | null;
  config: Record<string, unknown> | null;
}

interface Score {
  task: string;
  metric: string;
  value: number;
  stderr: number | null;
}

const KIND_LABELS: Record<string, string> = {
  eval: "Evaluation",
  perf: "Performance",
  guardrails: "Guardrails",
};

const KIND_COLORS: Record<string, string> = {
  eval: "bg-amber-500/20 text-amber-400",
  perf: "bg-purple-500/20 text-purple-400",
  guardrails: "bg-pink-500/20 text-pink-400",
};

export default function HistoryPanel() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [kindFilter, setKindFilter] = useState<string>("");
  const [selectedRun, setSelectedRun] = useState<Run | null>(null);
  const [scores, setScores] = useState<Score[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchRuns = async (p = page, kind = kindFilter) => {
    setLoading(true);
    try {
      const data = await api.listRuns(kind || undefined, p);
      setRuns(data.runs);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { fetchRuns(1, kindFilter); }, [kindFilter]);

  const selectRun = async (run: Run) => {
    setSelectedRun(run);
    if (run.kind === "eval") {
      const data = await api.getRun(run.id);
      setScores(data.scores || []);
    } else {
      setScores([]);
    }
  };

  const goBack = () => {
    setSelectedRun(null);
    setScores([]);
  };

  const changePage = (p: number) => {
    setPage(p);
    fetchRuns(p, kindFilter);
  };

  if (selectedRun) {
    return <RunDetail run={selectedRun} scores={scores} onBack={goBack} />;
  }

  return (
    <div className="space-y-5 max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Run History</h2>
          <p className="text-sm text-zinc-400 mt-1">
            {total} total run{total !== 1 ? "s" : ""} recorded
          </p>
        </div>
        <button onClick={() => fetchRuns(page, kindFilter)}
          className="text-xs bg-zinc-800 hover:bg-zinc-700 px-3 py-1.5 rounded-lg transition-colors">
          {loading ? "Loading..." : "Refresh"}
        </button>
      </div>

      {/* Kind filter */}
      <div className="flex gap-2">
        {["", "eval", "perf", "guardrails"].map((k) => (
          <button key={k}
            onClick={() => { setKindFilter(k); setPage(1); }}
            className={`text-xs px-3 py-1.5 rounded-lg transition-colors ${
              kindFilter === k
                ? "bg-indigo-600/20 text-indigo-400"
                : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
            }`}>
            {k ? KIND_LABELS[k] : "All"}
          </button>
        ))}
      </div>

      {/* Run list */}
      {runs.length === 0 ? (
        <p className="text-zinc-500 text-sm py-8 text-center">
          No runs recorded yet. Run an evaluation, load test, or guardrails check to see history.
        </p>
      ) : (
        <div className="space-y-2">
          {runs.map((run) => (
            <button key={run.id} onClick={() => selectRun(run)}
              className="w-full text-left bg-zinc-900 border border-zinc-800 rounded-xl p-4 hover:border-zinc-700 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono text-zinc-500">#{run.id}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${KIND_COLORS[run.kind] || "bg-zinc-800 text-zinc-400"}`}>
                    {KIND_LABELS[run.kind] || run.kind}
                  </span>
                  <span className="text-sm text-zinc-300">{run.model}</span>
                </div>
                <div className="flex items-center gap-3">
                  {run.exit_code === 0 ? (
                    <span className="w-2 h-2 rounded-full bg-emerald-500" title="Success" />
                  ) : run.exit_code !== null ? (
                    <span className="w-2 h-2 rounded-full bg-red-500" title={`Exit code: ${run.exit_code}`} />
                  ) : (
                    <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" title="Running" />
                  )}
                  <span className="text-xs text-zinc-500">
                    {formatDate(run.started_at)}
                  </span>
                </div>
              </div>
              {run.config && (
                <div className="mt-2 text-xs text-zinc-500">
                  {run.kind === "eval" && (
                    <>Tasks: {(run.config.tasks as string[])?.join(", ")} | Limit: {String(run.config.limit ?? "")}</>

                  )}
                  {run.kind === "guardrails" && run.config.prompt != null && (
                    <>Prompt: &quot;{String(run.config.prompt).slice(0, 60)}&quot;</>
                  )}
                </div>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <button onClick={() => changePage(page - 1)} disabled={page <= 1}
            className="text-xs bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 px-3 py-1.5 rounded-lg">
            Prev
          </button>
          <span className="text-xs text-zinc-400">
            Page {page} of {totalPages}
          </span>
          <button onClick={() => changePage(page + 1)} disabled={page >= totalPages}
            className="text-xs bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 px-3 py-1.5 rounded-lg">
            Next
          </button>
        </div>
      )}
    </div>
  );
}


function RunDetail({ run, scores, onBack }: { run: Run; scores: Score[]; onBack: () => void }) {
  return (
    <div className="space-y-5 max-w-5xl">
      <button onClick={onBack} className="text-sm text-indigo-400 hover:text-indigo-300 flex items-center gap-1">
        &larr; Back to runs
      </button>

      <div className="flex items-center gap-3">
        <h2 className="text-xl font-semibold">
          Run #{run.id}
        </h2>
        <span className={`text-xs px-2 py-0.5 rounded-full ${KIND_COLORS[run.kind] || ""}`}>
          {KIND_LABELS[run.kind] || run.kind}
        </span>
        {run.exit_code === 0 ? (
          <span className="text-xs text-emerald-400">Success</span>
        ) : run.exit_code !== null ? (
          <span className="text-xs text-red-400">Failed (exit {run.exit_code})</span>
        ) : null}
      </div>

      {/* Run metadata */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 grid grid-cols-2 gap-3 text-sm">
        <div><span className="text-zinc-500">Model:</span> <span className="text-zinc-200">{run.model}</span></div>
        <div><span className="text-zinc-500">Started:</span> <span className="text-zinc-200">{formatDate(run.started_at)}</span></div>
        {run.finished_at && (
          <div><span className="text-zinc-500">Finished:</span> <span className="text-zinc-200">{formatDate(run.finished_at)}</span></div>
        )}
        {run.config && (
          <div className="col-span-2">
            <span className="text-zinc-500">Config:</span>{" "}
            <span className="text-zinc-300 font-mono text-xs">{JSON.stringify(run.config)}</span>
          </div>
        )}
      </div>

      {/* Scores table (eval runs) */}
      {scores.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b border-zinc-800">
            <h3 className="text-sm font-medium text-zinc-300">Scores</h3>
          </div>
          <table className="w-full text-sm">
            <thead className="bg-zinc-800/50">
              <tr>
                <th className="text-left px-4 py-2.5 font-medium text-zinc-400">Task</th>
                <th className="text-left px-4 py-2.5 font-medium text-zinc-400">Metric</th>
                <th className="text-right px-4 py-2.5 font-medium text-zinc-400">Score</th>
                <th className="text-right px-4 py-2.5 font-medium text-zinc-400">Stderr</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {scores.map((s, i) => (
                <tr key={i} className="hover:bg-zinc-800/40">
                  <td className="px-4 py-2.5 font-mono">{s.task}</td>
                  <td className="px-4 py-2.5 text-zinc-400">{s.metric}</td>
                  <td className="px-4 py-2.5 text-right font-mono text-indigo-400">{s.value.toFixed(4)}</td>
                  <td className="px-4 py-2.5 text-right font-mono text-zinc-500">
                    {s.stderr != null ? `\u00B1${s.stderr.toFixed(4)}` : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Sample-level drill-down for eval runs */}
      {run.kind === "eval" && (
        <SamplesViewer runId={run.id} tasks={scores.map((s) => s.task).filter((v, i, a) => a.indexOf(v) === i)} />
      )}
    </div>
  );
}


function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch {
    return iso;
  }
}
