import { useState, useEffect } from "react";
import { api, type ProgressInfo } from "../api";
import ProgressBar from "./ProgressBar";

interface MetricRow {
  concurrency: string;
  prompt_type: string;
  cache: string;
  stop_seq: string;
  ttft_ms: string;
  tpot: string;
  latency_p50: string;
  latency_p95: string;
  latency_p99: string;
  gpu_util: string;
}

interface Props {
  running: string | null;
  progress: ProgressInfo | null;
}

export default function PerfPanel({ running, progress }: Props) {
  const [metrics, setMetrics] = useState<MetricRow[] | null>(null);
  const [launching, setLaunching] = useState(false);

  const fetchMetrics = async () => {
    try {
      const data = await api.perfMetrics();
      if (data.metrics) setMetrics(data.metrics);
    } catch { /* ignore */ }
  };

  useEffect(() => { fetchMetrics(); }, []);
  useEffect(() => { if (running === null) fetchMetrics(); }, [running]);

  const runPerf = async () => {
    setLaunching(true);
    await api.perfRun();
    setLaunching(false);
    fetchMetrics();
  };

  const isRunning = running === "perf" || launching;

  // Build simple bar chart data from metrics
  const chartData = metrics
    ? Object.entries(
        metrics.reduce<Record<string, { p50: number[]; p95: number[]; p99: number[] }>>((acc, r) => {
          const key = `c=${r.concurrency}`;
          if (!acc[key]) acc[key] = { p50: [], p95: [], p99: [] };
          acc[key].p50.push(+r.latency_p50);
          acc[key].p95.push(+r.latency_p95);
          acc[key].p99.push(+r.latency_p99);
          return acc;
        }, {})
      ).map(([label, v]) => ({
        label,
        p50: avg(v.p50),
        p95: avg(v.p95),
        p99: avg(v.p99),
      }))
    : [];

  const maxVal = Math.max(...chartData.map((d) => d.p99), 1);

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Performance</h2>
          <p className="text-sm text-zinc-400 mt-1">Load-test ollama and visualise latency metrics.</p>
        </div>
        <button onClick={runPerf} disabled={isRunning}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
          {isRunning ? "Running load test..." : "Run Load Test"}
        </button>
      </div>

      {/* Progress bar */}
      {isRunning && progress && progress.phase && (
        <ProgressBar progress={progress} />
      )}

      {/* Latency bar chart */}
      {chartData.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-medium text-zinc-300">Latency by Concurrency (ms)</h3>
          <div className="flex items-end gap-6 h-48">
            {chartData.map((d) => (
              <div key={d.label} className="flex-1 flex flex-col items-center gap-1">
                <div className="w-full flex justify-center gap-1 items-end h-40">
                  <Bar value={d.p50} max={maxVal} color="bg-indigo-500" label="P50" />
                  <Bar value={d.p95} max={maxVal} color="bg-amber-500" label="P95" />
                  <Bar value={d.p99} max={maxVal} color="bg-red-500" label="P99" />
                </div>
                <span className="text-xs text-zinc-400">{d.label}</span>
              </div>
            ))}
          </div>
          <div className="flex gap-4 justify-center text-xs text-zinc-400">
            <Legend color="bg-indigo-500" label="P50" />
            <Legend color="bg-amber-500" label="P95" />
            <Legend color="bg-red-500" label="P99" />
          </div>
        </div>
      )}

      {/* Metrics table */}
      {metrics && metrics.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-zinc-800/50">
              <tr>
                {["Conc.", "Prompt", "Cache", "Stop", "TTFT (ms)", "Tok/s", "P50", "P95", "P99", "GPU"].map(
                  (h) => (
                    <th key={h} className="px-3 py-2.5 text-left font-medium text-zinc-400 whitespace-nowrap">
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {metrics.map((r, i) => (
                <tr key={i} className="hover:bg-zinc-800/40">
                  <td className="px-3 py-2 font-mono">{r.concurrency}</td>
                  <td className="px-3 py-2">{r.prompt_type}</td>
                  <td className="px-3 py-2">
                    <span className={`text-xs px-1.5 py-0.5 rounded ${r.cache === "warm" ? "bg-emerald-900/40 text-emerald-400" : "bg-zinc-800 text-zinc-400"}`}>
                      {r.cache}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-zinc-400">{r.stop_seq}</td>
                  <td className="px-3 py-2 font-mono text-sky-400">{r.ttft_ms}</td>
                  <td className="px-3 py-2 font-mono">{r.tpot}</td>
                  <td className="px-3 py-2 font-mono text-indigo-400">{r.latency_p50}</td>
                  <td className="px-3 py-2 font-mono text-amber-400">{r.latency_p95}</td>
                  <td className="px-3 py-2 font-mono text-red-400">{r.latency_p99}</td>
                  <td className="px-3 py-2 text-xs text-zinc-500 max-w-[120px] truncate">{r.gpu_util}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!metrics && (
        <p className="text-zinc-500 text-sm">No metrics yet. Run the load test to generate data.</p>
      )}
    </div>
  );
}

function avg(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function Bar({ value, max, color, label }: { value: number; max: number; color: string; label: string }) {
  const pct = Math.max(2, (value / max) * 100);
  return (
    <div className="flex flex-col items-center gap-0.5 w-5" title={`${label}: ${value.toFixed(0)} ms`}>
      <span className="text-[10px] text-zinc-500">{value.toFixed(0)}</span>
      <div className={`w-full rounded-t ${color}`} style={{ height: `${pct}%` }} />
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`w-2.5 h-2.5 rounded-sm ${color}`} />
      {label}
    </div>
  );
}
