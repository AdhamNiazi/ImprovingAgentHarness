import { useState, useEffect } from "react";
import { api, type ProgressInfo } from "../api";
import ProgressBar from "./ProgressBar";

const CONFIGS = [
  { id: "baseline", label: "Baseline (loglikelihood)", desc: "Raw lm-eval default scoring" },
  { id: "template_only", label: "Template Only", desc: "Instruction-tuned template + answer normalization" },
  { id: "fewshot_3", label: "Few-shot (3)", desc: "Template + 3-shot semantic retrieval" },
  { id: "full", label: "Full (SC k=5)", desc: "Template + 3-shot + self-consistency" },
];

interface Props {
  running: string | null;
  progress: ProgressInfo | null;
}

interface ConfigStats {
  accuracy: number;
  ci_95_low: number;
  ci_95_high: number;
  n: number;
  elapsed_s: number;
  config?: string;
}

export default function ImprovePanel({ running, progress }: Props) {
  const [limit, setLimit] = useState(20);
  const [launching, setLaunching] = useState(false);
  const [preparing, setPreparing] = useState(false);
  const [results, setResults] = useState<Record<string, ConfigStats> | null>(null);
  const [predictions, setPredictions] = useState<{ config: string; items: any[]; stats: ConfigStats | null; page: number; totalPages: number } | null>(null);

  const fetchResults = async () => {
    try {
      const data = await api.improveResults();
      if (data.results) setResults(data.results);
    } catch { /* ignore */ }
  };

  useEffect(() => { fetchResults(); }, []);
  useEffect(() => {
    if (running === null) fetchResults();
  }, [running]);

  const isRunning = running?.startsWith("improve") || launching || preparing;

  const prepareData = async () => {
    setPreparing(true);
    await api.improvePrepare();
    setPreparing(false);
  };

  const runAll = async () => {
    setLaunching(true);
    await api.improveRunAll(limit);
    setLaunching(false);
    fetchResults();
  };

  const runSingle = async (config: string) => {
    setLaunching(true);
    await api.improveRun(config, limit);
    setLaunching(false);
    fetchResults();
  };

  const viewPredictions = async (config: string, page = 1) => {
    const data = await api.improvePredictions(config, page);
    setPredictions({
      config,
      items: data.predictions || [],
      stats: data.stats || null,
      page: data.page || 1,
      totalPages: data.total_pages || 1,
    });
  };

  const baseline = results?.baseline;
  const bestImproved = results
    ? Object.entries(results)
        .filter(([k]) => k !== "baseline")
        .sort(([, a], [, b]) => (b.accuracy ?? 0) - (a.accuracy ?? 0))[0]
    : null;

  return (
    <div className="space-y-6 max-w-5xl">
      <h2 className="text-xl font-semibold">Part E: HellaSwag Improvement</h2>
      <p className="text-sm text-zinc-400">
        Compare baseline vs optimized inference on HellaSwag (same model, no finetuning).
      </p>

      {/* Controls */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-4">
        <div className="flex items-center gap-4 flex-wrap">
          <label className="text-sm text-zinc-400 flex items-center gap-2">
            Limit per config
            <input type="number" value={limit} onChange={(e) => setLimit(+e.target.value)}
              className="w-20 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-sm" />
          </label>

          <button onClick={prepareData} disabled={!!isRunning}
            className="bg-zinc-700 hover:bg-zinc-600 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
            {preparing ? "Preparing..." : "1. Prepare Data"}
          </button>

          <button onClick={runAll} disabled={!!isRunning}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
            {launching ? "Running..." : "2. Run All Configs"}
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          {CONFIGS.map((c) => (
            <button key={c.id} onClick={() => runSingle(c.id)} disabled={!!isRunning}
              className="bg-zinc-800 hover:bg-zinc-700 disabled:opacity-50 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border border-zinc-700"
              title={c.desc}>
              Run: {c.label}
            </button>
          ))}
        </div>
      </div>

      {/* Progress */}
      {isRunning && progress && progress.phase && (
        <ProgressBar progress={progress} />
      )}

      {/* Lift summary */}
      {baseline && bestImproved && (
        <div className="bg-gradient-to-r from-emerald-900/30 to-indigo-900/30 border border-emerald-800/50 rounded-xl p-5">
          <div className="flex items-center gap-6">
            <div>
              <div className="text-xs text-zinc-400 uppercase tracking-wider">Baseline</div>
              <div className="text-2xl font-mono font-bold text-zinc-300">{(baseline.accuracy * 100).toFixed(1)}%</div>
            </div>
            <div className="text-2xl text-zinc-500">-&gt;</div>
            <div>
              <div className="text-xs text-zinc-400 uppercase tracking-wider">Best ({bestImproved[0]})</div>
              <div className="text-2xl font-mono font-bold text-emerald-400">{(bestImproved[1].accuracy * 100).toFixed(1)}%</div>
            </div>
            <div className="ml-auto text-right">
              <div className="text-xs text-zinc-400 uppercase tracking-wider">Lift</div>
              <div className={`text-2xl font-mono font-bold ${
                bestImproved[1].accuracy > baseline.accuracy ? "text-emerald-400" : "text-red-400"
              }`}>
                {bestImproved[1].accuracy > baseline.accuracy ? "+" : ""}
                {((bestImproved[1].accuracy - baseline.accuracy) * 100).toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results table */}
      {results && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-zinc-800/50">
              <tr>
                <th className="text-left px-4 py-3 font-medium text-zinc-400">Config</th>
                <th className="text-right px-4 py-3 font-medium text-zinc-400">Accuracy</th>
                <th className="text-right px-4 py-3 font-medium text-zinc-400">95% CI</th>
                <th className="text-right px-4 py-3 font-medium text-zinc-400">N</th>
                <th className="text-right px-4 py-3 font-medium text-zinc-400">Time</th>
                <th className="text-right px-4 py-3 font-medium text-zinc-400">Lift</th>
                <th className="text-center px-4 py-3 font-medium text-zinc-400">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {Object.entries(results).map(([name, stats]) => {
                const lift = baseline && name !== "baseline"
                  ? ((stats.accuracy - baseline.accuracy) * 100)
                  : null;
                return (
                  <tr key={name} className="hover:bg-zinc-800/40">
                    <td className="px-4 py-2.5">
                      <span className="font-mono text-sm">{name}</span>
                      <div className="text-xs text-zinc-500">
                        {CONFIGS.find(c => c.id === name)?.desc ?? ""}
                      </div>
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-indigo-400">
                      {(stats.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-zinc-500 text-xs">
                      [{(stats.ci_95_low * 100).toFixed(1)}, {(stats.ci_95_high * 100).toFixed(1)}]
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-zinc-500">
                      {stats.n}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-zinc-500">
                      {stats.elapsed_s?.toFixed(1)}s
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono">
                      {lift !== null ? (
                        <span className={lift >= 0 ? "text-emerald-400" : "text-red-400"}>
                          {lift >= 0 ? "+" : ""}{lift.toFixed(2)}%
                        </span>
                      ) : (
                        <span className="text-zinc-600">--</span>
                      )}
                    </td>
                    <td className="px-4 py-2.5 text-center">
                      <button onClick={() => viewPredictions(name)}
                        className="text-xs text-indigo-400 hover:text-indigo-300 underline">
                        View
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Predictions drill-down */}
      {predictions && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium">
              Predictions: <span className="text-indigo-400 font-mono">{predictions.config}</span>
              {predictions.stats && (
                <span className="ml-2 text-zinc-500">
                  ({(predictions.stats.accuracy * 100).toFixed(1)}% acc, {predictions.stats.n} samples)
                </span>
              )}
            </h3>
            <button onClick={() => setPredictions(null)}
              className="text-xs text-zinc-500 hover:text-zinc-300">Close</button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-zinc-800/50">
                <tr>
                  <th className="px-3 py-2 text-left text-zinc-400">#</th>
                  <th className="px-3 py-2 text-left text-zinc-400">Question (truncated)</th>
                  <th className="px-3 py-2 text-center text-zinc-400">Pred</th>
                  <th className="px-3 py-2 text-center text-zinc-400">True</th>
                  <th className="px-3 py-2 text-center text-zinc-400">Correct</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {predictions.items.map((p: any) => (
                  <tr key={p.idx} className={`hover:bg-zinc-800/40 ${p.correct ? "" : "bg-red-900/10"}`}>
                    <td className="px-3 py-1.5 font-mono text-zinc-500">{p.idx}</td>
                    <td className="px-3 py-1.5 text-zinc-300 max-w-xs truncate">{p.question}</td>
                    <td className="px-3 py-1.5 text-center font-mono text-indigo-400">
                      {p.pred_label ?? p.pred}
                    </td>
                    <td className="px-3 py-1.5 text-center font-mono text-zinc-400">
                      {p.true_label ?? p.label}
                    </td>
                    <td className="px-3 py-1.5 text-center">
                      {p.correct ? (
                        <span className="text-emerald-400">Yes</span>
                      ) : (
                        <span className="text-red-400">No</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {predictions.totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 pt-2">
              <button
                onClick={() => viewPredictions(predictions.config, predictions.page - 1)}
                disabled={predictions.page <= 1}
                className="px-3 py-1 text-xs bg-zinc-800 rounded disabled:opacity-30">
                Prev
              </button>
              <span className="text-xs text-zinc-500">
                Page {predictions.page} / {predictions.totalPages}
              </span>
              <button
                onClick={() => viewPredictions(predictions.config, predictions.page + 1)}
                disabled={predictions.page >= predictions.totalPages}
                className="px-3 py-1 text-xs bg-zinc-800 rounded disabled:opacity-30">
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
