import { useState, useEffect } from "react";
import { api } from "../api";

interface Sample {
  id: number;
  doc_id: number;
  task: string;
  question: string;
  choices: string[];
  correct_idx: number;
  model_idx: number | null;
  correct: number;
  acc: number | null;
  acc_norm: number | null;
  log_probs: (number | null)[];
}

interface Props {
  runId: number;
  tasks: string[];
}

const CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"];
const PER_PAGE = 15;

export default function SamplesViewer({ runId, tasks }: Props) {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [taskFilter, setTaskFilter] = useState<string>("");
  const [correctFilter, setCorrectFilter] = useState<string>("all");
  const [loading, setLoading] = useState(false);

  const fetchSamples = async (p = page, task = taskFilter, filter = correctFilter) => {
    setLoading(true);
    try {
      const data = await api.getRunSamples(runId, {
        task: task || undefined,
        filter: filter === "all" ? undefined : filter,
        page: p,
        per_page: PER_PAGE,
      });
      setSamples(data.samples ?? []);
      setTotal(data.total ?? 0);
      setTotalPages(data.total_pages ?? 1);
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => {
    setPage(1);
    fetchSamples(1, taskFilter, correctFilter);
  }, [runId, taskFilter, correctFilter]);

  const changePage = (p: number) => {
    setPage(p);
    fetchSamples(p);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-zinc-300">
          Per-Question Results
          <span className="text-zinc-500 font-normal ml-2">({total} samples)</span>
        </h3>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        {tasks.length > 1 && (
          <div className="flex gap-1">
            <button onClick={() => setTaskFilter("")}
              className={`text-xs px-2.5 py-1 rounded-lg ${!taskFilter ? "bg-indigo-600/20 text-indigo-400" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"}`}>
              All tasks
            </button>
            {tasks.map((t) => (
              <button key={t} onClick={() => setTaskFilter(t)}
                className={`text-xs px-2.5 py-1 rounded-lg ${taskFilter === t ? "bg-indigo-600/20 text-indigo-400" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"}`}>
                {t}
              </button>
            ))}
          </div>
        )}

        <div className="flex gap-1 ml-auto">
          {(["all", "correct", "incorrect"] as const).map((f) => (
            <button key={f} onClick={() => setCorrectFilter(f)}
              className={`text-xs px-2.5 py-1 rounded-lg ${
                correctFilter === f
                  ? f === "correct" ? "bg-emerald-500/20 text-emerald-400"
                    : f === "incorrect" ? "bg-red-500/20 text-red-400"
                    : "bg-indigo-600/20 text-indigo-400"
                  : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
              }`}>
              {f === "all" ? "All" : f === "correct" ? "Correct" : "Wrong"}
            </button>
          ))}
        </div>
      </div>

      {loading && <p className="text-xs text-zinc-500">Loading...</p>}

      {/* Samples list */}
      {samples.length === 0 && !loading ? (
        <p className="text-zinc-500 text-sm py-4 text-center">No samples found.</p>
      ) : (
        <div className="space-y-2">
          {samples.map((s) => (
            <SampleCard key={s.id} sample={s} />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-1">
          <span className="text-xs text-zinc-500">
            Showing {(page - 1) * PER_PAGE + 1}-{Math.min(page * PER_PAGE, total)} of {total}
          </span>
          <div className="flex items-center gap-2">
            <button onClick={() => changePage(page - 1)} disabled={page <= 1}
              className="text-xs bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 px-3 py-1.5 rounded-lg">
              Prev
            </button>
            {/* Page number pills */}
            {pageNumbers(page, totalPages).map((p, i) =>
              p === "..." ? (
                <span key={`dots-${i}`} className="text-xs text-zinc-600">...</span>
              ) : (
                <button key={p} onClick={() => changePage(p as number)}
                  className={`text-xs px-2.5 py-1 rounded-lg ${
                    p === page ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                  }`}>
                  {p}
                </button>
              )
            )}
            <button onClick={() => changePage(page + 1)} disabled={page >= totalPages}
              className="text-xs bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 px-3 py-1.5 rounded-lg">
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


function SampleCard({ sample }: { sample: Sample }) {
  const { question, choices, correct_idx, model_idx, correct, acc_norm, log_probs, task, doc_id } = sample;
  const isCorrect = correct === 1;

  return (
    <div className={`bg-zinc-900 border rounded-xl p-4 space-y-3 ${
      isCorrect ? "border-emerald-900/50" : "border-red-900/50"
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${isCorrect ? "bg-emerald-500" : "bg-red-500"}`} />
          <span className="text-xs text-zinc-500 font-mono">{task} #{doc_id}</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          {acc_norm != null && <span>acc_norm: {acc_norm.toFixed(1)}</span>}
          <span className={isCorrect ? "text-emerald-400 font-medium" : "text-red-400 font-medium"}>
            {isCorrect ? "CORRECT" : "WRONG"}
          </span>
        </div>
      </div>

      {/* Question */}
      <p className="text-sm text-zinc-200 leading-relaxed">{question}</p>

      {/* Choices */}
      <div className="grid gap-1.5">
        {choices.map((choice, i) => {
          const isModelPick = i === model_idx;
          const isAnswer = i === correct_idx;
          let bg = "bg-zinc-800/50";
          let border = "border-transparent";
          let text = "text-zinc-400";

          if (isAnswer && isModelPick) {
            bg = "bg-emerald-900/30"; border = "border-emerald-700/50"; text = "text-emerald-300";
          } else if (isAnswer) {
            bg = "bg-emerald-900/20"; border = "border-emerald-800/40"; text = "text-emerald-400";
          } else if (isModelPick) {
            bg = "bg-red-900/20"; border = "border-red-800/40"; text = "text-red-400";
          }

          const lp = log_probs?.[i];

          return (
            <div key={i} className={`flex items-center gap-3 px-3 py-2 rounded-lg border text-sm ${bg} ${border}`}>
              <span className={`font-mono font-bold text-xs w-5 ${text}`}>
                {CHOICE_LABELS[i]}
              </span>
              <span className={`flex-1 ${text}`}>{choice}</span>
              <div className="flex items-center gap-2 shrink-0">
                {lp != null && (
                  <span className="text-xs font-mono text-zinc-600" title="Log probability">
                    {lp.toFixed(2)}
                  </span>
                )}
                {isModelPick && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-700 text-zinc-300">model</span>
                )}
                {isAnswer && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-800/50 text-emerald-300">answer</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


function pageNumbers(current: number, total: number): (number | string)[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const pages: (number | string)[] = [1];
  if (current > 3) pages.push("...");
  for (let i = Math.max(2, current - 1); i <= Math.min(total - 1, current + 1); i++) {
    pages.push(i);
  }
  if (current < total - 2) pages.push("...");
  pages.push(total);
  return pages;
}
