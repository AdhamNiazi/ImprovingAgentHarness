import { useState } from "react";
import { api, type ProgressInfo } from "../api";
import ProgressBar from "./ProgressBar";

interface Props {
  running: string | null;
  progress: ProgressInfo | null;
}

interface DeterminismResult {
  prompt: string;
  trials: number;
  identical: boolean;
  responses: string[];
}

export default function GuardrailsPanel({ running, progress }: Props) {
  const [launching, setLaunching] = useState(false);
  const [fullOutput, setFullOutput] = useState<string[] | null>(null);

  // Quick determinism check
  const [detPrompt, setDetPrompt] = useState("What is 2 + 2?");
  const [detResult, setDetResult] = useState<DeterminismResult | null>(null);
  const [detLoading, setDetLoading] = useState(false);

  const runFull = async () => {
    setLaunching(true);
    const res = await api.guardrailsRun();
    setFullOutput(res.output ?? []);
    setLaunching(false);
  };

  const runDeterminism = async () => {
    setDetLoading(true);
    const res = await api.guardrailsDeterminism(detPrompt, 3);
    setDetResult(res);
    setDetLoading(false);
  };

  const isRunning = running === "guardrails" || launching;

  return (
    <div className="space-y-6 max-w-4xl">
      <h2 className="text-xl font-semibold">Guardrails & Determinism</h2>
      <p className="text-sm text-zinc-400">
        Verify that identical prompts with deterministic settings produce identical outputs.
      </p>

      {/* Quick test */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-4">
        <h3 className="text-sm font-medium text-zinc-300">Quick Determinism Check</h3>
        <p className="text-xs text-zinc-500">
          Sends the same prompt 3 times with temperature=0, seed=42, top_k=1 and checks for byte-identical responses.
        </p>
        <div className="flex gap-3">
          <input value={detPrompt} onChange={(e) => setDetPrompt(e.target.value)}
            className="flex-1 bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500"
            placeholder="Prompt to test..." />
          <button onClick={runDeterminism} disabled={detLoading}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors shrink-0">
            {detLoading ? "Testing..." : "Test"}
          </button>
        </div>

        {detResult && (
          <div className="space-y-3">
            <div className={`flex items-center gap-2 text-sm font-medium ${detResult.identical ? "text-emerald-400" : "text-red-400"}`}>
              <span className={`w-3 h-3 rounded-full ${detResult.identical ? "bg-emerald-500" : "bg-red-500"}`} />
              {detResult.identical ? "PASS - All responses identical" : "FAIL - Responses differ"}
            </div>
            <div className="space-y-2">
              {detResult.responses.map((r, i) => (
                <div key={i} className="bg-zinc-800/60 rounded-lg p-3">
                  <div className="text-xs text-zinc-500 mb-1">Trial {i + 1}</div>
                  <pre className="text-xs whitespace-pre-wrap text-zinc-300 max-h-32 overflow-y-auto">{r}</pre>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Progress bar */}
      {isRunning && progress && progress.phase && (
        <ProgressBar progress={progress} />
      )}

      {/* Full suite */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-4">
        <h3 className="text-sm font-medium text-zinc-300">Full Validation Suite</h3>
        <p className="text-xs text-zinc-500">
          Runs the complete guardrails script: 5 prompts x 3 trials + output format validation.
        </p>
        <button onClick={runFull} disabled={isRunning}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
          {isRunning ? "Running..." : "Run Full Suite"}
        </button>

        {fullOutput && (
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 max-h-96 overflow-y-auto font-mono text-xs space-y-0.5">
            {fullOutput.map((line, i) => (
              <div key={i} className={
                line.includes("[PASS]") ? "text-emerald-400" :
                line.includes("[FAIL]") ? "text-red-400" :
                line.includes("[OK]") ? "text-emerald-400" :
                line.includes("===") ? "text-zinc-500 font-bold" :
                "text-zinc-300"
              }>{line}</div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
