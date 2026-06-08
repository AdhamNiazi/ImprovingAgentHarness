import { useState } from "react";
import { api } from "../api";

export default function PlaygroundPanel() {
  const [prompt, setPrompt] = useState("Explain quicksort in 3 sentences.");
  const [maxTokens, setMaxTokens] = useState(256);
  const [response, setResponse] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ elapsed_s?: number; eval_count?: number; model?: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const data = await api.generate(prompt, maxTokens);
      if (data.error) {
        setError(data.error);
      } else {
        setResponse(data.response);
        setMeta({ elapsed_s: data.elapsed_s, eval_count: data.eval_count, model: data.model });
      }
    } catch (e: any) {
      setError(e.message);
    }
    setLoading(false);
  };

  return (
    <div className="space-y-4 max-w-3xl">
      <h2 className="text-xl font-semibold">Playground</h2>
      <p className="text-sm text-zinc-400">Send prompts directly to the model and see responses.</p>

      <div className="space-y-3">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          className="w-full bg-zinc-900 border border-zinc-700 rounded-lg p-3 text-sm resize-y focus:outline-none focus:border-indigo-500"
          placeholder="Enter your prompt..."
        />

        <div className="flex items-center gap-4">
          <label className="text-sm text-zinc-400 flex items-center gap-2">
            Max tokens
            <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)}
              className="w-20 bg-zinc-900 border border-zinc-700 rounded px-2 py-1 text-sm" />
          </label>

          <button onClick={generate} disabled={loading || !prompt.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors ml-auto">
            {loading ? "Generating..." : "Generate"}
          </button>
        </div>
      </div>

      {error && <div className="bg-red-900/30 border border-red-800 text-red-300 rounded-lg p-3 text-sm">{error}</div>}

      {response !== null && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-3">
          <pre className="text-sm whitespace-pre-wrap leading-relaxed">{response}</pre>
          {meta && (
            <div className="flex gap-4 text-xs text-zinc-500 pt-2 border-t border-zinc-800">
              <span>Model: {meta.model}</span>
              <span>Time: {meta.elapsed_s}s</span>
              <span>Tokens: {meta.eval_count}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
