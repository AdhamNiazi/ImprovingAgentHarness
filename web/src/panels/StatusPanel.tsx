import { useEffect, useState } from "react";
import { api } from "../api";

interface Status {
  ollama_running: boolean;
  ollama_url: string;
  models: string[];
  target_model: string;
  model_ready: boolean;
  has_eval_results: boolean;
  has_perf_metrics: boolean;
}

export default function StatusPanel() {
  const [status, setStatus] = useState<Status | null>(null);
  const [loading, setLoading] = useState(false);
  const [startingServe, setStartingServe] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      setStatus(await api.status());
    } catch {
      setStatus(null);
    }
    setLoading(false);
  };

  useEffect(() => { refresh(); }, []);

  const startServe = async () => {
    setStartingServe(true);
    await api.serveStart();
    setStartingServe(false);
    refresh();
  };

  const Dot = ({ ok }: { ok: boolean }) => (
    <span className={`inline-block w-2.5 h-2.5 rounded-full ${ok ? "bg-emerald-500" : "bg-red-500"}`} />
  );

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">System Status</h2>
        <button onClick={refresh} disabled={loading}
          className="text-xs bg-zinc-800 hover:bg-zinc-700 px-3 py-1.5 rounded-lg transition-colors">
          {loading ? "Checking..." : "Refresh"}
        </button>
      </div>

      {!status ? (
        <p className="text-zinc-500">Could not reach backend API. Is the server running?</p>
      ) : (
        <div className="grid gap-4">
          <Card>
            <Row label="Ollama Server"><Dot ok={status.ollama_running} /> {status.ollama_running ? "Running" : "Stopped"}</Row>
            <Row label="URL">{status.ollama_url}</Row>
            <Row label="Models">{status.models.length > 0 ? status.models.join(", ") : "None"}</Row>
            <Row label="Target Model">
              <Dot ok={status.model_ready} /> {status.target_model} {status.model_ready ? "(ready)" : "(not pulled)"}
            </Row>
          </Card>

          <Card>
            <Row label="Eval Results"><Dot ok={status.has_eval_results} /> {status.has_eval_results ? "Available" : "Not yet run"}</Row>
            <Row label="Perf Metrics"><Dot ok={status.has_perf_metrics} /> {status.has_perf_metrics ? "Available" : "Not yet run"}</Row>
          </Card>

          {!status.ollama_running && (
            <button onClick={startServe} disabled={startingServe}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
              {startingServe ? "Starting ollama..." : "Start Ollama & Pull Model"}
            </button>
          )}

          {status.ollama_running && !status.model_ready && (
            <button onClick={startServe} disabled={startingServe}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
              {startingServe ? "Pulling model..." : "Pull Model"}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 space-y-3">{children}</div>;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-zinc-400">{label}</span>
      <span className="flex items-center gap-2">{children}</span>
    </div>
  );
}
