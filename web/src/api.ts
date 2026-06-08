const BASE = "";

async function post(url: string, body?: unknown) {
  const res = await fetch(BASE + url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  return res.json();
}

async function get(url: string) {
  const res = await fetch(BASE + url);
  return res.json();
}

export const api = {
  status: () => get("/api/status"),
  serveStart: () => post("/api/serve/start"),
  clientRun: () => post("/api/client/run"),
  generate: (prompt: string, max_tokens = 256) =>
    post("/api/client/generate", { prompt, max_tokens }),
  evalRun: (tasks: string[], limit: number) =>
    post("/api/eval/run", { tasks, limit }),
  evalResults: () => get("/api/eval/results"),
  evalSamples: () => get("/api/eval/samples"),
  perfRun: () => post("/api/perf/run"),
  perfMetrics: () => get("/api/perf/metrics"),
  guardrailsRun: () => post("/api/guardrails/run"),
  guardrailsDeterminism: (prompt: string, trials = 3) =>
    post("/api/guardrails/determinism", { prompt, trials }),

  listRuns: (kind?: string, page = 1, perPage = 20) =>
    get(`/api/runs?${new URLSearchParams({ ...(kind ? { kind } : {}), page: String(page), per_page: String(perPage) })}`),
  getRun: (id: number) => get(`/api/runs/${id}`),
  getRunSamples: (id: number, opts: { task?: string; filter?: string; page?: number; per_page?: number } = {}) => {
    const params: Record<string, string> = { page: String(opts.page ?? 1), per_page: String(opts.per_page ?? 25) };
    if (opts.task) params.task = opts.task;
    if (opts.filter) params.filter = opts.filter;
    return get(`/api/runs/${id}/samples?${new URLSearchParams(params)}`);
  },
  getRunPerf: (id: number) => get(`/api/runs/${id}/perf`),
  getRunGuardrails: (id: number) => get(`/api/runs/${id}/guardrails`),
  historyScores: () => get("/api/history/scores"),

  improvePrepare: () => post("/api/improve/prepare"),
  improveRun: (config: string, limit: number) =>
    post("/api/improve/run", { config, limit }),
  improveRunAll: (limit: number) =>
    post("/api/improve/run-all", { limit }),
  improveResults: () => get("/api/improve/results"),
  improvePredictions: (config: string, page = 1, perPage = 25) =>
    get(`/api/improve/predictions/${config}?page=${page}&per_page=${perPage}`),
};

export interface ProgressInfo {
  tag: string;
  phase: string;
  pct: number;
  current: number;
  total: number;
  elapsed: string;
  eta: string;
  speed: string;
}

export type WsMsg =
  | { type: "start"; tag: string }
  | { type: "log"; tag: string; line: string }
  | { type: "progress"; tag: string; phase: string; pct: number; current: number; total: number; elapsed: string; eta: string; speed: string }
  | { type: "done"; tag: string; exit_code: number };

export function connectWs(onMsg: (msg: WsMsg) => void) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${window.location.host}/ws/logs`);
  ws.onmessage = (e) => onMsg(JSON.parse(e.data));
  ws.onclose = () => setTimeout(() => connectWs(onMsg), 2000);
  return ws;
}
