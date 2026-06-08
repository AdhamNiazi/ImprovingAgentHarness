.PHONY: serve client eval perf guardrails install all api web dev

# ── Python venv ──────────────────────────────────────────────────────
VENV  = .venv
PY    = $(VENV)/Scripts/python
PIP   = $(VENV)/Scripts/pip

venv:
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt

install: venv
	cd web && npm install

# ── Pipeline scripts ─────────────────────────────────────────────────
serve:
	$(PY) serve/serve.py

client:
	$(PY) serve/client.py

eval:
	$(PY) eval_runner/run_eval.py

perf:
	$(PY) perf/load_test.py

guardrails:
	$(PY) guardrails/validate.py

# ── Dashboard ────────────────────────────────────────────────────────
api:
	$(PY) -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

web:
	cd web && npm run dev

# Start both backend API + frontend dev server
dev:
	@echo Starting API server and frontend...
	start /B $(PY) -m uvicorn api.server:app --port 8000 --reload
	cd web && npm run dev

all: serve eval perf guardrails
