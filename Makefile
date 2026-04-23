.PHONY: sync fmt lint typecheck test check \
        docker-build compose-up compose-down \
        mlflow-ui \
        run-min run-real run-sans run-agentic \
        run-min-docker \
        eval eval-full

sync:
	uv sync --extra dev

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/

test:
	uv run pytest

check: lint typecheck test

# --- Docker / compose ------------------------------------------------

docker-build:
	docker build -t ai-auditor .

# Bring up the supporting services (MLflow + Ollama). On first run the
# ollama-pull sidecar downloads the model (~4 GB for qwen2.5:7b-instruct);
# follow progress with `docker compose logs -f ollama-pull`.
compose-up:
	docker compose up -d mlflow ollama ollama-pull

compose-down:
	docker compose down

# Pull (or re-pull) the configured OLLAMA_MODEL into the ollama volume.
# Useful when changing OLLAMA_MODEL without tearing the stack down.
compose-pull-model:
	docker compose up --no-deps ollama-pull

mlflow-ui:
	@echo "Launching local MLflow UI against ./mlruns (Ctrl-C to stop)"
	uv run mlflow ui --host 0.0.0.0 --port 5000

# --- One-off analyze runs -------------------------------------------

run-min:
	uv run ai-auditor analyze data/examples/minimal_policy.pdf

run-real:
	uv run ai-auditor analyze data/examples/northwestern_infosec_policy.pdf

run-sans:
	uv run ai-auditor analyze data/examples/sans_acceptable_use.pdf

run-agentic:
	uv run ai-auditor analyze data/examples/sans_acceptable_use.pdf --agentic

# Analyse the minimal sample PDF via docker compose. Infra (MLflow +
# Ollama + model pull) is brought up automatically via depends_on; on
# first run expect a multi-minute wait while the model is pulled into
# the ollama_data volume.
run-min-docker:
	docker compose run --rm ai-auditor analyze /app/data/examples/minimal_policy.pdf

# --- Strategy evaluation --------------------------------------------

# Fast iteration: small corpus, three sample PDFs.
eval:
	CONTROLS_PATH=data/controls/iso27001_annex_a_small.yaml \
	uv run python scripts/run_eval.py

# Full 33-control corpus across all three sample PDFs.
eval-full:
	uv run python scripts/run_eval.py
