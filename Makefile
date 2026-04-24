.PHONY: sync fmt lint typecheck test check \
        docker-build compose-up compose-down compose-pull-model \
        mlflow-ui \
        run-tiny run-min run-real run-agentic \
        docker-run-min docker-run-real docker-run-agentic \
        eval eval-full \
        generate-queries

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
	docker compose up -d ollama ollama-pull

compose-down:
	docker compose down

# Pull (or re-pull) the configured OLLAMA_MODEL into the ollama volume.
# Useful when changing OLLAMA_MODEL without tearing the stack down.
compose-pull-model:
	docker compose up --no-deps ollama-pull

mlflow-ui:
	@echo "Launching local MLflow UI (Ctrl-C to stop)"
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# --- One-off analyze runs -------------------------------------------

run-tiny:
	CONTROLS_PATH=data/controls/iso27001_annex_a_small.yaml \
	uv run ai-auditor analyze data/examples/minimal_policy.pdf --skip-summary

run-min:
	uv run ai-auditor analyze data/examples/minimal_policy.pdf

run-real:
	uv run ai-auditor analyze data/examples/northwestern_infosec_policy.pdf

run-agentic:
	uv run ai-auditor analyze data/examples/sans_acceptable_use.pdf --agentic

# --- One-off analyze runs via docker compose -------------------------
#
# Same three sample PDFs, run inside the `ai-auditor` service. Infra
# (MLflow + Ollama + model pull) is brought up automatically via
# depends_on; on first run expect a multi-minute wait while the model
# is pulled into the ollama_data volume. Reports are written to ./out
# on the host via the volume mount.

docker-run-min:
	docker compose run --rm ai-auditor analyze /app/data/examples/minimal_policy.pdf

docker-run-real:
	docker compose run --rm ai-auditor analyze /app/data/examples/northwestern_infosec_policy.pdf

docker-run-agentic:
	docker compose run --rm ai-auditor analyze /app/data/examples/sans_acceptable_use.pdf --agentic

# --- Strategy evaluation --------------------------------------------

# Fast iteration: small corpus, three sample PDFs.
eval:
	CONTROLS_PATH=data/controls/iso27001_annex_a_small.yaml \
	uv run python scripts/run_eval.py

# Full 33-control corpus across all three sample PDFs.
eval-full:
	uv run python scripts/run_eval.py

# --- One-off: populate Control.queries via LLM ---------------------

# Generate 3-5 short retrieval-oriented paraphrases for each control
# whose `queries` is empty, and write them back to the YAML. Pass
# --force to regenerate controls that already have queries populated.
generate-queries:
	uv run python scripts/generate_queries.py
