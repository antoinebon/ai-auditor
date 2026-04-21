.PHONY: sync fmt lint typecheck test check docker-build run-min run-real run-sans run-agentic

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

docker-build:
	docker build -t ai-auditor .

run-min:
	uv run ai-auditor analyze data/examples/minimal_policy.pdf

run-real:
	uv run ai-auditor analyze data/examples/northwestern_infosec_policy.pdf

run-sans:
	uv run ai-auditor analyze data/examples/sans_acceptable_use.pdf

run-agentic:
	uv run ai-auditor analyze data/examples/sans_acceptable_use.pdf --agentic
