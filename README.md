# ai-auditor

ISO 27001:2022 policy gap analyser. Reads a security policy PDF, checks it
against a curated subset of Annex A controls, and writes a structured gap
report with evidence citations per control.

Built as a LangGraph pipeline around a local Ollama model. No cloud
inference, no per-run cost, and the policy document never leaves the
machine — a good fit for a compliance-domain demo.

```text
$ uv run ai-auditor analyze data/examples/northwestern_infosec_policy.pdf
Analysing data/examples/northwestern_infosec_policy.pdf
  model=qwen2.5:7b-instruct @ http://localhost:11434
  controls=data/controls/iso27001_annex_a.yaml
  agentic=False

                       Coverage by theme
┏━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Theme           ┃ Total ┃ Covered ┃ Partial ┃ Not covered ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Organizational  │    15 │      11 │       3 │           1 │
│ People          │     4 │       3 │       1 │           0 │
│ Physical        │     4 │       2 │       1 │           1 │
│ Technological   │    10 │       7 │       2 │           1 │
├─────────────────┼───────┼─────────┼─────────┼─────────────┤
│ Total           │    33 │      23 │       7 │           3 │
└─────────────────┴───────┴─────────┴─────────┴─────────────┘

wrote out/report.json
wrote out/report.md
```

## Quick start

### Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/) — for running from
  source.
- [Ollama](https://ollama.com/) with a tool-calling / JSON-mode-capable
  model pulled:

  ```
  ollama pull qwen2.5:7b-instruct
  ```

- Or just Docker + `docker compose` — the compose stack runs Ollama in
  a container and pulls the model automatically (see "Run in Docker"
  below).

### Run from source

```
uv sync --extra dev
cp .env.example .env       # adjust OLLAMA_HOST / OLLAMA_MODEL if needed
uv run ai-auditor analyze data/examples/minimal_policy.pdf
uv run ai-auditor analyze data/examples/northwestern_infosec_policy.pdf --agentic
```

Reports are written to `out/report.json` and `out/report.md`. Use
`--output <dir>` to redirect. `--agentic` swaps deterministic multi-query
retrieval for a bounded ReAct retrieval agent.

### Run in Docker

`docker compose` brings up Ollama (CPU-only), a one-shot model-pull
sidecar, and MLflow alongside the app — no host Ollama required. The
first run pulls the configured model (~4 GB for `qwen2.5:7b-instruct`)
into a named volume; subsequent runs reuse it.

```
make compose-up           # start mlflow + ollama + pull the model
make docker-run-min       # analyse data/examples/minimal_policy.pdf
make docker-run-agentic   # same, with the agentic retrieval path
```

Change the model with `OLLAMA_MODEL=llama3.2:3b make compose-up`. To
re-pull after changing the model without tearing the stack down:
`make compose-pull-model`.

To use a host Ollama instead, override `OLLAMA_HOST` on the
`ai-auditor` service (e.g. `OLLAMA_HOST=http://host.docker.internal:11434`)
and add `extra_hosts: ["host.docker.internal:host-gateway"]`.

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — runtime architecture,
  LangGraph pipeline, state schema, data model, deterministic vs agentic
  paths, post-validation, LLM integration, DI pattern, trade-offs,
  extension points. Start here if you want to understand how the
  pipeline fits together.
- **[docs/evaluation.md](docs/evaluation.md)** — eval harness, metrics
  (agreement, kappa, evidence Jaccard), MLflow schema, how to read
  results, limitations.

The short version: parse PDF → sentence-aware chunks → sentence-transformer
embeddings into an in-memory ChromaDB → fan out one assessment branch per
control via LangGraph `Send` → either deterministic multi-query retrieval
+ single JSON-mode LLM call, or a bounded ReAct agent — both funnel
through a shared post-validator that drops fabricated citations → synthesize
one `Report`. See `docs/architecture.md` for the full story.

## Limitations

- **Text-extractable PDFs only.** Scanned documents need OCR and we don't
  ship it.
- **Single-document analysis.** Gaps filled by a sibling document (thin
  Access Control Policy + a fuller Employee Handbook) are not detected.
- **LLM-based judgments, single annotator.** Verdicts are a triage aid,
  not an audit decision. No expert-annotated ground truth.
- **Small-model trade-offs.** `qwen2.5:7b-instruct` occasionally emits
  malformed JSON; the one-shot retry in `call_json` handles most cases.
  A frontier cloud model would fail less, at the cost of the
  local/offline story.
- **Heuristic PDF parsing.** Heading detection uses font size + numeric
  prefixes; documents with non-standard structure roll up into a single
  synthetic section.

See `docs/architecture.md` (Key design trade-offs) for the production
gaps we'd close next: per-judgment audit trail on every run, caching,
prompt-injection hardening, persistent vector store, multi-tenancy.

## Development

```
make sync              # uv sync --extra dev
make fmt               # ruff format + ruff check --fix
make check             # lint + typecheck + test
make run-min           # analyse the minimal sample PDF (source)
make run-agentic       # same, --agentic
make docker-run-min    # analyse via docker compose
make eval              # small-corpus strategy evaluation
make eval-full         # full 33-control corpus
make generate-queries  # populate Control.queries via the LLM
make mlflow-ui         # local mlflow ui against ./mlruns
```

Full target list in the [Makefile](Makefile).

## AI assistance disclosure

This project was built with the help of Claude (Anthropic) as a coding
assistant. All architectural decisions — the graph shape, the
deterministic/agentic split, the post-validation layer, the dependency
stack — are my own. Prompts, trade-offs, and the domain framing are
informed by interview-prep discussions with Claude; the repo history
shows the resulting decisions landing in code.
