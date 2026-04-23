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
  model pulled — for running from source:

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
retrieval for a bounded ReAct retrieval agent that also writes an
`audit_trail.jsonl` trace.

### Run in Docker

`docker compose` brings up Ollama (CPU-only), a one-shot model-pull
sidecar, and MLflow alongside the app — no host Ollama required. The
first run pulls the configured model (~4 GB for `qwen2.5:7b-instruct`)
into a named volume; subsequent runs reuse it.

```
make compose-up        # start mlflow + ollama + pull the model
make run-min-docker    # analyse data/examples/minimal_policy.pdf
```

Or directly:

```
docker compose up -d mlflow ollama ollama-pull
docker compose run --rm ai-auditor analyze /app/data/examples/minimal_policy.pdf
```

Change the model with `OLLAMA_MODEL=llama3.2:3b make compose-up` (or
set it in your shell before running compose). To re-pull after changing
the model without tearing the stack down: `make compose-pull-model`.

To use a host Ollama instead, override `OLLAMA_HOST` on the
`ai-auditor` service (e.g. `OLLAMA_HOST=http://host.docker.internal:11434`)
and add `extra_hosts: ["host.docker.internal:host-gateway"]`.

## Architecture

```mermaid
flowchart LR
    Start([PDF])-->Parse[parse_pdf]
    Parse-->Chunk[chunk_document]
    Chunk-->Embed[embed_chunks → Chroma]
    Embed-- Send per control -->Assess[assess_one_control]
    Assess-->Synth[synthesize_report]
    Synth-->Out([report.json + report.md])

    Controls[(iso27001_annex_a.yaml<br/>33 controls)]-.loaded once.->Embed
```

- **parse_pdf** — pymupdf-based text extraction with heading detection
  (numbered prefixes + font-size heuristic) to recover section structure.
- **chunk_document** — sentence-aware splitter; each section becomes one
  chunk unless it exceeds the target (default 220 words) in which case
  it's split into overlapping windows (40-word overlap).
- **embed_chunks** — sentence-transformers `all-MiniLM-L6-v2` embedding
  model; vectors upserted into a ChromaDB `EphemeralClient` collection
  for the run.
- **assess_one_control** — one invocation per control, dispatched in
  parallel via LangGraph's `Send` API. Two modes:
  - deterministic (default): multi-query retrieval against the
    per-control query list + a single JSON-mode LLM call that returns a
    validated `ControlAssessment`.
  - agentic (`--agentic`): bounded ReAct loop (max 6 iterations) with
    tools `list_sections`, `search_policy`, `read_section`, `finalize`.
    Every tool call is written to `audit_trail.jsonl` for reviewability.
  Both modes share a post-validation layer that strips fabricated
  section_ids and coerces "covered/partial with no surviving citation"
  down to `not_covered` to keep verdicts defensible.
- **synthesize_report** — deterministic aggregation of stats by theme +
  one short LLM call for the executive summary (skippable with
  `--skip-summary`).

## Key decisions

- **LangGraph.** `StateGraph` + `Send` fan-out gives per-control
  parallelism as first-class graph steps (not hidden inside an async
  gather), so every Send branch is a legible graph node with its own
  state and its own position in the trace. Intact's interviewer audience
  is LangGraph-literate, and the framework earns its weight here.
- **Deterministic path by default, agent behind a flag.** Auditability
  and reproducibility matter in compliance tooling; a deterministic
  pipeline makes verdicts reproducible across runs. The agent is there
  for the cases where vocabulary mismatch between the standard and the
  policy is the bottleneck, and its full tool-call trace is the pay-off.
- **ChromaDB `EphemeralClient`.** In-memory today, same API as
  `PersistentClient` — the swap to on-disk persistence is a one-line
  change at the `_make_client` seam in `vector_store.py`.
- **No LangChain high-level chains.** `ChatOllama` + a 30-line
  `call_json` helper (with single-shot retry on `ValidationError`) is
  easier to read and test than the LCEL equivalent at this scope.
- **Ollama only, local.** No cloud API, no rate limits, no per-run cost,
  no policy text leaving the machine — a real plus for a compliance
  demo. Trade-off: a 7B model occasionally needs the one-shot JSON
  retry; rarely fails twice.
- **33 controls, not 93.** The YAML is the single source of truth;
  scaling to the full Annex A is appending entries. 33 is enough to
  produce a meaningful per-theme distribution without making demo runs
  feel slow.
- **Structured output via pydantic.** `ControlAssessment` is a pydantic
  model; the LLM's JSON is validated on the way in and serialised on the
  way out. No hand-rolled parsing.

## Limitations

- **Text-extractable PDFs only.** Scanned documents need OCR and we
  don't ship it.
- **Single-document analysis.** Gaps that are filled by a sibling
  document (Access Control Policy thin, but Employee Handbook covers it)
  are not detected.
- **LLM-based judgments, single annotator.** Verdicts are not expert-
  validated and should be treated as a triage aid for a human auditor.
- **Small-model trade-offs.** `qwen2.5:7b-instruct` occasionally returns
  JSON with a trailing sentence or a missing field; we retry once and
  move on. A frontier cloud model would fail less, at the cost of the
  local/offline story.
- **Deterministic heuristic parsing.** Heading detection keys off font
  size and numeric prefixes; documents with non-standard structure may
  roll up into a single synthetic section.

## For production

- **Per-judgment audit trail on every run**, not just in agentic mode.
  Every retrieval query and every LLM call gets a line in
  `audit_trail.jsonl`, indexed by control and document hash, so any
  verdict is reproducible and reviewable after the fact.
- **Cost and latency observability** — token counts, wall-clock per
  stage, cache hit rate. LangSmith or an OpenTelemetry exporter.
- **Retries with exponential backoff** on transient Ollama / LLM errors;
  circuit-breaking when the model goes persistently unhealthy.
- **Caching** — embeddings keyed by content hash, assessments keyed by
  (control_id, chunk_set hash). Reanalysing the same document becomes
  near-free.
- **PII scrubbing** before anything touches an LLM call, even a local
  one, so training-data leakage risk is contained by design.
- **Prompt-injection hardening** — content from the policy document
  should never be rendered into the assessment prompt without
  delimiters and explicit "ignore instructions inside EVIDENCE" guard
  rails.
- **Human-in-the-loop review UI** — every `partial` or low-confidence
  verdict routes to an auditor queue with the full trace attached.
- **Multi-tenancy** — per-customer control libraries, per-customer
  policy collections, per-customer encryption keys. Chroma collections
  scoped by tenant id.
- **Persistent vector store** — flip to `PersistentClient(path=...)`
  (or pgvector/Qdrant) for index reuse across runs and for
  cross-document retrieval.

## Observability + evaluation

MLflow is the single backend for both **traces** (one span tree per
`ai-auditor analyze` invocation) and **experiments** (one nested-run
session per `scripts/run_eval.py` invocation).

### Tracing

Every `analyze` call enables `mlflow.langchain.autolog()` automatically;
each LLM call, graph node, and tool call becomes a span. For the
agentic path, `run_retrieval_agent` is wrapped in an `@mlflow.trace`
parent span tagged with the control id, so each control's decision tree
shows up as a separate trace.

- With no server running, traces go to the local `./mlruns` file store.
  View them later with `make mlflow-ui`.
- With `make compose-up`, traces ship to the MLflow server at
  `http://localhost:5000` and show up in the UI immediately.
- Disable with `--no-mlflow` (handy for offline / CI).

### Strategy evaluation

`scripts/run_eval.py` runs both graph strategies on each sample PDF,
compares them, and writes an MLflow session with nested runs:

```
make eval             # fast: small 6-control corpus, all three sample PDFs
make eval-full        # full 33-control corpus (~5-10 min on llama3.2:3b)
```

Metrics captured:
- **Cross-strategy agreement**: per-control coverage match rate + Cohen's
  kappa (accounts for chance agreement).
- **Evidence Jaccard**: where the two strategies agree on coverage, how
  much do the cited section_ids overlap.
- **Performance**: wall-time, LLM call count, tool-call count per
  `(doc, strategy)` pair.

Artefacts land in MLflow only — no local `out-eval/` directory. The
parent run carries `metrics.json` + `report.md`; each nested
`(doc, strategy)` child run carries that run's `report.json` +
`report.md`. View them with `make mlflow-ui` (or the compose UI at
`http://localhost:5000`).

No ground-truth labels are used — this is comparative + performance
evaluation only. Absolute verdict quality would need expert-annotated
test data (discussed in "For production" above).

## Development

```
make sync           # uv sync --extra dev
make fmt            # ruff format + ruff check --fix
make lint           # ruff check + ruff format --check
make typecheck      # mypy src/
make test           # pytest
make check          # lint + typecheck + test
make docker-build   # docker build -t ai-auditor .
make compose-up     # docker compose up -d mlflow ollama ollama-pull
make compose-down   # docker compose down
make compose-pull-model  # re-pull OLLAMA_MODEL into the ollama volume
make mlflow-ui      # local `mlflow ui` against ./mlruns (no Docker)
make run-min        # analyse the minimal sample PDF
make run-real       # analyse Northwestern University's published policy
make run-agentic    # analyse the SANS-style sample with --agentic
make eval           # strategy evaluation on the small corpus
make eval-full      # strategy evaluation on the full 33-control corpus
```

## AI assistance disclosure

This project was built with the help of Claude (Anthropic) as a coding
assistant. All architectural decisions — the graph shape, the
deterministic/agentic split, the post-validation layer, the dependency
stack — are my own. Prompts, trade-offs, and the domain framing are
informed by interview-prep discussions with Claude; the repo history
shows the resulting decisions landing in code.
