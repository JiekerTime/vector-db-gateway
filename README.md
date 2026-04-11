# vector-db-gateway

`vector-db-gateway` is an independent vector infrastructure service for self-hosted systems.

It provides:

- local embedding inference with queue-aware micro-batching
- unified vector search, count, and upsert APIs
- centralized collection registry
- centralized embedding model registry
- caller-aware priority routing with fairness safeguards
- operational endpoints for health, status, queues, and metrics

It does not implement application-specific orchestration or database-specific CRUD passthrough.

## Design

The service sits between arbitrary callers and vector infrastructure:

```text
caller
  ├── POST /embed
  ├── POST /search
  ├── POST /count
  └── POST /upsert/chunks
       |
       v
vector-db-gateway
  ├── embedding backend
  ├── queue router
  ├── fairness-aware scheduler
  ├── collection registry
  └── qdrant facade
```

Key properties:

- independent service, not a library hidden inside another application
- caller-agnostic request model
- separate queue policies for `realtime`, `interactive`, and `batch`
- online requests can preempt low-priority work
- low-priority work has anti-starvation protection

## API

### `POST /embed`

Generate vectors for one or more texts.

### `POST /search`

Search a registered collection with either a supplied vector or raw text.

### `POST /count`

Count points in a registered collection using a simplified filter format.

### `POST /upsert/chunks`

Upsert text chunks. Missing vectors are generated through the embedding backend.

### `POST /upsert/points`

Upsert points that already contain vectors.

### `GET /health`

Lightweight health probe.

### `GET /status`

Detailed runtime status, queue state, backend state, and collection registry.

### `GET /models`

List registered embedding models and vector metadata.

### `GET /capabilities`

Machine-friendly capability discovery for CLI and agent clients.

### `GET /queues`

Current queue depths and recent scheduler activity.

### `GET /metrics`

Prometheus-style metrics.

### `POST /transform/embed`

Migration-safe callback surface for bulk re-embedding jobs.

### `POST /agent/action`

Single action entrypoint for CLI and agent integrations.

## Configuration

Configuration is read from `config.yaml`.

Main sections:

- `embedding`
- `models`
- `qdrant`
- `queues`
- `routing_rules`
- `operation_priority`
- `fairness`
- `collections`

All collection metadata is centralized in config. Callers do not need to know vector size or distance settings.

Model registry and collection registry are intentionally decoupled:

- collections can move to a new embedding model over time
- migrations can re-embed into a different vector size without changing caller code
- external migration workers can target `/transform/embed` as a stable callback endpoint

## Request model

The service routes requests based on generic caller metadata:

- `caller`
- `operation`

The default configuration uses caller patterns such as:

- `realtime/*`
- `interactive/*`
- `batch/*`

Deployments can replace these rules without changing application code.

## Local development

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the service:

```bash
uvicorn server:app --host 0.0.0.0 --port 8526
```

Run tests:

```bash
python3 -m unittest discover -s tests -v
```

## Example requests

Health:

```bash
curl http://localhost:8526/health
```

Status:

```bash
curl http://localhost:8526/status -H "X-API-Key: change-me"
```

Embed:

```bash
curl http://localhost:8526/embed \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{
    "caller": "realtime/demo",
    "operation": "query",
    "texts": ["hello vector gateway"]
  }'
```

Search by text:

```bash
curl http://localhost:8526/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{
    "caller": "interactive/demo",
    "collection": "documents",
    "text": "system design",
    "limit": 5
  }'
```

Migration callback:

```bash
curl http://localhost:8526/transform/embed \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{
    "texts": ["chunk one", "chunk two"],
    "model": "default"
  }'
```

## Project layout

```text
server.py
config.yaml
requirements.txt
Dockerfile
vector_gateway/
  app.py
  config.py
  models/
  backends/
  core/
tests/
```

## Migration notes

See [MIGRATION.md](MIGRATION.md) for model evolution, vector dimension migration, and callback protocol integration.

An example callback migration task is available at [examples/callback-reembed-task.yaml](examples/callback-reembed-task.yaml).
