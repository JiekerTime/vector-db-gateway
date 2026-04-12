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
- selectable embedding devices (`cpu`, `cuda`, `auto`)
- online requests can preempt low-priority work
- low-priority work has anti-starvation protection

## Migration runtime

The current migration runtime is intentionally split across control-plane and execution-plane roles:

- `vector-db-gateway`: routing truth, logical collection state, migration events, and runtime API surface
- `do-mig`: queue-aware migration runner integrated into the gateway codebase
- `db-migrator`: execution engine for copy, transform, verify, pause, and resume
- `write-disk`: persistent queue surface for scheduled migration slices
- `n8n`: timer and orchestration trigger, not the source of routing truth

The standard production chain is:

```text
n8n -> do-mig -> write-disk -> vector-db-gateway / db-migrator
```

This means `db-migrator` is no longer treated as the migration control plane. It is a worker-style service invoked by the gateway-owned migration runtime.

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

### `GET /collections/logical`

List logical collections, current routing targets, and recent migration events.

### `GET /collections/logical/{name}/migration/events`

Read append-only migration events for resume-safe orchestration.

### `GET /do-mig/queue/items`

Inspect queued migration slices stored in `write-disk`.

### `POST /do-mig/queue/import`

Import migration queue items into the configured queue channel.

### `POST /do-mig/queue/run`

Advance the integrated migration runner by one scheduling step.

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

Queue configuration can also steer the default embedding device:

- `realtime` can prefer `cuda`
- `interactive` can prefer `auto`
- `batch` can prefer `cpu`

Request payloads can still override this with `device`.

The gateway now validates dense vector size at the API boundary and before Qdrant writes:

- wrong-size query vectors are rejected as `400`, not forwarded as Qdrant `500`
- wrong-size write vectors are rejected before upsert
- wrong model-to-collection pairings are rejected when the model registry exposes vector size
- blank texts, empty id lists, empty payload patches, and illegal `search_mode` values are rejected at request validation time
- malformed sparse vectors are rejected before they can reach scheduler or Qdrant
- upserts to dense or hybrid collections must still include a usable dense vector payload

Model registry and collection registry are intentionally decoupled:

- collections can move to a new embedding model over time
- migrations can re-embed into a different vector size without changing caller code
- external migration workers can target `/transform/embed` as a stable callback endpoint
- migration orchestration can store partition and checkpoint metadata without changing caller code

Current production migration shape:

- `knowledge`: hybrid target, `dense + sparse`
- `decision_memory`: hybrid `v2` target, with new traffic already routed to `decision_memory_v2` and legacy data backfilled asynchronously

Operational role split:

- `vector-db-gateway` owns logical collection routing and migration truth
- `db-migrator` executes migration tasks but does not own migration truth
- `do-mig` owns queue dispatch and window-aware progression

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
    "texts": ["hello vector gateway"],
    "device": "cpu"
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
    "model": "default",
    "device": "auto"
  }'
```

## Deployment device modes

Default deployment exposes GPUs to the container and lets runtime policy choose devices.

CPU-only deployment:

```bash
VECTOR_GATEWAY_GPU_MODE=off ./deploy.sh
```

Watchtower is disabled for this container by default during `./deploy.sh` because the image is normally built locally rather than pulled from a public registry. Override only if the image name is backed by a real registry:

```bash
VECTOR_GATEWAY_WATCHTOWER_ENABLE=true ./deploy.sh
```

`./deploy.sh hotpatch` restarts the running container. Treat it as a short service interruption and avoid using it during an active migration slice unless the change is urgent.

If local images are monitored by Watchtower with unqualified names such as `vector-db-gateway:latest` or `db-migrator:latest`, Watchtower may try to pull them from `docker.io/library/*` and report `pull access denied`. That log means the image is not a public Hub image; it does not by itself prove the running service is unhealthy.

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
