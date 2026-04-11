# Model And Dimension Evolution

This service is built so callers do not need to care about vector size or embedding backend details.

## Registries

Two registries are intentionally separate:

- `models`: embedding model registry
- `collections`: collection registry

That separation allows:

- the same model to back multiple collections
- collection versioning across model generations
- re-embedding into a new vector size during migration
- staged query/write cutovers

## Callback protocol

The stable callback target for bulk re-embedding is:

```text
POST /transform/embed
```

Expected request:

```json
{
  "texts": ["chunk one", "chunk two"],
  "model": "default"
}
```

Expected response:

```json
{
  "model": "default",
  "model_name": "BAAI/bge-m3",
  "vector_size": 1024,
  "vectors": [[...], [...]]
}
```

Any migration worker that can issue HTTP callback transforms can use this protocol.

## Recommended migration flow

1. Register a new model under `models`.
2. Register a new target collection under `collections`.
3. Read source records from the old collection.
4. Re-embed through `/transform/embed`.
5. Write into the target collection.
6. Verify target counts and samples.
7. Flip callers to the new collection or alias.

## Logical Collections And Runtime Control

Production callers should use logical collection names such as `knowledge` or `decision_memory`.

The gateway owns:

- logical collection to physical collection routing
- current `read_target`
- current `write_targets`
- migration phase such as `prepare`, `dual_write`, `backfill`, `verify`, `live`, `rollback`
- a persisted migration event history in SQLite

Migration workers should not infer routing from old task files. They should read the current runtime state from the gateway.

## Partitioned Backfill

Partitioning is a database concern, not a business term.

Examples:

- `knowledge`: `partition_key = expert_id`
- `decision_memory`: `partition_key = created_at_bucket` or no partitioning

The split dimension is chosen by the migration task, not by the caller. A migration worker can backfill one partition at a time while the gateway keeps the logical collection stable.

## do-mig Responsibilities

`do-mig` should own the migration queue and execution state.

Recommended queue payload:

```json
{
  "logical_collection": "knowledge",
  "task_id": "mig-20260411-knowledge-v3",
  "phase": "backfill",
  "partition_key": "expert_id",
  "partition_values": ["karpathy", "lecun"],
  "window": {
    "start": "01:00",
    "stop_dispatch": "01:55",
    "pause_at": "02:00"
  },
  "attempt": 1,
  "priority": 50
}
```

Recommended dispatch rules:

- one queue item should represent one execution slice, not the whole migration
- a day can contain many windows, for example `01:00-02:00`, `05:00-06:00`, `13:00-14:00`, `21:00-22:00`
- each window should stop dispatch a few minutes before the hard pause time
- `do-mig` should enqueue the next slice immediately after the current slice finishes or pauses
- `decision_memory` should use the same queue model instead of a separate manual migration path

Recommended queue fields:

- `logical_collection`
- `task_id`
- `phase`
- `partition_key`
- `partition_values`
- `window`
- `attempt`
- `checkpoint`
- `priority`
- `next_run_at`
- `status`
- `last_error`

Suggested low-impact schedule:

```json
{
  "timezone": "Asia/Shanghai",
  "windows": [
    {"start": "00:30", "stop_dispatch": "01:05", "pause_at": "01:10"},
    {"start": "02:30", "stop_dispatch": "03:05", "pause_at": "03:10"},
    {"start": "04:30", "stop_dispatch": "05:05", "pause_at": "05:10"},
    {"start": "06:30", "stop_dispatch": "07:05", "pause_at": "07:10"}
  ]
}
```

This keeps migration throughput high while avoiding the main daytime traffic window.

Recommended ownership split:

- `vector-db-gateway`: routing truth and migration phase history
- `db-migrator`: execution engine for copy, transform, verify, pause, and resume
- `do-mig`: queue, execution state, scheduling policy, retries, and dispatch policy
- `n8n`: the standard timer and orchestration entrypoint; it triggers `do-mig` on schedule
- `write-disk`: the execution surface used by `do-mig` for queued task persistence or dispatch

## Scheduling Standard

The standard production chain should be:

```text
n8n -> do-mig -> write-disk -> vector-db-gateway / db-migrator
```

This means:

- recurring time windows should be defined in `n8n`
- `do-mig` decides what queue item is runnable in the current window
- `write-disk` is used as the execution and persistence surface for scheduled jobs
- the gateway remains the runtime source of truth for routing and migration phase

Emergency takeover tools such as local `cron` or direct CLI loops can still be used temporarily, but they are not the long-term scheduling contract.

## Resume Semantics

The gateway now persists both the current migration state and an append-only event history.

Each control-plane action can include structured `metadata`, for example:

- `partition_key`
- `partition_values`
- `window`
- `attempt`
- `checkpoint`

Useful endpoints:

- `GET /collections/logical`
- `GET /collections/logical/{logical_name}`
- `GET /collections/logical/{logical_name}/migration/events`
- `POST /collections/logical/{logical_name}/migration/{action}`

That lets `do-mig` resume safely:

1. read the logical collection runtime state
2. read recent migration events
3. compare its own queued task with the last recorded phase and checkpoint
4. continue with the next partition batch or resume the paused one

## Agent And CLI Integration

Machine clients can use:

- `GET /capabilities`
- `GET /models`
- `GET /collections`
- `POST /agent/action`

This keeps the control plane stable for future CLI and agent integrations.
