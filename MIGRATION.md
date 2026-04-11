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

## Agent And CLI Integration

Machine clients can use:

- `GET /capabilities`
- `GET /models`
- `GET /collections`
- `POST /agent/action`

This keeps the control plane stable for future CLI and agent integrations.
