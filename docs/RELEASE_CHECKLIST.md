# Release Checklist

## Completed locally

- Isolated draft directory; no server changes.
- Core detector hashes match the three hashes supplied from the server.
- Only selected detector files included; abandoned variants excluded.
- Corrected main-run results recorded separately from historical results.
- No model weights, data, queue logs or personal connection settings included.

## Blocking public release

- Obtain current server package initializers/registries, base detector, losses,
  metrics, optimizers, logger, and dataset dependencies.
- Obtain the CURRENT hybrid loader with its installed SBI split guard.
  Do not substitute the pre-fix loader from the October 18 snapshot.
- Obtain the actual SBI channel-correction generator and its dependencies.
- Integrate region-preserving refinement and compact L12 generation into a
  portable CLI without changing their scientific behavior.
- Snapshot the exact corrected main configuration and ablation configurations.
- Verify actual installed dependency versions; do not invent version pins.
- Verify upstream licenses and keep required attribution.
- Normalize configurable paths without changing the data or split selection.
- Provide a shared model factory for training AND evaluation. Mean aggregation
  must not silently fall back to FTCA when loading an ablation checkpoint.
- Map corrected ablation outcomes to their exact configs and source hashes.
- Run a small end-to-end test in a clean environment, followed by comparison
  against the server's fixed-checkpoint predictions.

## Deliberately not done

No GitHub upload, repository creation, server source edit, experiment restart,
or publication-ready claim. Old training wrappers are not copied into the public
candidate because they contain experiment-specific absolute paths.
