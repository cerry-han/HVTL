# Snapshot reconciliation: 2026-10-21

This records the initial reconciliation stage. For subsequent entry-point and
portability changes, use RELEASE_USAGE.md and configuration_check.json.

The private server archive contains 400 source and provenance files. Every
file matched its exported SHA-256. All 11 initial draft sources matched the
current server snapshot. All Python files parsed on the local interpreter;
this is not an import, dependency-installation or GPU execution test.

Selected dataset, metric, trainer and optimizer support files have now been
copied without changes. The corrected hybrid loader and its split guard are
included. Author headers are retained. The exported upstream license is
preserved in `licenses/DeepFakeBench-LICENSE`; third-party attribution still
needs review. This does not assign a license to new contributions.

The recorded environment is an observation, not an installable lock file.
Package metadata alone does not establish a compatible clean installation.

## Remaining release work

- Narrow detector and loss registration to the released method.
- Provide portable training, evaluation and ablation entry points.
- Parameterize dataset paths and include sanitized corrected configurations.
- Integrate the corrected text-generation pipeline, not obsolete generators.
- Verify imports and install dependencies in a clean environment.
- Run dataset preflight and a GPU smoke test using the release directory.
- Complete secret/path and third-party attribution checks before publication.

Old configurations, queue scripts and unrelated detectors remain only in the
private snapshot. No experiment results have been changed by this preparation.
