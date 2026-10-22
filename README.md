# HVTL

Hierarchical Vision-Text Learning for Generalizable Face Forgery Detection.

Main-model research implementation. See `docs/RELEASE_USAGE.md` for
installation notes, data paths, training and evaluation commands.

## Method

- Four-level textual supervision and cross-layer visual aggregation (FTCA).
- Within-layer ranking of related forgery and natural descriptions (HERL).
- Monotonic cross-depth responses to higher-level textual references (PFSA).
- Image-only inference; frame probabilities are averaged for video evaluation.

## Contents

- `training/detectors/`: selected model implementations, preserving original names.
- `data_tools/`: source-video split verification helper.
- `experiments/`: isolated text-mapping implementation.
- `evaluation/`: video score aggregation helper.
- `docs/`: provenance, experiment scope and data requirements.
- `tests/`: lightweight source/aggregation checks, not a training validation suite.

Use `run.py`, not individual detector files. The included tests cover source
consistency, path handling and evaluation utilities. Clean-environment GPU
training has not yet been validated for this packaged release.
No dataset, pretrained weights, training logs or credentials are included.

## Reported Corrected Main Run

| Dataset | Frame AUC | Video AUC |
| --- | ---: | ---: |
| Celeb-DF-v1 | 0.908584 | 0.943548 |
| Celeb-DF-v2 | 0.886328 | 0.933014 |
| DFDCP | 0.902897 | 0.934280 |
| DFDC | 0.796892 | 0.822370 |
| UADFV | 0.966487 | 0.982091 |
| Macro mean | 0.892238 | 0.923061 |

These values come from the supplied completed-run output, not a local rerun.
The checkpoint was selected by the highest mean frame AUC over the five target
test datasets. This is target-test-based selection, not independent validation.
Video evaluation uses that same checkpoint and averages fake probabilities
within each video. Raw prediction files and model weights are not included.

## Local Checks

From the repository root:

```bash
python -m unittest discover -s tests -v
```

These checks do not load CLIP, access datasets or establish GPU reproducibility.

## Attribution and Licensing

Parts of the implementation derive from the existing DeepFakeBench-based
research project. Preserve existing author headers. Upstream licenses,
modification notices and redistribution conditions must be verified before
public release. No new blanket license is assigned by this draft.
