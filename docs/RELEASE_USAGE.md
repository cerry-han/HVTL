# Release candidate usage

This directory is an independent release candidate, not the running experiment
directory. Do not overwrite the active server checkout with it.

## Environment and verification

Use a separate folder with the existing working research environment first.
`docs/server_environment_versions.json` records observed package metadata.
`requirements.in` lists direct dependencies; it is NOT a validated installation
recipe. A clean environment installation has not yet been verified. In
particular, do not blindly install the latest dependency versions.

From the release root:

```bash
python -m unittest discover -s tests -v
python run.py doctor
```

`doctor` checks actual runtime imports after checking package availability.
It does not load CLIP weights or allocate a model on the GPU.

## Data paths

Prepare `configs/paths.local.json` using `configs/paths.example.json` as its
schema. It is ignored by Git. Relative paths resolve beside the path JSON;
absolute paths are also supported. With no `--paths`, configuration paths
resolve from the release root.

The LMDB root must contain `FaceForensics++_lmdb`, `Celeb-DF-v1_lmdb`,
`Celeb-DF-v2_lmdb`, `DFDCP_lmdb`, `DFDC_lmdb`, and `UADFV_lmdb`. The metadata
directory must contain matching dataset JSON files. Use the corrected original
text database and the corrected train-only SBI database, not historical text.
SBI descriptions and `manifest.jsonl` must refer to matching source identities
and accessible real/fake image paths. Relocating the outer directory does not
rewrite paths embedded in those files. Preflight rejects missing SBI images.

```bash
python run.py check --paths configs/paths.local.json
python run.py smoke --paths configs/paths.local.json --gpu 0
```

The smoke command loads one small batch and performs a forward/backward pass.
It allocates GPU memory but does not update parameters or save a checkpoint.
It tests the first-epoch path only, not later PFSA activation, full-batch memory,
long-run stability or all ablations. Run it only when GPU capacity is available.

## Main training and evaluation

```bash
python run.py train --paths configs/paths.local.json --gpu 0
python run.py evaluate --paths configs/paths.local.json --gpu 0 --checkpoint /path/to/test/avg/ckpt_best.pth
```

Evaluation requires trusted local checkpoints because the inherited checkpoint
loader uses `torch.load`. Each invocation records a resolved configuration in a
new `outputs/launches` directory. Data and outputs are not distributed.

The default training budget remains epoch indices 0 through 10, inclusive.
Selection remains the highest mean frame AUC on the five target test datasets.
This is not independent-validation selection. The release does not change the
training loss, optimizer, seed or schedule to improve reported results.

This entry point currently supports the corrected full B/16 main model only.
Do not use it for mean-aggregation or mapping checkpoints. Those require their
matching architecture overrides; their public launchers remain pending.

## Verified locally

- Source hash reconciliation and Python syntax.
- Named/random mappings and video probability aggregation.
- CLI help, path resolution and rejection of missing data.
- Main configuration compared to the server's merged configuration: changes
  are paths, task label and disabled feature dumping (also disabled in the
  original command).

Not yet verified: clean-environment installation, CUDA execution, datasets,
full text-generation pipeline and ablation runtime entry points. Do not claim
this candidate is a fully reproduced public release yet.
