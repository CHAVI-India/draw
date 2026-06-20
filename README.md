# DRAW — Deep Radiotherapy Autosegmentation Workflow

[![Paper](https://img.shields.io/badge/Paper-Springer%20CCIS-blue.svg)](https://doi.org/10.1007/978-3-031-93709-5_14)
[![Conference](https://img.shields.io/badge/Conference-CVIP%202024-green.svg)](https://link.springer.com/chapter/10.1007/978-3-031-93709-5_14)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Citations-orange.svg)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZgBCreQAAAAJ&citation_for_view=ZgBCreQAAAAJ:qjMakFHDy7sC)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

End-to-end auto-segmentation pipeline for radiotherapy (RT) planning. DRAW reads DICOM CT studies, predicts organs-at-risk and clinical target volumes for multiple cancer sites, and writes back DICOM RT-Struct files for use in clinical treatment planning systems. Built on top of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) with custom logic for overlapping clinical labels and parallel multi-model inference.

![Sample Prediction](assets/BiomedicalSegmentation-Results-final.drawio.png)

---

## Status

DRAW is deployed across **three hospitals including Tata Medical Centre, Kolkata**, where the pipeline supports clinical RT planning workflows on **300+ CT scans per day**.

The prostate-cancer configuration (`TSPrime`) is the published flagship and is the reference for evaluating new cancer-site configurations.

## Paper

> **End-to-End Prostate Cancer Segmentation for RT Planning**
> Sandip Dutta⋆, Surajit Kundu, Santam Chakraborty, Indranil Mallick, Sougata Maity, Aranya Sarkar, Soumyajit Das, Sanjoy Chatterjee, Rimpa Basu Achari, Moses Arunsingh, Tapesh Bhattacharyya, Jayanta Mukhopadhyay, Nishant Chakravorty.
> *Computer Vision and Image Processing (CVIP), 2024.*
> Springer CCIS, Vol. 2477, pp. 192–204.
> IIT Kharagpur (Dept. of CSE) × Tata Medical Centre, Kolkata (Dept. of Radiation Oncology).
> DOI: [10.1007/978-3-031-93709-5_14](https://doi.org/10.1007/978-3-031-93709-5_14) | [[Google Scholar]](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZgBCreQAAAAJ&citation_for_view=ZgBCreQAAAAJ:qjMakFHDy7sC)
> ⋆ Correspondence: sandip28dutta@gmail.com

## Headline results (prostate, TSPrime configuration)

Eight RT structures segmented end-to-end on a 78-patient prospective trial dataset (split 48 / 12 / 18). Best configuration: PlainConvUNet + connected-component post-processing.

| Structure        | Dice  | Clinical acceptability (acceptable / minor edits) |
|------------------|-------|---------------------------------------------------|
| Bladder          | 0.972 | 63% / 32% |
| Anorectum        | 0.873 | 95% / 5%  |
| Bag_Bowel        | 0.910 | 37% / 53% |
| Femur_Head_L     | 0.895 | 53% / 47% |
| Femur_Head_R     | 0.883 | 78% / 22% |
| Penilebulb       | 0.677 | 47% / 47% |
| CTVp             | 0.850 | 42% / 32% |
| CTVn             | 0.852 | 47% / 37% |

The 0.677 Dice on penile bulb matches the published inter-observer variability among expert human delineators (0.67–0.70). It reflects the structural ambiguity of the anatomy, not a model failure.

End-to-end inference time: **30 min → 5 min per case** on training hardware (Xeon E5-1660 v4, GTX 1080 Ti 11GB). On a hospital workstation (Intel i7-12700K, NVIDIA T1000), the same pipeline runs in 15 minutes — still a 50% saving over manual delineation, with no clinician time required.

## Supported cancer sites

DRAW uses a YAML-driven model registry: each cancer site is configured via a file in [`config_yaml/`](config_yaml/), and adding a new site is a config change rather than a code change.

| Site               | Config                                                                      | Notes                          |
|--------------------|-----------------------------------------------------------------------------|--------------------------------|
| Prostate           | [`ts_prime.yml`](config_yaml/ts_prime.yml)                                  | Published reference (CVIP'24)  |
| Gynaecological     | [`ts_gyne.yml`](config_yaml/ts_gyne.yml)                                    | Full-bladder protocol           |
| Breast             | [`ts_breast_new.yml`](config_yaml/ts_breast_new.yml)                        |                                |
| Head & Neck        | [`ts_headneck.yml`](config_yaml/ts_headneck.yml)                            |                                |
| CNS                | [`ts_cns.yml`](config_yaml/ts_cns.yml)                                      |                                |
| Lung               | [`draw_lung.yml`](config_yaml/draw_lung.yml)                                |                                |
| Cranio-spinal (CSI)| [`draw_csi.yml`](config_yaml/draw_csi.yml)                                  |                                |

## Architecture

DRAW is a Click-based CLI (`draw`) layered into four packages under [`src/`](src/), with
dependencies pointing strictly inward (core never imports the database, the watcher, or
torch at import time):

| Package | Role | Depends on |
|---|---|---|
| [`draw_contracts`](src/draw_contracts/) | Pure DTOs + Protocols: `SegmentationJob`, `JobQueue`, `StatusSink`, `StorageBackend`. The seams between layers. | nothing |
| [`draw_core`](src/draw_core/) | Segmentation policy: preprocess, the nnU-Net adapter / warm predictor, postprocess, the `segment_study` orchestrator, the typed model registry. No DB, no GPU at import. | contracts |
| [`draw_conversion`](src/draw_conversion/) | Standalone DICOM ↔ NIfTI ↔ RT-Struct conversion. | core, contracts |
| [`draw_pipeline`](src/draw_pipeline/) | Scaffolding: SQL/in-memory job queue, watcher, continuous loop, Alembic, and the CLI entrypoint. | all of the above |

CLI commands:

| Command            | What it does |
|--------------------|--------------|
| `preprocess`       | Convert DICOM to NIfTI, normalise labels, prepare an nnU-Net dataset |
| `train-single-gpu` | Train an nnU-Net model on a prepared dataset |
| `predict`          | Run inference on a folder of studies and write DICOM RT-Struct (`--warm`, `--gpu-id`) |
| `start-pipeline`   | Continuous mode: watch a directory and process incoming studies |
| `zip-model`        | Export a trained model bundle for deployment |
| `db upgrade` / `db current` | Apply / inspect database migrations (the continuous pipeline auto-creates the schema on startup) |

### Engineering decisions worth surfacing

1. **Multi-model split for overlapping RT labels.** DICOM RT-Struct supports overlapping segmentations; NIfTI does not. CTVn (nodal target) routinely overlaps with Bag_Bowel. Rather than dropping overlap voxels by priority — which contaminates either bowel or CTVn — DRAW trains separate nnU-Net models per non-overlapping label group and recombines the outputs into a multi-label DICOM RT-Struct downstream. The TSPrime configuration uses three sub-models (OARs / CTVp / CTVn).
2. **Parallel multi-model inference.** A single PlainConvUNet does not saturate the GPU on a 512³-class CT volume. DRAW co-schedules the label-group models on the same GPU at inference, sharing input loading and post-processing. The constraint accepted for clinical correctness (label-group split) becomes the lever that fills the GPU. **~3× wall-clock speedup vs serial execution.**
3. **TTA disabled by default.** nnU-Net's default test-time augmentation runs each volume eight times with negligible Dice impact on the trial dataset. Disabling TTA gives an additional **~8× speedup**.
4. **Largest-3D-component post-processing for paired structures.** Femur_Head_L / Femur_Head_R get occasionally cross-classified across the midline. Retaining only the largest connected component per label eliminates the strays. Femur_Head_L Dice: 0.874 → 0.895. Neutral or positive on every other structure.
5. **Continuous prediction pipeline.** `start-pipeline` runs as a long-lived watcher that picks up new DICOM studies, runs inference, writes RT-Struct back, and records workflow state in the configured database. This is what powers the 300+ scans/day clinical deployment.
6. **DICOM ↔ NIfTI conversion glue** via [`dcmrtstruct2nii`](https://github.com/Sikerdebaard/dcmrtstruct2nii) (forward) and [`rt_utils`](https://github.com/qurit/rt-utils) (reverse).
7. **Database as a queue, with crash recovery.** Incoming studies are enqueued in a SQL
   table (`INIT → STARTED → PREDICTED → SENT`, plus `FAILED`). Workers claim items
   atomically; a lease + reaper re-queues studies stranded by a killed process, and the
   RT-Struct write is idempotent — so an interrupted run is reprocessed exactly-once in
   effect, not duplicated.

The custom training logic lives in [`Dutta-SD/nnunet_draw`](https://github.com/Dutta-SD/nnunet_draw), pinned as the `gpu` optional dependency in [`pyproject.toml`](pyproject.toml).

## Installation

DRAW uses [uv](https://docs.astral.sh/uv/) for dependency management. Dependencies are split
into **extras** so you can install only what a given machine needs — notably, the heavy
`torch` / nnU-Net stack lives behind the `gpu` extra, so the core and conversion layers
(and the test suite) install and run on a CPU-only box.

```bash
git clone https://github.com/CHAVI-India/draw.git
cd draw

# Inference / pipeline host (has an NVIDIA GPU):
uv sync --extra conversion --extra pipeline --extra gpu

# Dev / CI box (no GPU — runs the full non-GPU test suite):
uv sync --extra conversion --extra pipeline

cp template.env.draw.yml env.draw.yml   # then fill in the values
```

| Extra | Pulls in | Needed for |
|---|---|---|
| `conversion` | numpy, nibabel, pydicom, SimpleITK, dcmrtstruct2nii, rt-utils, isal | DICOM ↔ NIfTI conversion |
| `pipeline` | SQLAlchemy, alembic, watchdog, retry | the queue, watcher, continuous pipeline |
| `gpu` | torch, torchvision, the nnU-Net fork | training and inference |

> **GPU/CUDA note:** `torch` is intentionally unpinned. Install the wheel matching your
> host CUDA driver (e.g. the `cu118` index used historically), then `uv sync --extra gpu`.

Required values in `env.draw.yml`: `DB_URL`, `DB_NAME`, `TABLE_NAME`, `WATCH_DIR`,
`MODEL_DEF_ROOT`. The nnU-Net data directories default to `data/nnUNet_{raw,preprocessed,results}`.

## Running DRAW

All commands are invoked via the installed `draw` console script. With uv, prefix them with
`uv run` (or activate the venv: `source .venv/bin/activate`, then call `draw` directly).

```bash
uv run draw --help          # list all commands
uv run draw <command> --help
```

### 1. Preprocess — DICOM → nnU-Net dataset

Converts a folder of DICOM study directories into an nnU-Net training dataset for a model.

```bash
uv run draw preprocess \
    --root-dir data/raw/TS_Prime \   # parent dir; each child is one DICOM series
    --dataset-id 720 \               # 3-digit nnU-Net dataset id
    --dataset-name TSPrime \         # a model name from config_yaml/
    --start 0                        # sample-numbering offset (for appending batches)
# --only-original  : skip RT-Struct parsing (inference-shaped data with no ground truth)
```

### 2. Train — single GPU

Plans and trains one nnU-Net model on a prepared dataset.

```bash
uv run draw train-single-gpu \
    --model-name TSPrime \
    --dataset-id 720 \
    --model-fold 0 \
    --gpu-id 0 \
    --determine-postprocessing \     # also compute the connected-component postproc
    --train-continue                 # resume from the latest checkpoint
```

For multi-GPU (DDP) training, drive nnU-Net's trainer directly — see
[`bin/train_ddp.sh`](bin/train_ddp.sh). [`bin/ts_train.sh`](bin/ts_train.sh) is a
parameterized preprocess+train wrapper (`bash bin/ts_train.sh -n TSPrime -i 720 -d 0 -p -v`).

### 3. Predict — one-shot inference on a folder

Runs inference on every study directory under `--root-dir` and writes DICOM RT-Structs.

```bash
uv run draw predict \
    --root-dir data/raw/TSPrime_test \   # parent dir of DICOM series
    --preds-dir output \                 # where RT-Structs are written
    --dataset-name TSPrime \
    --only-original
# --warm          : keep model weights resident in-process across sub-models (faster;
#                   requires the gpu extra). Default is the cold-start subprocess path.
# --gpu-id 0      : pin inference to a specific GPU / MIG slice.
```

### 4. Start the continuous pipeline (clinical deployment)

Long-lived process: a watcher enqueues new DICOM studies dropped into `WATCH_DIR`, and a
worker segments them and writes RT-Structs. **The schema is auto-created on startup** — no
manual migration step for a fresh single-machine deployment.

```bash
uv run draw start-pipeline
```

Run it under a process supervisor for clinical uptime, e.g. a `systemd` unit with
`Restart=always` (Linux), so a crash auto-restarts and the reaper recovers any in-flight
study. See [`documentation/deployment.md`](documentation/deployment.md).

### 5. Database migrations

The pipeline auto-creates the table on startup, so most deployments need nothing here. To
evolve the schema on a long-lived database, use Alembic via the CLI:

```bash
uv run draw db current     # show the current schema revision
uv run draw db upgrade     # apply migrations up to head
```

### 6. Export a model for deployment

```bash
uv run draw zip-model --model-name TSPrime --dataset-id 720
```

## Development

```bash
uv sync --extra conversion --extra pipeline   # no GPU needed
uv run pytest                                 # full suite runs CPU-only
uv run ruff check src/ tests/
```

The test suite (unit + functional) runs without a GPU or a database server: conversion is
exercised on synthetic DICOM, segmentation orchestration uses a fake predictor, and the
queue is tested against in-memory and SQLite backends. GPU inference itself is validated
separately on real NVIDIA hardware.

## Documentation

Comprehensive documentation lives under [`documentation/`](documentation/):

- [`architecture.md`](documentation/architecture.md) — system design, module structure, key engineering decisions
- [`installation.md`](documentation/installation.md) — prerequisites, setup, hardware requirements
- [`cli_reference.md`](documentation/cli_reference.md) — full CLI reference for all commands
- [`configuration.md`](documentation/configuration.md) — environment file, YAML model configs, tunable constants
- [`data_flow.md`](documentation/data_flow.md) — format conversions (DICOM ↔ NIfTI ↔ RT-Struct), axis handling
- [`training_guide.md`](documentation/training_guide.md) — end-to-end training workflow, cloud training, resumption
- [`pipeline.md`](documentation/pipeline.md) — continuous prediction pipeline, state machine, protocol routing
- [`database.md`](documentation/database.md) — schema, queue operations, monitoring queries
- [`deployment.md`](documentation/deployment.md) — single-machine and multi-hospital deployment, troubleshooting

## Citation

If you use DRAW or its prostate configuration in academic work, please cite the paper:

```bibtex
@inproceedings{dutta2024drawrt,
  title     = {End-to-End Prostate Cancer Segmentation for RT Planning},
  author    = {Dutta, Sandip and Kundu, Surajit and Chakraborty, Santam and Mallick, Indranil and Maity, Sougata and Sarkar, Aranya and Das, Soumyajit and Chatterjee, Sanjoy and Achari, Rimpa Basu and Arunsingh, Moses and Bhattacharyya, Tapesh and Mukhopadhyay, Jayanta and Chakravorty, Nishant},
  booktitle = {Computer Vision and Image Processing (CVIP)},
  series    = {Communications in Computer and Information Science},
  volume    = {2477},
  pages     = {192--204},
  year      = {2024},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-93709-5_14}
}
```

Please also cite the underlying framework:

```bibtex
@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F and Kohl, Simon A A and Petersen, Jens and Maier-Hein, Klaus H},
  journal = {Nature Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021}
}
```

## License

[Apache License 2.0](LICENSE).

## Acknowledgements

This work was carried out under grant IIT/SRIC/CS/RFE/2023-2024/016 (19-04-2023). Patient data was used with prior consent and de-identified per [Kundu et al., J. Digital Imaging 2022](https://doi.org/10.1007/s10278-021-00576-6). Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) by the Applied Computer Vision Lab at Helmholtz Imaging and DKFZ.
