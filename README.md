# DRAW — Deep Radiotherapy Autosegmentation Workflow

[![Paper](https://img.shields.io/badge/Paper-CVIP%202024-blue)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZgBCreQAAAAJ&citation_for_view=ZgBCreQAAAAJ:9yKSN-GCB0IC)
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
> IIT Kharagpur (Dept. of CSE) × Tata Medical Centre, Kolkata (Dept. of Radiation Oncology).
> [[Google Scholar]](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZgBCreQAAAAJ&citation_for_view=ZgBCreQAAAAJ:9yKSN-GCB0IC)
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

DRAW is a Click-based CLI with five top-level commands ([`main.py`](main.py)):

| Command            | Module                                                  | What it does |
|--------------------|---------------------------------------------------------|--------------|
| `preprocess`       | [`draw/preprocess/`](draw/preprocess/)                  | Convert DICOM to NIfTI, normalise labels, prepare nnU-Net dataset |
| `train-single-gpu` | [`draw/train/`](draw/train/)                            | Train an nnU-Net model on a prepared dataset |
| `predict`          | [`draw/predict/`](draw/predict/)                        | Run inference on a study and write DICOM RT-Struct |
| `start-pipeline`   | [`draw/pipeline/`](draw/pipeline/)                      | Continuous mode: watch a directory and process incoming studies |
| `zip-model`        | [`draw/impex/`](draw/impex/)                            | Export a trained model bundle for deployment |

### Engineering decisions worth surfacing

1. **Multi-model split for overlapping RT labels.** DICOM RT-Struct supports overlapping segmentations; NIfTI does not. CTVn (nodal target) routinely overlaps with Bag_Bowel. Rather than dropping overlap voxels by priority — which contaminates either bowel or CTVn — DRAW trains separate nnU-Net models per non-overlapping label group and recombines the outputs into a multi-label DICOM RT-Struct downstream. The TSPrime configuration uses three sub-models (OARs / CTVp / CTVn).
2. **Parallel multi-model inference.** A single PlainConvUNet does not saturate the GPU on a 512³-class CT volume. DRAW co-schedules the label-group models on the same GPU at inference, sharing input loading and post-processing. The constraint accepted for clinical correctness (label-group split) becomes the lever that fills the GPU. **~3× wall-clock speedup vs serial execution.**
3. **TTA disabled by default.** nnU-Net's default test-time augmentation runs each volume eight times with negligible Dice impact on the trial dataset. Disabling TTA gives an additional **~8× speedup**.
4. **Largest-3D-component post-processing for paired structures.** Femur_Head_L / Femur_Head_R get occasionally cross-classified across the midline. Retaining only the largest connected component per label eliminates the strays. Femur_Head_L Dice: 0.874 → 0.895. Neutral or positive on every other structure.
5. **Continuous prediction pipeline.** `start-pipeline` runs as a long-lived watcher that picks up new DICOM studies, runs inference, writes RT-Struct back, and records workflow state in the configured database. This is what powers the 300+ scans/day clinical deployment.
6. **DICOM ↔ NIfTI conversion glue** via [`dcmrtstruct2nii`](https://github.com/Sikerdebaard/dcmrtstruct2nii) (forward) and [`rt_utils`](https://github.com/qurit/rt-utils) (reverse).

The custom training logic lives in [`Dutta-SD/nnunet_draw`](https://github.com/Dutta-SD/nnunet_draw), which DRAW pins as a dependency in [`requirements.txt`](requirements.txt).

## Installation

```bash
git clone https://github.com/CHAVI-India/draw.git
cd draw
pip install -r requirements.txt
cp template.env.draw.yml env.draw.yml   # then fill in the values
```

Required environment variables (in `env.draw.yml`): `DB_URL`, `DB_NAME`, `TABLE_NAME`, `WATCH_DIR`, `MODEL_DEF_ROOT`.

## Usage

CLI documentation per command lives under [`documentation/`](documentation/):

- [`cli_preprocess.md`](documentation/cli_preprocess.md)
- [`cli_train.md`](documentation/cli_train.md)
- [`cli_predict.md`](documentation/cli_predict.md)
- [`cli_start_pipeline.md`](documentation/cli_start_pipeline.md)
- [`cli_zip_model.md`](documentation/cli_zip_model.md)

Design documents:
- [`design_desktop_app.md`](documentation/design_desktop_app.md) — desktop-app deployment notes
- [`design_hld_cloud.md`](documentation/design_hld_cloud.md) — high-level cloud design
- [`training _model_paperspace.md`](documentation/training%20_model_paperspace.md) — Paperspace training walkthrough

## Citation

If you use DRAW or its prostate configuration in academic work, please cite the paper:

```bibtex
@inproceedings{dutta2024drawrt,
  title     = {End-to-End Prostate Cancer Segmentation for RT Planning},
  author    = {Dutta, Sandip and Kundu, Surajit and Chakraborty, Santam and Mallick, Indranil and Maity, Sougata and Sarkar, Aranya and Das, Soumyajit and Chatterjee, Sanjoy and Achari, Rimpa Basu and Arunsingh, Moses and Bhattacharyya, Tapesh and Mukhopadhyay, Jayanta and Chakravorty, Nishant},
  booktitle = {Computer Vision and Image Processing (CVIP)},
  year      = {2024}
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
