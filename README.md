# Deep Semantic Segmentation of Microscopic Blood Smears for Blood Cell Counting

CSE471 – Deep Learning for Computer Vision · Section K4P24VS · **Group 7**
Prabhbani Kaur (12409812) · Saksham Darwai (12409857) · Ayush Kaushik (12410308)

A UNet (implemented from scratch in PyTorch) segments red blood cells (RBC), white blood cells (WBC)
and platelets in peripheral blood-smear images from the free **BCCD** dataset, then converts the
predicted masks into automated cell counts and a cell differential (class proportions).

## Results (held-out test split, 72 images)

| Metric | RBC | WBC | Platelets |
|---|---|---|---|
| IoU | 0.615 | 0.816 | 0.570 |
| Dice | 0.761 | 0.899 | 0.726 |
| Count MAE (cells/image) | 2.88 | 0.06 | 0.26 |
| Count within ±1 | 40% | 100% | 99% |

Mean IoU over the three cell classes: **0.667** · mean Dice: **0.796** · pixel accuracy: 0.839.
Predicted differential vs ground truth: RBC 85.1 % vs 85.2 %, WBC 7.5 % vs 7.5 %, Platelets 7.4 % vs 7.3 %.

Read the **Limitations** section of `REPORT.md` before quoting these numbers: BCCD has bounding boxes
only, so masks are generated from boxes, and a few images have missing RBC labels.

## Project layout

```
BCCD/                   dataset (images + Pascal-VOC XML + official train/val/test lists)
src/common.py           constants, annotation parsing, box -> ellipse mask, metrics
src/prepare_data.py     builds cached 320x240 arrays + ground-truth counts
src/model.py            UNet (1.94 M parameters)
src/train.py            training loop (CE + Dice loss, augmentation, best-by-val-mIoU checkpoint)
src/tune_counting.py    tunes RBC watershed parameters on the VALIDATION split
src/evaluate.py         test IoU/Dice, counts, differential, figures
src/eval_clean_subset.py supplementary metrics excluding images with missing RBC labels
src/predict.py          inference on any smear image
checkpoints/unet_best.pt trained weights
outputs/                metrics (JSON/CSV), training log, figures, demo prediction
REPORT.md / .docx       full written report
```

## Reproduce

```bash
pip install -r requirements.txt
cd src
python prepare_data.py          # ~10 s
python train.py --epochs 40     # ~28 min on 2 CPU cores; much faster on a Colab GPU
python tune_counting.py         # optional, validation split only
python evaluate.py              # writes outputs/*
python predict.py ../BCCD/JPEGImages/BloodImage_00001.jpg out.png
```

Works on a laptop CPU or free Google Colab (upload the folder, `pip install -r requirements.txt`, run the same commands).
Dataset source: https://github.com/Shenggan/BCCD_Dataset (BCCD, MIT licence).
