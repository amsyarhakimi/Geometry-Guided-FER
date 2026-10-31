# Geometry-Guided Generalizable FER

**Anatomical Attention Aligns Mechanistically but Improves Accuracy Conditionally**

Course project for **COSE474 Deep Learning**, Korea University (Spring 2026) — Team 6.
Team: Jin Ho Kwak · **Amsyar Hakimi** · Chew Glenn

> Forked from the team repo [`jinskfam5-gif/20261RCOSE47400_TEAM6`](https://github.com/jinskfam5-gif/20261RCOSE47400_TEAM6).
> Full report: [`report/Geometry_Guided_Generalizable_FER_TEAM6.pdf`](report/Geometry_Guided_Generalizable_FER_TEAM6.pdf)

---

## TL;DR

- We study **zero-shot cross-dataset facial expression recognition (FER)**: train on one dataset, test on unseen ones.
- We start from **CAFE** (Zhang et al., ECCV 2024) and add one thing: a **fixed facial-anatomy prior**.
- 68 facial landmarks → **FACS codebook** → per-expression **Action Unit (AU) heatmaps** → used to gate CAFE's channel groups.
- **Mechanism works:** AU specialization gap rises from ~0 to **0.183–0.188**. Activations become more localized.
- **Accuracy gain is small:** FACS attention alone gives **+0.21 pp**. Augmentation gives **+3.11 pp** on all 5 sources.
- **Injection site matters:** a per-channel variant drops cross-domain accuracy by **~7 pp**.

---

## Method

| Component | Description |
|---|---|
| Backbone | Frozen **CLIP** visual encoder (512-d face feature) + **ResNet-18** FER branch (MS-Celeb weights) |
| CAFE mask | Sigmoid channel mask over 512 channels, split into 7 expression groups |
| Loss | `L = L_cls + 1.5·L_sep + 5.0·L_div` (original CAFE) |
| Landmarks | 68 points via `face_alignment` (Bulat & Tzimiropoulos) |
| AU prior | FACS codebook: expression → active AUs → landmark regions → 7×7 Gaussian heatmap `H_k` |
| Per-group gate | `F̃_Gk = F_Gk ⊙ H_k` for each expression group `k` (no learnable parameters) |
| Fallback | Uniform map for neutral class and failed detections (~8% of samples) |
| Augmentation | Albumentations: flip, rotation, crop, color jitter, Gaussian noise. Landmarks re-detected after augmentation |

**Mechanism metric (AU gap):** `gap = mean(diag(M)) − mean(offdiag(M))`, where `M[i,j]` is cosine similarity between group *i*'s activation and AU heatmap *j*.

---

## Results

Source-excluded cross-domain accuracy (avg. over 4 unseen targets). Gains in percentage points. Single seed.

| Source | Base | +Aug Δ | +FA Δ | +FA Δ (w/ Aug) | AU gap (FA) | AU gap (FA+Aug) |
|---|---|---|---|---|---|---|
| RAF-DB | 52.44 | +4.84 | +1.02 | −0.61 | 0.228 | 0.239 |
| FERPlus | 51.72 | +2.74 | −3.26 | +3.95 | 0.215 | 0.227 |
| AffectNet | 35.36 | +3.25 | +2.00 | +2.05 | 0.195 | 0.191 |
| MMA | 61.19 | +1.10 | −0.47 | −0.52 | 0.202 | 0.197 |
| SFEW | 32.48 | +3.63 | +1.77 | −0.71 | 0.077 | 0.086 |
| **Mean** | **46.64** | **+3.11** | **+0.21** | **+0.83** | **0.183** | **0.188** |

**Negative control — per-channel AU weighting** (separate run, compare within table):

| Setting | In-domain | Cross-domain | Δ |
|---|---|---|---|
| Baseline | 70.1 | 46.1 | – |
| +Aug | 70.6 | 50.7 | +4.6 |
| +ChannelW FA | 65.7 | 39.1 | −7.0 vs base |
| +ChannelW FA + Aug | 66.0 | 43.0 | −7.7 vs aug |

**By transfer difficulty (no aug):** FA helps hard transfers (+1.65 pp) and medium (+1.42 pp), not easy ones (−0.10 pp).

---

## Repository Structure

```
.
├── README.md
├── report/
│   ├── Geometry_Guided_Generalizable_FER_TEAM6.pdf   # final report
│   ├── midterm_presentation.pdf
│   └── final_presentation.pptx
├── notebooks/
│   ├── 01_baseline/
│   │   └── CAFE_backbone.ipynb               # CAFE baseline + unified label maps
│   ├── 02_augmentation/
│   │   └── CAFE_with_augmentation.ipynb      # Albumentations pipeline
│   ├── 03_au_weighted_lsep/
│   │   ├── Landmark_Weighted_LossSep_final.ipynb   # AU codebook + heatmap builder + AU-weighted L_sep
│   │   └── Augmentation+Landmark_Weighted_LossSep_face_alignment.ipynb
│   └── 04_ablations/
│       ├── Landmark_per_emotions_Ablation.ipynb   # MAIN: per-group FA, 5×5 cross-dataset (Table 1, 3)
│       ├── landmark_per_emotions_ablation.py
│       └── Landmark_per_channel_Ablation.ipynb    # negative control (Table 2)
└── src/
    └── cafe_backbone_AUmask.py                     # script version of early AU-mask model
```

---

## How to Run

All experiments were run on **Google Colab (GPU)**.

1. Open a notebook in Colab (e.g. `notebooks/04_ablations/Landmark_per_emotions_Ablation.ipynb`).
2. Upload `resnet18_msceleb.pth` to a `deeplearning` folder in Google Drive, or set the path in **Section 3**.
3. Datasets download automatically with `kagglehub`:
   - RAF-DB `shuvoalok/raf-db-dataset`
   - FERPlus `arnabkumarroy02/ferplus`
   - AffectNet `mstjebashazida/affectnet`
   - MMA `mahmoudima/mma-facial-expression`
   - SFEW `vlntnstarodub/datasetsfew`
4. Run the **Sanity Check** cell first. Then run training.
5. Default settings: `BATCH_SIZE = 64`, `EPOCHS = 20`, seed `3407`.

Main dependencies: `torch`, `torchvision`, `openai/CLIP`, `face_alignment`, `albumentations`, `opencv-python`, `scikit-learn`, `kagglehub`.

---

## My Contribution (Amsyar Hakimi)

> Workflow note: the team wrote and ran code in **Google Colab**, then uploaded finished notebooks to GitHub. So commit count does not reflect who wrote what.

- Built **`Landmark_Weighted_LossSep_final.ipynb`** — the first landmark/AU notebook in the project:
  - FACS **AU codebook** + landmark → 7×7 AU heatmap builder
  - **Per-image AU heatmap builder**
  - **512-channel cosine-similarity relevance scorer**
  - **AU-weighted `L_sep` supervisor**
- These AU components were reused by the later face_alignment, augmentation, and ablation notebooks.
- Co-authored the final report.

Evidence in the original team repo:
- [`c50af70`](https://github.com/jinskfam5-gif/20261RCOSE47400_TEAM6/commit/c50af701864b4829240ddf54eb40ce3e80becf9f) — added `Landmark_Weighted_LossSep_final.ipynb` (2026-05-29)
- [`ad38ef5`](https://github.com/jinskfam5-gif/20261RCOSE47400_TEAM6/commit/ad38ef5277e4f33a2d08e49c10452bc183d81e2c) — fixed per-image AU heatmap builder (2026-05-29)

**Teammates:** Chew Glenn — baseline CAFE setup and data augmentation. Jin Ho Kwak — AU-mask experiments, face_alignment version, ablation pipelines, report.

---

## References

- Y. Zhang et al. *Generalizable Facial Expression Recognition.* ECCV 2024. (CAFE)
- A. Radford et al. *CLIP.* ICML 2021.
- A. Bulat & G. Tzimiropoulos. *How far are we from solving 2D & 3D face alignment?* ICCV 2017.
- P. Ekman & W. Friesen. *Facial Action Coding System.* 1978.
- S. Belharbi et al. *Guided interpretable FER via spatial AU cues.* IEEE FG 2024.
- A. Buslaev et al. *Albumentations.* Information, 2020.

## License & Credit

Academic course project. CAFE model code is adapted from the original authors (Zhang et al., ECCV 2024). Datasets belong to their owners and are not redistributed here.
