# Image Retrieval on MIRFLICKR-25K

A content-based image retrieval (CBIR) system built on the [MIRFLICKR-25K](http://press.liacs.nl/mirflickr/) dataset — 25,000 Flickr images with user tags and expert annotations.

Covers the full pipeline from feature extraction to approximate nearest-neighbor search, with all clustering and classification algorithms implemented from scratch.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B--32-green)

---

## Overview

| Component | What it does |
|---|---|
| **Feature Extraction** | 8 feature spaces per image — MPEG-7 hand-crafted descriptors + OpenCLIP learned embeddings + PCA/SVD hybrid fusions |
| **k-NN Retrieval** | Query-by-example across all 8 feature spaces (Euclidean for MPEG-7, cosine for CLIP) |
| **Text-to-Image Search** | Encode a text query with CLIP's text encoder, rank images by cosine similarity |
| **Dimensionality Reduction** | PCA and SVD fusion of hand-crafted + learned features (705-d → 512-d) |
| **Clustering** | DBSCAN and Spectral Clustering from scratch, evaluated with 5 quality metrics |
| **Tag Ranking** | Decision tree (entropy/info gain) and cosine-based ranking in CLIP's shared space |
| **LSH Indexing** | SimHash-based approximate nearest-neighbor search (L tables × h hyperplanes) |

## Feature Spaces

| Model | Type | Dims | Source |
|---|---|---|---|
| EdgeHistogram-150 | MPEG-7 | 150 | Pre-computed (local + semi-global + global edge histograms) |
| HomogeneousTexture-43 | MPEG-7 | 43 | Pre-computed (texture frequency) |
| OpenCLIP-Image | Learned | 512 | ViT-B-32 visual encoder |
| OpenCLIP-TagText | Learned | 512 | User tags → CLIP text encoder |
| OpenCLIP-AnnotationText | Learned | 512 | Expert annotations → CLIP text encoder |
| OpenCLIP-Fused | Learned | 512 | Weighted avg (0.5 image + 0.25 tag + 0.25 annotation) |
| PCA-Fused | Hybrid | 512 | Concatenated 705-d → PCA reduction |
| SVD-Fused | Hybrid | 512 | Concatenated 705-d → SVD reduction |

## Results Highlights

- **Inherent dimensionality analysis** shows MPEG-7 spaces carry heavy redundancy (150-d Edge Histogram → ~33 effective dims), while CLIP embeddings use their space more efficiently.
- **OpenCLIP-Fused** consistently retrieves semantically relevant results across query-by-example and text-to-image search.
- **PCA/SVD fused** representations capture complementary signals from both hand-crafted and learned features.
- **LSH** with 10 tables / 8-bit keys reduces the candidate set by ~85% with minimal retrieval quality loss.

## Project Structure

```
├── bdm_project.ipynb          # Main notebook (all code + results)
├── data/
│   ├── mirflickr/             # MIRFLICKR-25K images + metadata
│   │   └── meta/tags/         # User tags per image
│   ├── mirflickr25k_annotations_v080/  # Expert annotations
│   ├── features_edgehistogram/         # Pre-computed MPEG-7
│   └── features_homogeneoustexture/    # Pre-computed MPEG-7
├── features/                  # Extracted feature matrices (.npy) + metadata
└── README.md
```

## Setup

### Requirements

```bash
pip install numpy matplotlib pillow torch open_clip_torch scikit-learn
```

### Data

1. Download the [MIRFLICKR-25K](http://press.liacs.nl/mirflickr/) dataset and extract into `data/mirflickr/`
2. Download the [annotations](http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip) and extract into `data/mirflickr25k_annotations_v080/`
3. The MPEG-7 descriptors should be in `data/features_edgehistogram/` and `data/features_homogeneoustexture/`

### Run

```bash
jupyter notebook bdm_project.ipynb
```

Feature extraction (Section 2) takes ~20 minutes on CPU. Pre-extracted `.npy` files in `features/` let you skip directly to retrieval and clustering.

## Built With

- **PyTorch** + **OpenCLIP** — ViT-B-32 (laion2b_s34b_b79k checkpoint)
- **NumPy** — PCA, SVD, distance computations, all clustering/hashing from scratch
- **Matplotlib** — visualization
- **scikit-learn** — t-SNE for cluster visualization only

## License

MIT
