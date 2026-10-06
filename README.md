# Population-Aware Structural Representation Learning for MRI-Based Autism Spectrum Disorder Classification

This repository contains the implementation of a population-aware deep learning framework for Autism Spectrum Disorder (ASD) classification using T1-weighted structural magnetic resonance imaging (sMRI).

The proposed framework integrates patch-wise CNN feature extraction, attention-based subject representation, population graph learning, graph convolutional networks (GCN), and Transformer-based population-level contextual modeling.

---

## Framework Overview

The proposed pipeline is:

MRI
→ Patch Extraction
→ ResNet18
→ Attention-Based Subject Representation
→ Population Graph
→ GCN
→ Transformer
→ ASD/TD Classifier

The framework learns both subject-level structural representations and population-level relationships between subjects.

---

## Dataset

Experiments are conducted using the ABIDE-I dataset.

The study includes:

- Total participants: 1112
- ASD participants: 539
- TD participants: 573
- Acquisition sites: 17
- Imaging modality: T1-weighted structural MRI

The ABIDE-I dataset and MRI images are **not included** in this repository.

Users must obtain the dataset separately and prepare the images according to the expected structure described below.

---

## Dataset Structure

The current dataset loader expects class-specific folders:

ABIDE1/
├── ASD/
│   ├── 50002.png
│   ├── 50003.png
│   ├── 50004.png
│   └── ...
│
└── TD/
    ├── 50030.png
    ├── 50031.png
    ├── 50032.png
    └── ...
