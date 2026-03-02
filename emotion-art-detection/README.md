# Plutchik-48: A Geometrically Coherent RGB Manifold for Affective Feature Extraction

> **This repository is directly associated with the manuscript submitted to 
> The Visual Computer (Springer Nature).**
> If you use this dataset or code in your research, please cite our manuscript 
> (citation details below).

---

## Overview
The Plutchik-48 dataset is a deterministic feature space that maps 48 distinct 
RGB color points to a hierarchical taxonomy of emotions rooted in Robert Plutchik's 
Psycho-Evolutionary Theory. This repository contains the complete dataset, 
derivation pipeline, and experimental validation scripts.

---

## Repository Structure
```
Emotion-Detection-Through-Colors-Analysis/
│
├── emotion-art-detection/
│   ├── data/
│   │   └── labels.csv          # The Plutchik-48 dataset (48 RGB-emotion mappings)
│   ├── scripts/
│   │   ├── derivation.py       # Hybrid Derivation Strategy pipeline
│   │   ├── kmeans_clustering.py# K-Means feature extraction
│   │   └── validation.py       # WEA metric computation
│   └── figures/
│       ├── primary_space.png   # Fig 3: Primary feature space
│       └── dyadic_space.png    # Fig 4: Dyadic combinations space
│
└── README.md
```

---

## Requirements & Dependencies

Install all dependencies using:
```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| Python | 3.9+ |
| numpy | 1.23.0 |
| pandas | 1.5.0 |
| scikit-learn | 1.2.0 |
| matplotlib | 3.6.0 |
| tensorflow | 2.11.0 |
| scipy | 1.9.0 |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AmudhanManimaran/Emotion-Detection-Through-Colors-Analysis.git
cd Emotion-Detection-Through-Colors-Analysis
```

### 2. Load the Plutchik-48 Dataset
```python
import pandas as pd
df = pd.read_csv('emotion-art-detection/data/labels.csv')
print(df.head())
```

### 3. Run K-Means Emotion Detection
```bash
python emotion-art-detection/scripts/kmeans_clustering.py --input your_image.jpg
```

### 4. Compute WEA Score for an Artwork
```bash
python emotion-art-detection/scripts/validation.py --input your_image.jpg
```

---

## Dataset Format (labels.csv)

| Column | Type | Description |
|---|---|---|
| R | Integer (0–255) | Red channel value |
| G | Integer (0–255) | Green channel value |
| B | Integer (0–255) | Blue channel value |
| Label | String | Emotion name (e.g., Joy, Rage) |
| Category | String | Primary / Intensity / Dyad |
| Hex | String | Hex code (e.g., #FFFF00) |

---

## How to Cite

If you use the Plutchik-48 dataset or this code, please cite:
```
Manimaran, A., & Lingaswamy, S. (2026). Plutchik-48: A Geometrically Coherent 
RGB Manifold for Affective Feature Extraction. 
The Visual Computer. [Submitted]
DOI: [will be updated upon acceptance]
```

**Zenodo DOI badge:** [![DOI](https://zenodo.org/badge/DOI/XXXXX.svg)](https://doi.org/XXXXX)
*(Update this after completing Step 4 below)*

---

## License
This dataset and code are released under the **MIT License**.

---

## Contact
- Amudhan Manimaran — amudhanmanimaran.am@gmail.com
- Dr. Sindhia Lingaswamy — sindhia@nitt.edu
