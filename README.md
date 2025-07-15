# TwinTrack: A Twin-Level Feature Synthesis and Long-Term Coherence Framework for Multi-Object Animal Tracking

## Overview
TwinTrack is a multi-object animal tracking framework designed for challenging outdoor farm environments. It integrates a Twin-Level Contextual Feature Synthesizer (TLCFS), a Dynamic Long-Term Temporal Consistency (DLTC) module, and a Tracking Enhancement Loss (TEL) to achieve robust, long-term, and identity-consistent tracking under occlusion, lighting variation, and viewpoint changes.

**Paper:**
> Ying, R., Wang, J., Liu, C., & Nguyen, B. K. (2024). A Twin-Level Feature Synthesis and Long-Term Coherence Framework for Multi-Object Animal Tracking in Outdoor Farm Environments. *Engineering Applications of Artificial Intelligence*.

## Features
- Twin-Level Contextual Feature Synthesizer (TLCFS) with dynamic fusion
- Dynamic Long-Term Temporal Consistency (DLTC) memory module
- Tracking Enhancement Loss (TEL) for feature and trajectory consistency
- Ablation study and backbone switch (ResNet-50/101)
- Cross-dataset validation and robustness analysis
- Grad-CAM and tracking result visualization
- Reproducible experiments and evaluation metrics (MOTA, IDF1, HOTA, AssA, ISR)

## Directory Structure
```
project/
├── twintrack/
│   ├── models/
│   │   ├── tlcfs.py
│   │   ├── dltc.py
│   │   ├── backbone.py
│   │   └── __init__.py
│   ├── datasets/
│   │   ├── dct.py
│   │   ├── animaltrack.py
│   │   ├── bucktales.py
│   │   ├── harvardcow.py
│   │   └── __init__.py
│   ├── losses/
│   │   ├── tel.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   ├── logger.py
│   │   └── __init__.py
│   ├── config.py
│   ├── train.py
│   ├── test.py
│   ├── visualize.py
│   └── __init__.py
├── scripts/
│   ├── download_data.sh
│   └── preprocess.py
├── requirements.txt
├── README.md
└── figures/
```

## Installation
```bash
conda create -n twintrack python=3.9 -y
conda activate twintrack
pip install -r requirements.txt
```

## Data Preparation
- Download datasets (DCT, AnimalTrack, BuckTales, HarvardCow) as per instructions in `scripts/download_data.sh` or manually.
- Organize data as:
```
data/
  DCT/
  AnimalTrack/
  BuckTales/
  HarvardCow/
```
- Annotation format: CVAT or MOTChallenge format (see `twintrack/datasets/`).

## Training
```bash
python twintrack/train.py --config twintrack/config.py --dataset DCT --backbone resnet101
```

## Testing/Evaluation
```bash
python twintrack/test.py --config twintrack/config.py --dataset DCT --checkpoint <path_to_ckpt>
```

## Visualization
```bash
python twintrack/visualize.py --config twintrack/config.py --dataset DCT --checkpoint <path_to_ckpt> --mode gradcam
```

## Main Parameters
- Backbone: `resnet50` or `resnet101`
- Memory length, association factor, TEL lambda, batch size, learning rate, etc. (see `config.py`)
- Ablation: enable/disable TLCFS, DLTC, TEL via config

## Reproducing Experiments
- All main/ablation/cross-dataset/robustness experiments can be reproduced by adjusting `config.py` and running the above scripts.
- For Grad-CAM and failure analysis, use `visualize.py` with appropriate flags.

<!-- ## Citation
```
@article{Ying2024TwinTrack,
  title={A Twin-Level Feature Synthesis and Long-Term Coherence Framework for Multi-Object Animal Tracking in Outdoor Farm Environments},
  author={Ying, Renhui and Wang, Jinjin and Liu, Chongxiao and Nguyen, Bao Kha},
  journal={Engineering Applications of Artificial Intelligence},
  year={2024}
}
``` -->

## License
MIT License 