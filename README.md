# Multi-modal CrossViT for Visual Localization
[![Paper](https://img.shields.io/badge/Paper-Springer-blue)](https://doi.org/10.1007/s11042-024-20382-w)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![arXiv](https://img.shields.io/badge/MTAP-2024-b31b1b.svg)](https://link.springer.com/article/10.1007/s11042-024-20382-w)

Official implementation of "Multi-modal CrossViT using 3D spatial information for visual localization" (Multimedia Tools and Applications, 2024) by Junekoo Kang, Mark Mpabulungi, and Hyunki Hong. | [[Paper](https://drive.google.com/file/d/16deTO1LvQE-eh0E4dOQJt9njEz26IRIu/view?usp=sharing)] | [[Online](https://link.springer.com/article/10.1007/s11042-024-20382-w?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20241018&utm_content=10.1007%2Fs11042-024-20382-w)] |   

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Pipeline Steps](#pipeline-steps)
- [Evaluation](#evaluation)
- [Performance Highlights](#performance-highlights)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview
This research introduces a state-of-the-art hierarchical framework for visual localization leveraging a multi-modal CrossViT architecture. Our approach uniquely combines image features with 3D spatial information, achieving superior performance with significantly reduced computational requirements.

### Key Innovations
- **Multi-modal Architecture**: Dual-branch CrossViT integrating visual and 3D spatial information
- **RoPE-based Encoding**: Advanced 3D spatial information encoding using Rotary Position Embedding
- **Spatial Contrastive Learning**: Novel training strategy utilizing shared 3D points and IoU-based similarity metrics
- **Knowledge Distillation**: Optimized inference through teacher-student model transfer
- **Computational Efficiency**: Achieves 58.9× fewer FLOPs and 21.6× fewer parameters compared to NetVLAD

## Architecture

### Training Architecture
![Fig 2 (a)](https://github.com/user-attachments/assets/7d9881c4-f7a9-496e-be1c-f54928ca426e)  

The training pipeline implements a sophisticated dual-branch architecture:
- **Image Branch**: Advanced visual feature processing using CrossViT
- **Spatial Branch**: Specialized 3D information handling with RoPE encoding

### Inference Architecture
![Fig 2 (b)](https://github.com/user-attachments/assets/b42417f3-ee4e-43ce-9b69-565312f3b1a2)  

Our inference model employs a lightweight student architecture maintaining spatial awareness through knowledge distillation, optimized for real-world deployment.

## Requirements

### Environment Setup
```bash
conda create -n spatial_contrastive_learning python=3.8
conda activate spatial_contrastive_learning
pip install -r requirements.txt
```

### Core Dependencies
```txt
pytorch>=1.7.0
hloc
opencv-python
numpy
pandas
scikit-learn
tensorboard
tqdm
```

## Project Structure
```
├── datasets/
│   └── aachen/              # Aachen Day-Night dataset
├── models/
│   ├── crossvit_official.py                    # Base CrossViT
│   └── crossvit_PE_RT_official_MultiModal.py   # Multi-modal CrossViT
├── DataBase/               # Preprocessed data
├── pipeline_sfm_visuallocalization.ipynb  # SfM pipeline
├── preprocessing.py        # Point cloud processing
├── generate_RoPE_embeddings.ipynb  # RoPE encoding
├── train_multimodal_crossvit_teacher.py  # Teacher model training
├── train_knowledge_distillation_student.py  # Student model training
├── generate_localization_pairs.py  # Pair generation
└── final_pipeline.ipynb    # End-to-end pipeline
```

## Pipeline Steps

### 1. Data Preprocessing
```bash
# SfM and preprocessing
jupyter notebook pipeline_sfm_visuallocalization.ipynb
python preprocessing.py

# RoPE embeddings generation
jupyter notebook generate_RoPE_embeddings.ipynb
```

### 2. Model Training

#### Teacher Model Training
```bash
python train_multimodal_crossvit_teacher.py
```
Implements:
- Dual-branch architecture
- Spatial contrastive learning
- Hard negative sampling

#### Student Model Training
```bash
python train_knowledge_distillation_student.py
```
Features:
- Gaussian kernel-based embedding transfer
- Single-image inference optimization
- Spatial awareness preservation

### 3. Visual Localization Pipeline

#### Reference Pair Generation
```bash
python generate_localization_pairs.py
```

#### Pose Estimation
```bash
jupyter notebook final_pipeline.ipynb
```
Pipeline components:
1. Student model image retrieval
2. Local feature matching
3. PnP-RANSAC pose estimation

## Performance Highlights

### Camera Pose Estimation Accuracy
| Condition | (0.25m, 2°) | (0.5m, 5°) | (5m, 10°) |
|-----------|-------------|------------|------------|
| Daytime   | 87.3%       | 95.0%      | 97.6%      |
| Nighttime | 87.8%       | 89.8%      | 95.9%      |

### Image Retrieval Performance (P@K)
| Models | P@200 | P@150 | P@100 | P@50 | P@20 | P@5 | P@1 |
|--------|--------|--------|--------|-------|-------|------|------|
| Ours | **0.8209** | **0.7727** | **0.7206** | **0.6889** | **0.7383** | **0.8368** | 0.8976 |
| NetVLAD | 0.4611 | 0.4427 | 0.4257 | 0.4529 | 0.5980 | 0.8219 | **0.9425** |

### Computational Efficiency
| Models | FLOPs (GB) | Parameters (MB) |
|--------|------------|-----------------|
| NetVLAD | 94.3 | 148.9 |
| AP-GEM | 86.2 | 105.3 |
| CRN | 94.3 | 148.9 |
| SARE | 94.3 | 148.9 |
| HAF | 1791.2 | 158.8 |
| Patch-NetVLAD | 94.2 | 148.7 |
| **Ours** | **1.6** | **6.9** |

## Citation
```bibtex
@article{kang2024multi,
  title={Multi-modal {CrossViT} using {3D} spatial information for visual localization},
  author={Kang, Junekoo and Mpabulungi, Mark and Hong, Hyunki},
  journal={Multimedia Tools and Applications},
  year={2024},
  publisher={Springer},
  doi={10.1007/s11042-024-20382-w}
}
```

## Acknowledgments
- HLOC for the hierarchical localization framework
- CrossViT for the transformer architecture
- Aachen Day-Night dataset for evaluation

## Contact
For questions or issues:
- Junekoo Kang (engineerjkk@naver.com) or (engineerjkk@cau.ac.kr)

## Resources
- Google Drive: [download link](https://drive.google.com/drive/folders/xxx)
