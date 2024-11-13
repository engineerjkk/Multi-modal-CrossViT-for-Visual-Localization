(업데이트 중 2024.11.07)
# Spatial Contrastive Learning
[![Paper](https://img.shields.io/badge/Paper-Springer-blue)](https://doi.org/10.1007/s11042-024-20382-w)  
This is the code for **Multi-modal CrossViT using 3D spatial information for visual localization** by Junekoo Kang, Mark Mpabulungi & Hyunki Hong. | Published: 18 Oct 2024 | SCIE | [[Paper](https://drive.google.com/file/d/16deTO1LvQE-eh0E4dOQJt9njEz26IRIu/view?usp=sharing)] | [[Online](https://link.springer.com/article/10.1007/s11042-024-20382-w?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20241018&utm_content=10.1007%2Fs11042-024-20382-w)] |   

![Fig 2 (a)](https://github.com/user-attachments/assets/7d9881c4-f7a9-496e-be1c-f54928ca426e)  
<p align="center">
  Architectures of the proposed method in training for global localization  
</p>

![Fig 2 (b)](https://github.com/user-attachments/assets/b42417f3-ee4e-43ce-9b69-565312f3b1a2)  
<p align="center">
  Inference for visual localization  
</p>

This repository provides the official implementation of "Multi-modal CrossViT using 3D spatial information for visual localization" (Multimedia Tools and Applications, 2024). The proposed approach leverages both image features and 3D spatial information through a novel multi-modal architecture for accurate camera pose estimation.

## Key Features
- Multi-modal CrossViT architecture leveraging 2D images and 3D spatial information
- RoPE-based 3D spatial information encoding
- Novel spatial contrastive learning strategy using shared 3D points
- Knowledge distillation from teacher to student model for efficient inference
- Significant reduction in computational requirements (58.9× fewer FLOPs, 21.6× fewer parameters than NetVLAD)

## Performance Highlights

### Camera Pose Estimation Accuracy
- Daytime accuracy: 87.3% (0.25m, 2°), 95.0% (0.5m, 5°), 97.6% (5m, 10°)
- Nighttime accuracy: 87.8% (0.25m, 2°), 89.8% (0.5m, 5°), 95.9% (5m, 10°)

### Image Retrieval Performance (P@K)
| Models | P@200 | P@150 | P@100 | P@50 | P@20 | P@5 | P@1 |
|--------|--------|--------|--------|-------|-------|------|------|
| Ours | **0.8209** | **0.7727** | **0.7206** | **0.6889** | **0.7383** | **0.8368** | 0.8976 |
| NetVLAD | 0.4611 | 0.4427 | 0.4257 | 0.4529 | 0.5980 | 0.8219 | **0.9425** |

### Computational Efficiency Comparison
| Models | FLOPs (GB) | Parameters (MB) |
|--------|------------|-----------------|
| NetVLAD | 94.3 | 148.9 |
| AP-GEM | 86.2 | 105.3 |
| CRN | 94.3 | 148.9 |
| SARE | 94.3 | 148.9 |
| HAF | 1791.2 | 158.8 |
| Patch-NetVLAD | 94.2 | 148.7 |
| **Ours** | **1.6** | **6.9** |

Key advantages of our approach:
- Significantly better precision at higher K values (P@200 to P@20)
- 58.9× fewer FLOPs than NetVLAD (1.6 GB vs 94.3 GB)
- 21.6× fewer parameters than NetVLAD (6.9 MB vs 148.9 MB)
- Competitive performance with state-of-the-art methods while maintaining lower computational requirements  

## Requirements

### Environment Setup
```bash
conda create -n spatial_contrastive_learning python=3.8
conda activate spatial_contrastive_learning
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 1.7.0
- HLOC
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Tensorboard
- tqdm

## Project Structure
```
├── datasets/
│   └── aachen/              # Aachen Day-Night dataset
├── models/
│   ├── crossvit_official.py
│   └── crossvit_PE_RT_official_MultiModal.py
├── preprocessing.py         # Data preprocessing and 3D point cloud generation
├── train_multimodal_crossvit_teacher.py  # Teacher model training
├── train_knowledge_distillation_student.py  # Student model training
├── generate_localization_pairs.py  # Image retrieval pair generation
└── final_pipeline.ipynb    # End-to-end visual localization pipeline
```

## Pipeline Steps

### 1. Preprocessing
```bash
# Run SfM and preprocessing
jupyter notebook pipeline_sfm_visuallocalization.ipynb
python preprocessing.py

# Generate RoPE embeddings
jupyter notebook generate_RoPE_embeddings.ipynb
```

The preprocessing step includes:
- Structure-from-Motion (SfM) for 3D point cloud generation
- Point cloud processing for spatial information
- Feature extraction and database preparation

### 2. Model Training

#### Teacher Model
```bash
python train_multimodal_crossvit_teacher.py
```
Trains the multi-modal CrossViT model using:
- Dual-branch architecture for image and 3D spatial features
- Spatial contrastive learning with IoU-based similarity
- Hard negative sampling strategy

#### Student Model
```bash
python train_knowledge_distillation_student.py
```
Implements knowledge distillation with:
- Gaussian kernel-based embedding transfer
- Single-image input for efficient inference
- Preservation of spatial awareness

### 3. Visual Localization

#### Generate Image Pairs
```bash
python generate_localization_pairs.py
```
Creates pairs for image retrieval using learned embeddings.

#### Camera Pose Estimation
```bash
jupyter notebook final_pipeline.ipynb
```
Performs visual localization:
- Image retrieval using student model
- Local feature matching
- PnP-RANSAC pose estimation

## Evaluation
Evaluate on the [Visual Localization Benchmark](https://www.visuallocalization.net/):
1. Generate pose estimates using the pipeline
2. Submit `Pose_Estimation_Results.txt` to the benchmark

## Citation
If you find this work useful, please cite our paper:
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
For questions or issues, please open an issue or contact the authors:
- Junekoo Kang (engineerjkk@naver.com) or (engineerjkk@cau.ac.kr)


