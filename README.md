# Spatial-Contrastive-Learning (업데이트 중 2024.11.05)
Multi-modal CrossViT using 3D Spatial Information for Visual Localization (SCIE) / Published: 18 Oct 2024  
[[Paper](https://drive.google.com/file/d/16deTO1LvQE-eh0E4dOQJt9njEz26IRIu/view?usp=sharing)] [[Online](https://link.springer.com/article/10.1007/s11042-024-20382-w?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20241018&utm_content=10.1007%2Fs11042-024-20382-w)]  

# Pipeline Instructions
- 1~7. 데이터 전처리
- 8, 10. Train
- 11. Inference for global locaization (image retrieval)
- 12. Camera pose estimation

1. pipeline_sfm_visuallocalization.ipynb
- dataset의 structure-from-motion을 수행하여 3D Point를 생성합니다. 
- 기존 NetVLAD가 적용된 Visual Localization을 수행하실 수 있습니다. 


**순차적으로 실행하시면 됩니다.**  

2. Database preparation:
```bash
python prepare_database_3column.py
```
- Output : DataBase3Column.pkl
- Positive image pair 간 3D Bounding box의 IoU 값을 계산

3. CSV preparation:
```bash
python prepare_database_csv.py
```

4. Point cloud processing:
```bash
python point_cloud_iou_processor.py
```
- Output : output_iou_database.pkl  

5. Feature distance computation:
```bash
python compute_feature_distances.py
```
- Output: DataBase_Norm_Tiny.pickle  

6. Rotary embeddings generation:
```bash
python generate_RoPE_embeddings.ipynb    
```
- Output: Large_Patch_14x14_RT_RoPE_Tensor.pickle

7. Validation data preparation:
```bash
python prepare_validation_data.py
```

8. Teacher model training:
```bash
python train_multimodal_crossvit_teacher.py
```

9. Query validation preparation:
```bash
python prepare_validation_query.py
```
- Output : ValidationAll_Teacher_Key_Real.pickle 

10. Student model training:
```bash
python train_knowledge_distillation_student.py
```

11. Localization pairs generation:
```bash
python generate_localization_pairs.py
```

**최종 테스트 파일**
12. final_pipeline.ipynb
- 11번에서 생성한 'Retrieved_Images'를 적용하며 Visual Localization을 수행합니다. 
- Output : Pose_Estimation_Results
- Benchmark : https://www.visuallocalization.net/

---


### 국문 요약:  
1. 게재처(학회/학술지):    
    - Multimedia Tools and Materials (SCIE)    
    - Accepted: 8 Oct 2024
    - Published: 18 Oct 2024 

2. 저자구분: 1 저자      

3. 논문 주요내용: 대규모 환경의 계층적 Visual Localization 단계에서 Global Localization을 위한 새로운 Image Retrieval 모델 제안   
- Multi-modal CrossViT (Vision Transformer) 모델 :    
    - 이미지 특징과 3D 공간 정보를 모두 활용하여 더 견고한 Global descriptor 생성    
    - 두 개의 별도 브랜치를 통해 이미지 패치와 3D 공간 정보를 처리       
- Spatial Contrastive Learning:        
    - Anchor 이미지에 대한 Positive 및 Negative 이미지 셋을 공유된 3D 포인트 존재 여부에 따라 지정   
    - 3D 바운딩 박스 간 IoU(Intersection over Union)를 유사성 측정에 활용        
- Knowledge Distillation 접근법 적용:       
    - Multi-modal CrossViT(Teacher)의 임베딩 능력을 단일 이미지 입력 모델(Student)로 전달  
    - 실제 응용단계에서 제약 조건을 고려한 효율적인 추론 모델 구현      
- 성능 평가 및 분석:    
    - 기존 모델 대비 파라미터 21.6배, FLOPs 58.9배 감소     
    - 이미지 검색 정확도(Precision@K) 5-fold cross validation 결과 대부분에서 성능 향상     
    - 일부 벤치마크에서 카메라 포즈 정확도(meter, degree) 향상   

4. 결론: 이를 통해 모델을 크게 줄이면서도 이미지 검색 정확도를 향상해 SCIE 급 저널에 주저자로 제출했습니다. 질의 이미지와 동일 공간의 참조 이미지를 빠르고 정확하게 검색하면, 기하학적 매칭 연산을 통해 질의 이미지의 정확한 카메라 포즈를 추정할 수 있어 자율주행, 로봇 분야에서 파급 효과를 볼 수 있습니다.  

### 선행 연구:  
• 석사 학위 논문 (2023.12): [영상 위치 파악을 위한 공간적 CrossViT 기반 다중 모달 메트릭 학습](https://dcollection.cau.ac.kr/srch/srchDetail/000000241188?treePageNum=1&navigationSize=10&orgYn=all&thesisDegree=all&pageSize=10&ajax=false&searchText=%5B%EC%A0%84%EC%B2%B4%3AVisual+localization%5D&agreeYn=all&sortField=score&sortDir=desc&searchOption=km&searchOperator2=%2B&searchOperator3=%2B&searchWhere1=all&searchWhere2=all&insCode=211052&searchWhere3=all&searchKeyWord1=Visual+localization&query=%2B%28%28all%3Avisual%2Blocalization%29%29&itemTypeCode=all&start=0&searthTotalPage=0&rows=10&pageNum=1&searchTotalCount=0)  
• SCIE 논문 2 저자 (2023.03 출판): [Clustering Reference Images Based on Covisibility for Visual Localization](https://www.techscience.com/cmc/v75n2/52040)   

---
# Purpose
<p align="center">
  <img src="https://github.com/user-attachments/assets/27cb6dda-0b9b-4b9e-a2d3-193918806657" alt="Purpose">
</p>
- The proposed approach is capable of generating distinct embeddings for scenes that are physically distant from each other even when they have similar visual features. Here, the embeddings of an anchor image (black outline) and its corresponding positive image (red outline) are close to each other in the embedding space but that of the negative image (blue outline) is farther away.   

# Contribution
●	A multi-modal CrossViT architecture, capable of generating effective global representations from 2D images and 3D spatial information as part of a hierarchical framework.  
●	A RoPE-based 3D spatial information encoding approach for the multi-modal CrossViT architecture.  
●	A novel spatial contrastive learning strategy, in which positive and negative image sets for each anchor image are designated based on the presence of shared 3D points as well as a quantitative measure of similarity based on the IoU between 3D bounding boxes that are generated from a pair of images.  
●	A knowledge distillation approach to transfer the embedding capacity of the proposed multi-modal CrossViT architecture onto an architecture that takes a single image as input. This bridges the gap between the rich multi-modal representations and the practical constraints of real-world applications.  
●	A comprehensive analysis of the performance of various global and local matching technique combinations in a hierarchical pipeline, as well as the effects of hard negative sampling and data augmentation techniques such as day-to-night stylization on large-scale datasets.  


# Pipe-Line
<p align="center">
  <img src="https://github.com/user-attachments/assets/dfbdde7a-63c7-44af-a12e-01ed88ac0269" alt="Pipe-Line">
</p>
- Architectures of the proposed method in training for global localization (a) and inference for visual localization (b).   

# Architecture
<p align="center">
  <img src="https://github.com/user-attachments/assets/f160ace0-70f0-4b27-b52a-6ece1ea4c80a" alt="Architecture">
</p>
[Left] : Architecture of proposed multi-modal CrossViT model, an extension of the CrossViT architecture.        
[Right] :Transferring teacher model representational capabilities to student model. 

# Data Augmentation
<p align="center">
  <img src="https://github.com/user-attachments/assets/bc47f81c-5d3a-4530-95e8-1c322090ab88" alt="Data Augmentation">
</p>
- CycleGAN-based style transfer model [58, 59] was used to generate synthetic nighttime images including Random Cropping, Random Color Distortions, Random Gaussian Blur, etc.  

# Loss Function
### Training Teacher Model
<p align="center">
  <img src="https://github.com/user-attachments/assets/4bed04f1-9009-44e2-aaa5-491fb02d957d" alt="Loss Function 1">
</p> 
- A pair of images captured from the same scene and 3D bounding boxes generated from their 3D points.    
 
ㅤ  
<p align="center">
  <img src="https://github.com/user-attachments/assets/b8b3e578-1cfc-4770-8372-6f54f2292be8" alt="Loss Function 2">
</p>
- The IoU values across all positive image pairs in the dataset are used to normalize IoU values   
ㅤ  
ㅤ  
<p align="center">
  <img src="https://github.com/user-attachments/assets/2564f1c2-f3dc-4e03-8d9c-b35522ad1753" alt="Loss Function 3">
</p>
- d_(i,j) represents the Euclidean distance between embedding vectors generated by the proposed multi-modal CrossViT model. w_(i,j) is the degree of proximity between positive image pairs that is expressed as a continuous value ranging from 0 (negative) to 1 (same image). Optimizing this loss results in more similar embeddings for images captured in closer proximity to each other. 
ㅤ
    
### Knowledge Distillation for Training Student Model  
<p align="center">
  <img src="https://github.com/user-attachments/assets/8bb146a3-6df3-4924-8494-d6ec2d8f6fbc" alt="Knowledge Distillation 1">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/438aacd5-6aa9-4a44-959c-9b0a51314364" alt="Knowledge Distillation 2">
</p>
- The student model is trained to have an embedding space similar to that of the teacher model. Since the loss function ensures that the representations generated from an anchor and negative image pair are even farther away from each other in the embedding space, the student model generates more robust representations than the teacher model. 
ㅤ    

# Results
<p align="center">
  <img src="https://github.com/user-attachments/assets/c812ee5e-e2d9-4f52-9d9f-72e3ae83fb6a" alt="Result 1">
</p>  

[Left]: Query image    
[Right]: Retrieved reference images   

<p align="center">
  <img src="https://github.com/user-attachments/assets/e265257c-2a91-4fdf-bccf-10d9669be934" alt="Result 2">
</p>
- 10 anchor images (a) and t-SNE plots of their global descriptors generated using: NetVLAD (b), the proposed multi-modal CrossViT teacher model (c) and the student model (d)  

<p align="center">
  <img src="https://github.com/user-attachments/assets/6129cdb5-a463-4b1e-8690-dabaeb973255" alt="Result 3">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/59efa180-5a93-413f-b525-a2809230e60e" alt="Result 4">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/a248943b-32db-4a09-8750-c8c96c78f3a3" alt="Result 5">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/631c769f-2184-482e-b89b-71922e57e413" alt="Result 6">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/165cce51-7157-45c8-b7ee-b55a21e1d8a5" alt="Result 7">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f923f384-b096-4e6d-9158-7f5cb7100453" alt="Result 8">
</p>

-----
# Conclusion
In order to facilitate more robust camera pose estimation, this study proposes a multi-modal CrossViT that leverages both image features and 3D spatial information in generating robust descriptors in the global localization stage of hierarchical visual localization frameworks. Prior to training, the positive and negative image sets for each anchor image are designated based on the presence of shared 3D points. The proposed teacher model was trained using a spatial contrastive loss in which the degree of proximity between anchor and positive images is used as a quantitative measure of similarity. In order to transfer the embedding capacity of this multi-modal architecture to an inference model that only requires a single image input with no 3D spatial information, a Gaussian kernel-based knowledge distillation approach was used. The proposed approach was evaluated on the large-scale Aachen Day-Night and Aachen Day-Night_v1.1 datasets. Experimental results show that the proposed approach requires significantly less memory and processing power than previous state-of-the-art approaches but achieves comparable performance. More specifically, it achieves accuracies of 87.3%, 95.0%, and 97.6% for daytime cases and 87.8%, 89.8%, and 95.9% for nighttime cases under thresholds of “0.25m, 2°”, “0.5m, 5°” and “5m, 10°” respectively with 58.9 times fewer FLOPs and 21.6 times fewer parameters than the NetVLAD model. As highlighted in Figures 10-12, the proposed approach achieves robust representations containing visual and 3D spatial information across a range of scenes. However, it is unable to generate sufficiently robust embeddings for scenes that occur only once or twice in the imbalanced training dataset. In order to address this, we intend to constitute and pre-train with significantly larger datasets that are suitable for SfM reconstruction. Weighting coefficients are to be used to further emphasize the impact of anchor images with smaller positive sets during training. Hard negative sampling techniques that consider both spatial proximity and visual similarity are also to be leveraged in improving recall performance.

