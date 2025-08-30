# Carvana Image Masking Challenge

<img src="https://www.nerdwallet.com/assets/blog/wp-content/uploads/2021/11/carvana-logo-vector.png" alt="Carvana Logo" width="400"/>

Welcome to the Carvana Image Masking Challenge repository. This project focuses on **semantic segmentation** of cars as part of the [Carvana Image Masking Challenge on Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge). The goal is to generate **precise masks** for cars in images.

---

## Table of Contents
- [Objective](#objective)
- [Evaluation](#evaluation)
- [Model](#model)
- [Data](#data)
- [Data Augmentation](#data-augmentation)
- [Installation](#installation)
- [Getting Started (Training)](#getting-started-training)
- [Inference](#inference)
- [Results](#results)
- [Post Analysis](#post-analysis)
- [Code](#code)
- [References](#references)
- [Folder Structure](#folder-structure)
- [License](#license)

---

## Objective
The Carvana Image Masking Challenge aims to generate highly precise masks for cars in images.  
Semantic segmentation is used to identify the boundaries of cars, contributing to applications such as autonomous driving and object detection.

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*GHRkj8iYO70Ws_wyfa9rhA.png" alt="Input Image">
  <br>
  <em>Input Image</em>
</p>

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*9Z_p2cEuu7uZaNmYtza8JA.png" alt="Output Image">
  <br>
  <em>Predicted Mask</em>
</p>

---

## Evaluation
The main evaluation metric is the **Dice coefficient** (equivalent to the F1-score in binary segmentation):

<img src="https://miro.medium.com/v2/resize:fit:544/1*5eHgttXxEMukIJdqr2_ZNw.png" alt="Dice Coefficient Formula" width="250"/>

- **Dice = 1** -> perfect overlap between predicted pixels (X) and ground truth (Y)  
- **Dice = 0** -> no overlap  

Our aim is to maximize Dice by improving overlap between prediction and ground truth.

---

## Model
I developed a custom **encoder–decoder architecture**, inspired by both **SegNet** and **U-Net**:

- Encoder: SegNet-style downsampling (Conv2D → BatchNorm → ReLU → MaxPool)  
- Decoder: U-Net-style upsampling with skip connections from encoder layers  
- Final layer: **Sigmoid** (binary mask output)

**Architecture Summary**:
- 7 encoder layers  
- 2 center convolutional layers  
- 7 decoder layers  
- 1 final classification layer  

<p align="center">
  <img src="https://production-media.paperswithcode.com/methods/segnet_Vorazx7.png" alt="Model Architecture">
  <br>
  <em>Encoder–Decoder with skip connections (final activation: sigmoid)</em>
</p>

*Note:* Hardware limitations (NVIDIA RTX 3060, 6GB VRAM) influenced design choices.

---

## Data
The dataset is provided by Kaggle:  
[Carvana Image Masking Challenge Data](https://www.kaggle.com/c/carvana-image-masking-challenge/data)

Expected folder structure:
```
data/
├── raw/
│   ├── train/          # input images
│   └── train_masks/    # ground truth masks
└── processed/          # preprocessed data
```

---

## Data Augmentation
To improve generalization, I applied minor augmentations:
- Random shifts  
- Scaling  
- Rotations  

These help the model perform better on unseen data.

---

## Installation

Clone the repository and set up the environment:

'''bash
git clone https://github.com/chaitanyapeshin/segmentation_for_color_change.git
cd segmentation_for_color_change
'''

**Conda**
'''bash
conda env create -f environment.yml
conda activate carvana
'''

---

## Getting Started (Training)
Run the training notebook:

'''bash
jupyter notebook notebooks/model.ipynb
'''

This will train the model and log progress to TensorBoard (`assets/tensorboard/`).

---

## Inference
Use the inference script to predict masks for new images:

'''bash
python infer.py --input path/to/image.jpg --output outputs/mask.png
'''

---

## Results
The model was trained with **Adam optimizer** and a custom loss = BCE + (1 - Dice).  
Validation performance after ~13 epochs:

| Metric             | Value |
|--------------------|-------|
| **Dice**           | 0.9956 |
| **IoU (Jaccard)**  | 0.9912 |
| **Pixel Accuracy** | 0.9971 |
| **Precision**      | 0.9965 |
| **Recall**         | 0.9948 |
| **Dice (5/50/95%)**| 0.992 / 0.996 / 0.998 |

---

## Post Analysis
- The model segments the **main body** of cars very well.  
- Struggles with fine details:
  - Dark shadows near wheels  
  - Cars painted similar to background  
  - Thin structures (antennas, roof racks)  

Despite these challenges, performance is **human-level or better** on most images.

---

## Code
- Model implementation: [`notebooks/model.ipynb`](./notebooks/model.ipynb)  
- Preprocessing: [`src/data`](./src/data)  

---

## References
- [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation (2015)](https://arxiv.org/abs/1511.00561) – Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla  
- [Fully Convolutional Networks for Semantic Segmentation (2016)](https://arxiv.org/abs/1411.4038) – Evan Shelhamer, Jonathan Long, Trevor Darrell  
- [Learning Deconvolution Network for Semantic Segmentation (2015)](https://arxiv.org/abs/1505.04366) – Hyeonwoo Noh, Seunghoon Hong, Bohyung Han  

---

## Folder Structure
```
.  
├── 29bb3ece3180_11.jpg  
├── assets/  
│   └── tensorboard/  
├── data/  
│   ├── processed/  
│   └── raw/  
├── LICENSE  
├── notebooks/  
│   └── model.ipynb  
├── README.md  
├── references/  
│   ├── 1411.4038.pdf  
│   ├── 1505.04366.pdf  
│   └── 1511.00561.pdf  
├── environment.yml  
├── requirements.txt  
├── sample_submission.csv  
└── src/  
    └── data/  
```

---

## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.