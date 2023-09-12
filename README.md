# Carvana Image Masking Challenge

![Carvana Challenge](https://www.carvana.com/images/carvana-logo.png)

Welcome to the Carvana Image Masking Challenge repository. This project focuses on semantic segmentation for cars as part of the Carvana Image Masking Challenge on Kaggle. The goal is to generate precise masks for cars in images.

## Table of Contents

- [Objective](#objective)
- [Evaluation](#evaluation)
- [Model](#model)
- [Data](#data)
- [Data Augmentation](#data-augmentation)
- [Training](#training)
- [Post Analysis](#post-analysis)
- [Code](#code)
- [References](#references)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [License](#license)

## Objective

The Carvana Image Masking Challenge aims to generate highly precise masks for cars in images. Semantic segmentation is used to precisely identify the boundaries of cars, contributing to various computer vision applications, including autonomous driving, object detection, and more.

## Evaluation

The evaluation metric used for this competition is the Dice coefficient, defined as follows:

![Dice coefficient formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/0c442abb87c6d649f9c4a1e17c52de4f6418f7cd)

A higher Dice coefficient indicates better segmentation accuracy. A perfect overlap between the predicted set of pixels (X) and the ground truth (Y) results in a Dice coefficient of 1. Our aim is to maximize this metric by improving the overlap between X and Y.

## Model

I developed a custom architecture for this task, building upon the SegNet architecture. The final model consists of:

- 7 encoder layers
- 2 center convolutional layers
- 7 decoder layers
- 1 final convolutional classification layer

Hardware limitations (NVIDIA RTX 3060 with 6GB VRAM) influenced my architecture choice. Key components include:

- Encoder layers: (conv2d, batchnorm, relu) x 2, followed by max-pooling
- Center layers: (conv2d, batchnorm, relu) x 2
- Decoder layers: upsampling2d, concatenate, (conv2d, batchnorm, relu) x 3, followed by max-pooling

Each decoder layer uses the output from the corresponding encoder layer (before max-pooling) to extract learnable features, similar to the SegNet architecture.

## Data

The data for this project is available through Kaggle and includes a collection of images and corresponding ground truth masks. The training data contains images of cars, and the challenge is to segment the cars accurately.

## Data Augmentation

To enhance the model's robustness, I applied minor data augmentation techniques, including random shifts, scaling, and rotations to the input images. This helps the model generalize better to unseen data.

## Training

I implemented the model using Keras with a Tensorflow backend. The training process included the following hyperparameters:

- Early stopping criteria: min_delta = 0.0001
- Optimizer: Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
- Loss function: Binary Cross-Entropy Loss + (1 - Dice Coefficient)

The validation set achieved a Dice coefficient of approximately 0.9956 after around 13 epochs.

## Post Analysis

Post-training analysis focused on the worst-performing images from the validation set. I observed that the model excelled at segmenting the main bodies of cars but struggled with certain details. These included dark shadows near the wheels, cars painted the same color as the background, small antennas, roof racks, and other challenging elements. These challenges are often difficult for humans to differentiate as well. Overall, our model achieved human-level performance or better for this task.

## Code

You can find the full implementation details and code of the model in the [notebooks](./notebooks/) directory, and for pre-processsing, in the [src/data](./src/data/) directory.

## References

You can find all the references in the [references](./references/) directory.

- [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation, 2015](https://arxiv.org/abs/1511.00561) - Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
- [Fully Convolutional Networks for Semantic Segmentation, 2016](https://arxiv.org/abs/1411.4038) - Evan Shelhamer, Jonathan Long, Trevor Darrell
- [Learning Deconvolution Network for Semantic Segmentation, 2015](https://arxiv.org/abs/1505.04366) - Hyeonwoo Noh, Seunghoon Hong, Bohyung Han

## Folder Structure
.
├── 29bb3ece3180_11.jpg
├── assets
│   └── tensorboard
├── data
│   ├── processed
│   └── raw
├── LICENSE
├── notebooks
│   └── model.ipynb
├── README.md
├── references
│   ├── 1411.4038.pdf
│   ├── 1505.04366.pdf
│   └── 1511.00561.pdf
├── requirements.txt
├── sample_submission.csv
└── src
    └── data

## Getting Started

To get started with the project, run the [model.ipynb](model.ipynb) notebook for training the model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
