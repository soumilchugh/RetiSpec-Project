# RetiSpec Project

This repository contains code for training and evaluating a CNN-based binary classifier using paired RGB and near-infrared image inputs.

The project uses a configurable training and testing pipeline with YACS configuration files, TensorBoard logging, and confusion-matrix based evaluation.

## Project Overview

The model predicts whether paired RGB and NIR images belong to one of two classes:

- **Forest**: class 0
- **River**: class 1

The network uses two image feature extractors, one for RGB images and one for NIR images, then fuses the learned representations before classification.

## Repository Structure

```text
config/
  defaults.py        Default YACS configuration
  train.yaml         Training-specific overrides
  test.yaml          Testing-specific overrides

model_training/
  dataset_loader.py  Data loading utilities
  dataset_creater.py Dataset creation utilities
  one_class_dataset.py
  transforms.py      Data transforms and augmentation
  network.py         CNN architecture

train.py             Training entry point
test.py              Testing entry point
models/              Saved training checkpoints and final model
logs/                TensorBoard logs
```

## Configuration

The project uses [YACS](https://github.com/rbgirshick/yacs) for experiment configuration.

- Default parameters live in `config/defaults.py`.
- Training-specific parameters live in `config/train.yaml`.
- Testing-specific parameters live in `config/test.yaml`.
- YAML values override defaults inside `train.py` and `test.py`.

## Network Architecture

The model has three main components:

1. RGB feature extractor
2. NIR feature extractor
3. Projection and classification head

Each feature extractor receives an image with shape `64 x 64 x C` and outputs an `8 x 8 x 32` feature map.

- RGB input channels: `C = 3`
- NIR input channels: `C = 1`

Each extractor includes three CNN blocks with convolution, batch normalization, ReLU activation, and max pooling. The extracted feature maps are flattened into 512-dimensional vectors and fused using elementwise multiplication.

The projection module contains two fully connected layers and maps the fused 512-dimensional representation to a 32-dimensional vector before classification.

## Data Augmentation

Training uses the `albumentations` library. Color-based augmentations were avoided because the model uses both RGB and NIR inputs. Augmentations focus on:

- Blur
- Distortion
- Scaling
- Shifting
- Rotation

## Dataset Split

The provided dataset contains `train` and `val` folders with balanced class counts.

Because no separate test dataset was provided:

- The validation images were used for testing.
- The training images were split into training and validation sets using an 80:20 ratio.

## TensorBoard

Training and validation curves can be viewed with TensorBoard:

```bash
tensorboard --logdir=logs/ --host localhost --port 8088
```

Then open:

```text
http://localhost:8088
```

## Evaluation

The network was trained for 20 epochs. Since the validation loss was still decreasing, longer training may improve performance.

The dataset is balanced, so accuracy is a reasonable evaluation metric. A confusion matrix is also computed for a more detailed view of class-level performance.

![Confusion Matrix](confusion_matrix.png?raw=true "Confusion Matrix")

## Portfolio Context

This project is part of Soumil Chugh's applied machine-learning and computer-vision portfolio, with emphasis on multimodal image inputs, CNN architecture design, and model evaluation.
