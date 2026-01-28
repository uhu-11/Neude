# Triangular Projection

This repository contains a brand new diversity-based testing criterion , `triangular projection(pt)` , for testing deep
neural networks.

## Introduction

Different from the traditional idea of structure coverage, we do not start from the activation state of neurons, but
from the perspective of **the diversity of the DNNs' output layer**, we obtain a test suite evaluation criterion of the
current model.

Based on the application scenarios of detection, selection and generation, we fully verified the effectiveness of
triangular projection in DNNs testing.

## Installation
Environment: python 3.7 on Linux

```
pip install -r requirements.txt
```

## Usage

Run pt tools with following cmd

`python tools.py --split_num {} --class_num {} --data_path {} --model_path {}`.

- split_num: The hyperparameters of the algorithm, the number of partitions of each triangular subspace. Larger values
  lead to a more fine-grained algorithm, but the computational cost also increases.
- class_num: Number of classifications in the model
- data_path: Path to the dataset, including images and labels.
- model_path: Path to the model.

An example

`python tools.py --split_num 4 --class_num 10 --data_path "data/mnist" --model_path "model/model_mnist_LeNet5.hdf5"`.

