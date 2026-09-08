# MNIST-Digit-Classification-Model

A simple, end-to-end deep learning project built in Python using **PyTorch**. This model classifies handwritten digits from the classic MNIST dataset using a custom Feed-Forward Neural Network (Multi-Layer Perceptron).

## Features
* Data Pipeline: Automatically downloads, normalizes, and batches the MNIST dataset using 'torchvision'.
* Neural Network Architecture: Built with PyTorch ('nn.Module'), featuring an input flattening layer, a hidden layer with ReLU activation, and a 10-class output layer.
* Training & Optimization: Implements a full training loop using Cross-Entropy Loss and the Adam optimizer.
* Visual Evaluation: Includes a custom **Matplotlib** script that pulls a test batch, runs predictions, and plots the results in a 2x5 grid (color-coded green for correct predictions and red for errors).

---

## Tech Stack
* Language: Python
* Frameworks & Libraries: PyTorch, Torchvision, Matplotlib, NumPy

---


