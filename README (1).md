# MNIST Handwritten Digit Classifier

A simple, end-to-end deep learning project built in Python using **PyTorch**. This model classifies handwritten digits from the classic MNIST dataset using a custom Feed-Forward Neural Network (Multi-Layer Perceptron).

## 🚀 Features
* **Data Pipeline:** Automatically downloads, normalizes, and batches the MNIST dataset using `torchvision`.
* **Neural Network Architecture:** Built with PyTorch (`nn.Module`), featuring an input flattening layer, a hidden layer with ReLU activation, and a 10-class output layer.
* **Training & Optimization:** Implements a full training loop using Cross-Entropy Loss and the Adam optimizer.
* **Visual Evaluation:** Includes a custom **Matplotlib** script that pulls a test batch, runs predictions, and plots the results in a 2x5 grid (color-coded green for correct predictions and red for errors).

---

## 🛠️ Tech Stack
* **Language:** Python
* **Frameworks & Libraries:** PyTorch, Torchvision, Matplotlib, NumPy

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mnist-neural-net.git
   cd mnist-neural-net
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision matplotlib
   ```

---

## 🏃‍♂️ Usage

Run the main script to download the dataset, train the model, print final testing accuracy, and launch the prediction visualization window:

```bash
python main.py
```

---

## 📊 Project Structure
```text
mnist-neural-net/
│
├── data/               # Automatically downloaded MNIST dataset
├── main.py             # Main training, evaluation, and visualization script
└── README.md           # Project documentation
```
