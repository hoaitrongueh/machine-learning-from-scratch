# Machine Learning from Scratch

Python exercises documenting my progress from gradient descent and linear regression to a neural network trained with NumPy. I implement the forward passes, losses, gradients, and parameter updates myself. Scikit-learn supplies datasets and data splitting; the MNIST models use no automatic differentiation.

## Learning progression

1. Gradient descent and linear regression, first with Python loops and then batch updates.
2. Binary logistic regression, cross-entropy, feature standardization, NumPy vectorization, and train/validation/test splits.
3. Real datasets: breast-cancer classification and handwritten-digit recognition.
4. A neural network with a ReLU hidden layer, manual backpropagation, and shuffled mini-batch SGD.

## MNIST milestones

| Model | Script | Train accuracy | Test accuracy |
| --- | --- | --- | --- |
| Softmax regression (784 → 10) | [mnist_softmax_regression.py](mnist_softmax_regression.py) | ~90.46% | ~90.46% |
| One-hidden-layer network (784 → 128 → 10) | [mnist_neural_network.py](mnist_neural_network.py) | 94.94% | 94.49% |

These are recorded results from my study runs, not newly reproduced benchmarks. Test accuracy improved by about 4.03 percentage points. The models also use different training schedules, so this is not a controlled architecture comparison.

Both scripts load OpenML `mnist_784`, version 1, scale pixels by 255, and use an 80/20 random split with `random_state=42`: 56,000 training images and 14,000 test images. This differs from MNIST's standard 60,000/10,000 split.

| Setting | Softmax regression | Neural network |
| --- | --- | --- |
| Epochs | 1,000 | 100 |
| Learning rate | 0.1 | 0.01 |
| Batch size | Full training set | 256 |
| Initialization | Zero weights and biases | Normal weights (standard deviation 0.01), zero biases |
| Random generator | No random weight initialization | NumPy `default_rng(42)` |

The neural network computes gradients through both layers and ReLU before updating parameters. Its printed epoch loss is weighted by batch size, including the smaller final batch, and averages losses observed as parameters change during training.

## Running an exercise

Install Python 3 and the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a milestone from the repository folder:

```bash
python mnist_softmax_regression.py
python mnist_neural_network.py
```

The first MNIST run downloads data from OpenML and needs internet access. Later runs can reuse scikit-learn's local cache. Training can take time, especially the full-batch baseline.

Earlier exercises remain standalone scripts, for example:

```bash
python logistic_regression_train_val_test.py
python breast_cancer_logistic_regression.py
```

## About the code

This is a learning repository. Repeated implementations preserve stages of my understanding, and older practice scripts may still contain mistakes. Small synthetic examples demonstrate calculations rather than real-world performance.

The softmax milestone was previously named `handwritten_classifier_mnist.py`. The neural-network milestone comes from my local `neural_network_from_scratch.py` exercise.

Next I plan to study initialization for ReLU networks and use a validation set for model selection while reserving the test set for final evaluation.
