# Single-Layer Perceptron (From Scratch)

```
single-layer-perceptron/
│
├── perceptron/
│ ├── **init**.py
│ │
│ ├── model.py
│ │ # Perceptron class:
│ │ # - weights, bias
│ │ # - forward pass
│ │ # - training loop
│ │ # - update rule
│ │
│ ├── activations.py
│ │ # step() → core perceptron activation
│ │ # sigmoid() → OPTIONAL extension (clearly labeled)
│ │
│ └── utils.py
│ # Data generation, plotting, helpers
│
├── examples/
│ ├── logic_gates.py
│ │ # AND, OR → should converge
│ │ # XOR → expected to FAIL (documented limitation)
│ │
│ └── binary_classification.py
│ # Linearly separable 2D data
│
├── tests/
│ └── test_perceptron.py
│ # Weight updates, predictions, convergence sanity checks
│
├── requirements.txt
│ # numpy
│ # matplotlib
│
├── README.md
│ # What a perceptron is
│ # Training rule explained
│ # Why XOR fails
│ # What this project is NOT
│
└── .gitignore
```

---

## Purpose of This Project

This repository implements a **single-layer perceptron from scratch** using only Python and NumPy.

The goal is **understanding**, not performance and not convenience.

This project deliberately avoids:

- scikit-learn
- TensorFlow
- PyTorch
- automatic differentiation
- hidden layers

If you finish this project correctly, you should be able to:

- Explain how a perceptron learns using only algebra and logic
- Implement the update rule without copying formulas
- Explain why XOR cannot be solved by a single-layer perceptron
- Draw the decision boundary by hand

If you cannot do those things, the project is not complete.

---

## What This Project Is (and Is Not)

### This project **is**

- A single neuron
- A linear classifier
- A historical and conceptual foundation of neural networks
- A learning exercise focused on mechanics and intuition

### This project **is not**

- A neural network framework
- Logistic regression (unless explicitly stated as an extension)
- Deep learning
- Optimized or production-ready code

---

## Repository Structure Overview

```text
single-layer-perceptron/
├── perceptron/
├── examples/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

Each directory has a **single responsibility**. Do not mix concerns.

---

## `perceptron/` — Core Implementation

This directory contains the actual perceptron logic.

### `perceptron/model.py`

This is the **heart of the project**.

You will implement a `Perceptron` class here.

#### Responsibilities

- Store weights and bias
- Compute predictions (forward pass)
- Train using the perceptron learning rule
- Update parameters when mistakes occur

#### Expected concepts inside this file

- Weight vector `w`
- Bias scalar `b`
- Dot product between input and weights
- Learning rate
- Epoch-based training loop

#### What you should _not_ add

- Multiple layers
- Backpropagation
- Optimizers
- Inheritance hierarchies

This file should contain **one class** and feel small and understandable.

---

### `perceptron/activations.py`

This file defines activation functions.

#### Required

- `step(x)`
  Returns `0` or `1` based on the sign of `x`.

This is the **true perceptron activation**.

#### Optional Extension

- `sigmoid(x)`
  Only include this if you clearly document that:

  - This turns the model into logistic regression
  - This is _not_ part of the original perceptron

Do not use sigmoid by default.

---

### `perceptron/utils.py`

This file contains **support code**, not learning logic.

#### Responsibilities

- Generate synthetic datasets
- Visualize data points
- Plot decision boundaries
- Small helper functions

This file should:

- Use NumPy and matplotlib
- Contain no training logic
- Contain no model updates

If math shows up here, it is probably misplaced.

---

## `examples/` — How the Perceptron Is Used

This directory demonstrates what the perceptron can and cannot do.

### `examples/logic_gates.py`

This file tests the perceptron on logical gates.

#### AND gate

- Should converge
- Linearly separable

#### OR gate

- Should converge
- Linearly separable

#### XOR gate

- **Must fail**
- Not linearly separable

The XOR example is **not a bug**.
It is the most important lesson in the repository.

You should document clearly in this file:

- That failure is expected
- Why it happens geometrically

---

### `examples/binary_classification.py`

This file moves beyond toy logic gates.

#### Responsibilities

- Train the perceptron on 2D numeric data
- Show predictions
- Plot decision boundaries
- Observe convergence behavior

This is where intuition about geometry forms.

---

## `tests/` — Correctness Checks

### `tests/test_perceptron.py`

This file ensures your implementation is **not silently wrong**.

#### Minimum tests you should write

- Weights change after training
- Predictions are correct on simple data
- Bias updates correctly
- Model converges on separable data

Tests should be simple and readable.
They are here to catch mistakes, not to impress anyone.

---

## `requirements.txt`

Expected dependencies:

```text
numpy
matplotlib
```

Nothing else is allowed.

If you feel tempted to add another library, stop and ask why.

---

## `.gitignore`

At minimum, ignore:

- `__pycache__/`
- `.DS_Store`
- `.ipynb_checkpoints/`

---

## Training Rule Explained (Conceptual)

The perceptron learns by correcting mistakes.

For each training example:

1. Compute a weighted sum of inputs
2. Apply the step function
3. Compare prediction to true label
4. If incorrect:

   - Adjust weights in the direction that reduces error
   - Adjust bias accordingly

This is **error-driven learning**, not gradient descent.

You should be able to explain this update rule in words and numbers.

---

## Why XOR Fails

XOR cannot be separated by a single straight line.

A single-layer perceptron:

- Can only create linear decision boundaries
- Cannot represent non-linear separations

This limitation is **the reason multi-layer networks exist**.

If XOR worked here, neural network history would be very different.

---

## What “Done” Means

This project is complete when:

- AND and OR converge consistently
- XOR fails consistently
- You can explain why without memorization
- You understand what adding a hidden layer changes
- You no longer feel tempted to jump straight to frameworks

---

## Recommended Next Steps (After Completion)

Only after finishing this project should you move on to:

- Multi-layer perceptrons
- Backpropagation
- Automatic differentiation
- Frameworks like PyTorch or TensorFlow

Skipping this foundation causes long-term confusion.

---

## Final Instruction to Yourself

If something feels “too easy,” slow down.
If something feels confusing, trace it by hand.
If something fails, understand _why_ before fixing it.

This project is not about speed.
It is about **ownership of the ideas**.
