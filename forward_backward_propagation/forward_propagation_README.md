# Forward Propagation - Complete Guide

## 📚 Table of Contents
1. [What is Forward Propagation?](#what-is-forward-propagation)
2. [Key Concepts](#key-concepts)
3. [The Process Step-by-Step](#the-process-step-by-step)
4. [Code Structure](#code-structure)
5. [Examples](#examples)
6. [Common Activation Functions](#common-activation-functions)
7. [Quick Reference](#quick-reference)

---

## What is Forward Propagation?

**Forward propagation** is the process of passing input data through a neural network to produce an output (prediction). Data "flows forward" from the input layer, through hidden layers (if any), to the output layer.

### Real-World Analogy
Think of it like an assembly line:
- **Input**: Raw materials enter the factory
- **Hidden Layers**: Different processing stations transform the materials
- **Output**: Final product comes out
- **Loss**: Quality check - how good is the final product?

---

## Key Concepts

### 1. **Linear Transformation**
At each layer, we compute: **Z = W × X + b**

- **W** (weights): How much importance to give each input
- **X** (input): The data coming into this layer
- **b** (bias): An offset to shift the result
- **Z** (output): The result before activation

**Example**: If predicting house prices from size and bedrooms:
```
Z = w1×size + w2×bedrooms + b
  = 100×1500 + 5000×3 + 10000
  = 175,000
```

### 2. **Activation Function**
After computing Z, we apply a non-linear function: **A = activation(Z)**

**Why?** Without activation functions, no matter how many layers you add, the network can only learn linear relationships. Activation functions allow learning complex patterns.

### 3. **Loss Function**
Measures how wrong our predictions are: **Loss = (Prediction - Truth)²**

Lower loss = better predictions

---

## The Process Step-by-Step

### Single Layer Network

```
Input (X) → [Linear: Z = W×X + b] → [Activation: A = σ(Z)] → Output
```

**Step 1**: Multiply inputs by weights and add bias
```python
Z = np.dot(X, W) + b
```

**Step 2**: Apply activation function
```python
A = sigmoid(Z)  # Converts to probability between 0 and 1
```

### Two Layer Network (with Hidden Layer)

```
Input (X) → [Layer 1] → [Layer 2] → Output (Prediction)
```

**Layer 1 (Hidden Layer)**:
```python
Z1 = X @ W1 + b1      # Linear transformation
A1 = relu(Z1)         # Activation (ReLU keeps positive values)
```

**Layer 2 (Output Layer)**:
```python
Z2 = A1 @ W2 + b2     # Use previous layer's output as input
A2 = sigmoid(Z2)      # Final activation (prediction)
```

**Loss Calculation**:
```python
Loss = mean((A2 - Y)²)  # How far are predictions from truth?
```

---

## Code Structure

### File: `forward_propagation.py`

```
📦 forward_propagation.py
 ├── 📊 Sample Data (X, Y)
 ├── 🔧 Activation Functions
 │   ├── sigmoid(z)        - For output probabilities
 │   ├── relu(z)           - For hidden layers
 │   └── tanh(z)           - Alternative activation
 ├── ➡️ Forward Propagation Functions
 │   ├── forward_propagation_single_layer()   - One layer network
 │   └── forward_propagation_two_layer()      - Two layer network
 ├── 📉 Loss Function
 │   └── compute_loss()    - Measures prediction error
 └── 🎯 Demonstration
     └── if __name__ == "__main__"  - Example usage
```

---

## Examples

### Example 1: Single Layer Network

**Goal**: Predict if a student passes (0 or 1) based on study hours

**Input**: `X = [2.5]` (2.5 hours studied)  
**Weight**: `W = [0.8]`  
**Bias**: `b = [-1.5]`

**Forward Pass**:
```python
# Step 1: Linear transformation
Z = W × X + b = 0.8 × 2.5 + (-1.5) = 0.5

# Step 2: Activation (sigmoid)
A = sigmoid(0.5) = 1/(1 + e^(-0.5)) ≈ 0.62

# Interpretation: 62% chance of passing
```

### Example 2: Two Layer Network with Multiple Inputs

**Goal**: Predict house price category (cheap/expensive)

**Input**: `X = [1500, 3]` (1500 sq ft, 3 bedrooms)  
**Architecture**: 2 inputs → 4 hidden neurons → 1 output

**Layer 1 (Hidden)**:
```python
Z1 = X @ W1 + b1        # Shape: (1, 4) - 4 hidden neurons
A1 = relu(Z1)           # Activation
# A1 might look like: [0, 1.2, 0.8, 0]  (ReLU keeps positives)
```

**Layer 2 (Output)**:
```python
Z2 = A1 @ W2 + b2       # Shape: (1, 1) - single output
A2 = sigmoid(Z2)        # Final prediction
# A2 = 0.75  →  75% chance it's expensive
```

---

## Common Activation Functions

### 1. **Sigmoid** σ(z) = 1 / (1 + e^(-z))

**Range**: 0 to 1  
**Use**: Output layer for binary classification (probabilities)  
**Graph**: S-shaped curve

```
Input:  -5   -2    0    2    5
Output: 0.01 0.12 0.5 0.88 0.99
```

### 2. **ReLU** (Rectified Linear Unit)

**Formula**: ReLU(z) = max(0, z)  
**Range**: 0 to ∞  
**Use**: Hidden layers (fast, prevents vanishing gradients)

```
Input:  -5   -2    0    2    5
Output:  0    0    0    2    5
```

### 3. **Tanh** (Hyperbolic Tangent)

**Range**: -1 to 1  
**Use**: Hidden layers (zero-centered outputs)

```
Input:  -5    -2     0     2     5
Output: -0.99 -0.96  0   0.96  0.99
```

---

## Quick Reference

### When to Use Which Activation?

| Layer Type | Recommended Activation | Why? |
|------------|------------------------|------|
| Hidden Layers | **ReLU** | Fast, avoids vanishing gradients |
| Output (Binary Classification) | **Sigmoid** | Outputs probability 0-1 |
| Output (Multi-class) | **Softmax** | Probabilities sum to 1 |
| Output (Regression) | **None (Linear)** | Can output any real number |

### Shape Guide

For a network with:
- `n` samples
- `f` features
- `h` hidden neurons
- `o` outputs

```
X:  (n, f)
W1: (f, h)  →  Z1: (n, h)  →  A1: (n, h)
W2: (h, o)  →  Z2: (n, o)  →  A2: (n, o)
```

### Key Formulas

```
Linear:     Z = W × X + b
Sigmoid:    σ(z) = 1 / (1 + e^(-z))
ReLU:       ReLU(z) = max(0, z)
MSE Loss:   L = (1/n) × Σ(ŷ - y)²
```

---

## Running the Code

```bash
# Navigate to the directory
cd forward_backward_propagation

# Run the demonstration
python forward_propagation.py
```

**Expected Output**:
- Example of single layer network
- Example of two layer network
- Loss values showing prediction accuracy

---

## Key Takeaways 🎯

1. **Forward propagation = making predictions** by passing data through layers
2. **Each layer** transforms data: Linear → Activation
3. **Weights and biases** are the parameters we'll learn (via backpropagation)
4. **Activation functions** add non-linearity, enabling complex pattern learning
5. **Loss function** tells us how good our predictions are
6. This is only **half the story** - backpropagation will teach us how to improve!

---

## What's Next?

After understanding forward propagation, learn:
- **Backward Propagation** (`backward_propagation_README.md`) - How to compute gradients
- **Gradient Descent** (`gradient_descent_README.md`) - How to update parameters

