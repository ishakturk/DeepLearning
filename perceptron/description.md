# Single-Layer Perceptron for Logic Gates

This project is a from-scratch implementation of a single-layer Perceptron using NumPy. Its purpose is to demonstrate the capabilities and, more importantly, the limitations of a single neuron by testing it on three basic logic gates: **AND**, **OR**, and **XOR**.

## What is a Perceptron?

A Perceptron is the simplest form of a neural network, consisting of a single neuron. It takes multiple binary inputs, applies a weight to each, sums them up, adds a bias, and then passes the result through an activation function (in this case, a step function) to produce a single binary output.

The core formula is:
$$ \text{output} = \text{activation}(\sum_{i}(\text{weight}_i \times \text{input}_i) + \text{bias}) $$

The model "learns" by adjusting its weights and bias based on the errors it makes during training.

## How to Run

1.  Make sure you have `numpy` installed.
2.  Save the accompanying Python code (e.g., `logic_gates.py`).
3.  Inside the Python file, in the `if __name__ == "__main__":` block, you can comment and uncomment the sections for `AND`, `OR`, and `XOR` to run each test.
4.  Execute the file from your terminal: `python logic_gates.py`

---

## The Experiments & Results

The goal is to see if the Perceptron can learn a set of weights and a bias that correctly classify all four possible inputs for each logic gate.

### ✅ AND Gate (Success)

The model successfully learns the AND gate. It quickly reaches 100% accuracy.

| Input 1 | Input 2 | Output |
| :-----: | :-----: | :----: |
|    0    |    0    |   **0** |
|    0    |    1    |   **0** |
|    1    |    0    |   **0** |
|    1    |    1    |   **1** |

**Why it works:** The AND gate is **linearly separable**. This means you can draw a single straight line to separate the `(1,1)` point (which outputs 1) from the other three points (which output 0). The Perceptron's job is to find the equation of that line.



### ✅ OR Gate (Success)

The model also successfully learns the OR gate, reaching 100% accuracy.

| Input 1 | Input 2 | Output |
| :-----: | :-----: | :----: |
|    0    |    0    |   **0** |
|    0    |    1    |   **1** |
|    1    |    0    |   **1** |
|    1    |    1    |   **1** |

**Why it works:** The OR gate is also **linearly separable**. You can draw a single straight line to separate the `(0,0)` point from the other three.



### ❌ XOR Gate (Failure)

The model **fails** to learn the XOR gate. You'll notice that the accuracy fluctuates (usually around 50% or 75%) but never reaches 100% and settles.

| Input 1 | Input 2 | Output |
| :-----: | :-----: | :----: |
|    0    |    0    |   **0** |
|    0    |    1    |   **1** |
|    1    |    0    |   **1** |
|    1    |    1    |   **0** |

**Why it fails:** The XOR gate is **not linearly separable**. There is no way to draw a *single straight line* that separates the points that should output 1 (`(0,1)` and `(1,0)`) from the points that should output 0 (`(0,0)` and `(1,1)`).



---

## 🧠 Key Takeaway

This experiment demonstrates the fundamental limitation of a single neuron. It can only solve problems where the different classes can be separated by a straight line (or a flat plane in higher dimensions).

The failure of the Perceptron to solve the XOR problem was a historically significant event that led to a decline in neural network research (the first "AI winter"). The solution, is to use **multiple layers of neurons** (a Multi-Layer Perceptron or MLP). By combining neurons, a network can create complex, non-linear decision boundaries, easily solving problems like XOR.