"""
Backward Propagation Implementation
====================================
This script demonstrates backward propagation (backpropagation) in a neural network.
Backpropagation computes gradients of the loss with respect to all parameters,
allowing us to update weights and biases to minimize the error.
"""

import numpy as np


# ============================================================================
# SAMPLE DATA
# ============================================================================
# Input features: Each row is a sample, each column is a feature
X = np.array([
    [0.5, 1.0],   # Sample 1: two features
    [2.5, 0.5],   # Sample 2: two features
    [1.5, 2.0]    # Sample 3: two features
])

# Target outputs: What we want the network to predict
Y = np.array([
    [0.2],  # Expected output for sample 1
    [0.9],  # Expected output for sample 2
    [0.6]   # Expected output for sample 3
])


# ============================================================================
# ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# ============================================================================

def sigmoid(z):
    """
    Sigmoid activation function.

    Formula: σ(z) = 1 / (1 + e^(-z))
    Output range: (0, 1)
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid function with respect to z.

    Formula: σ'(z) = σ(z) * (1 - σ(z))

    This derivative tells us how much the sigmoid output changes
    when we change the input z slightly.

    Args:
        z: Linear output (before activation)

    Returns:
        Derivative value(s)

    Note: If you already have σ(z), you can use: σ(z) * (1 - σ(z))
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """
    ReLU activation function.

    Formula: ReLU(z) = max(0, z)
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU function with respect to z.

    Formula: ReLU'(z) = 1 if z > 0, else 0

    The derivative is:
    - 1 when input is positive (gradient flows through)
    - 0 when input is negative or zero (gradient is blocked)

    Args:
        z: Linear output (before activation)

    Returns:
        Derivative value(s): 1 or 0
    """
    return (z > 0).astype(float)


def tanh_derivative(z):
    """
    Derivative of tanh function with respect to z.

    Formula: tanh'(z) = 1 - tanh²(z)

    Args:
        z: Linear output (before activation)

    Returns:
        Derivative value(s)
    """
    t = np.tanh(z)
    return 1 - t ** 2


# ============================================================================
# FORWARD PROPAGATION (needed for backprop)
# ============================================================================

def forward_propagation_two_layer(X, W1, b1, W2, b2):
    """
    Perform forward propagation and cache values needed for backpropagation.

    Architecture: Input → Hidden Layer (ReLU) → Output Layer (Sigmoid)

    Args:
        X: Input data, shape (n_samples, n_features)
        W1, b1: First layer parameters
        W2, b2: Second layer parameters

    Returns:
        cache: Dictionary with all intermediate values
    """
    n_samples = X.shape[0]

    # ===== LAYER 1: HIDDEN LAYER =====
    Z1 = np.dot(X, W1) + b1  # Linear transformation
    A1 = relu(Z1)             # ReLU activation

    # ===== LAYER 2: OUTPUT LAYER =====
    Z2 = np.dot(A1, W2) + b2  # Linear transformation
    A2 = sigmoid(Z2)          # Sigmoid activation (final prediction)

    # Cache everything needed for backward pass
    cache = {
        'X': X,    # Input data
        'Z1': Z1,  # Hidden layer linear output
        'A1': A1,  # Hidden layer activation
        'W1': W1,  # Hidden layer weights
        'b1': b1,  # Hidden layer bias
        'Z2': Z2,  # Output layer linear output
        'A2': A2,  # Final predictions
        'W2': W2,  # Output layer weights
        'b2': b2   # Output layer bias
    }

    return cache


# ============================================================================
# BACKWARD PROPAGATION FUNCTIONS
# ============================================================================

def compute_loss_gradient(Y_pred, Y_true):
    """
    Compute the gradient of the loss function with respect to predictions.

    Loss function: MSE = (1/2n) * Σ(y_pred - y_true)²
    Derivative: ∂Loss/∂y_pred = (y_pred - y_true) / n

    This is the starting point of backpropagation - it tells us how much
    each prediction contributes to the overall loss.

    Args:
        Y_pred: Predicted values, shape (n_samples, n_outputs)
        Y_true: True values, shape (n_samples, n_outputs)

    Returns:
        dA: Gradient of loss w.r.t. predictions, same shape as Y_pred
    """
    n_samples = Y_true.shape[0]

    # How much does each prediction differ from the truth?
    # Positive = we predicted too high, Negative = we predicted too low
    dA = (Y_pred - Y_true) / n_samples

    return dA


def backward_propagation_two_layer(cache, Y_true):
    """
    Perform backward propagation through a two-layer network.

    This computes gradients using the CHAIN RULE:
    - Start from the loss and work backwards through each layer
    - At each step, multiply by the local gradient (derivative)

    Architecture flow (backward):
    Loss ← Output Layer ← Hidden Layer ← Input

    Chain rule example for W2:
    ∂Loss/∂W2 = ∂Loss/∂A2 × ∂A2/∂Z2 × ∂Z2/∂W2

    Args:
        cache: Dictionary from forward propagation with all intermediate values
        Y_true: True labels, shape (n_samples, n_outputs)

    Returns:
        gradients: Dictionary containing gradients for all parameters
            - dW2: Gradient for output layer weights
            - db2: Gradient for output layer bias
            - dW1: Gradient for hidden layer weights
            - db1: Gradient for hidden layer bias
    """

    # Extract values from cache
    X = cache['X']
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']
    Z2 = cache['Z2']
    W2 = cache['W2']

    n_samples = X.shape[0]

    # ========================================================================
    # STEP 1: Compute gradient of loss w.r.t. final output (A2)
    # ========================================================================
    # This tells us: "How much does the loss change if we change A2?"
    # Formula: ∂Loss/∂A2 = (A2 - Y) / n
    dA2 = compute_loss_gradient(A2, Y_true)

    # ========================================================================
    # STEP 2: Backpropagate through OUTPUT LAYER (Layer 2)
    # ========================================================================

    # 2a. Gradient w.r.t. Z2 (before activation)
    # Apply chain rule: ∂Loss/∂Z2 = ∂Loss/∂A2 × ∂A2/∂Z2
    # where ∂A2/∂Z2 = sigmoid'(Z2) = A2 * (1 - A2)
    dZ2 = dA2 * sigmoid_derivative(Z2)

    # 2b. Gradient w.r.t. W2 (output layer weights)
    # Formula: ∂Loss/∂W2 = A1^T @ ∂Loss/∂Z2
    # This tells us how much each weight contributed to the error
    dW2 = np.dot(A1.T, dZ2)

    # 2c. Gradient w.r.t. b2 (output layer bias)
    # Formula: ∂Loss/∂b2 = sum(∂Loss/∂Z2) across all samples
    # Bias affects all samples equally, so we sum the gradients
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # 2d. Gradient w.r.t. A1 (hidden layer activations)
    # This propagates the error backwards to the previous layer
    # Formula: ∂Loss/∂A1 = ∂Loss/∂Z2 @ W2^T
    dA1 = np.dot(dZ2, W2.T)

    # ========================================================================
    # STEP 3: Backpropagate through HIDDEN LAYER (Layer 1)
    # ========================================================================

    # 3a. Gradient w.r.t. Z1 (before activation)
    # Apply chain rule: ∂Loss/∂Z1 = ∂Loss/∂A1 × ∂A1/∂Z1
    # where ∂A1/∂Z1 = relu'(Z1) = 1 if Z1 > 0, else 0
    dZ1 = dA1 * relu_derivative(Z1)

    # 3b. Gradient w.r.t. W1 (hidden layer weights)
    # Formula: ∂Loss/∂W1 = X^T @ ∂Loss/∂Z1
    dW1 = np.dot(X.T, dZ1)

    # 3c. Gradient w.r.t. b1 (hidden layer bias)
    # Formula: ∂Loss/∂b1 = sum(∂Loss/∂Z1) across all samples
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # ========================================================================
    # RETURN ALL GRADIENTS
    # ========================================================================
    gradients = {
        'dW2': dW2,  # How to adjust output layer weights
        'db2': db2,  # How to adjust output layer bias
        'dW1': dW1,  # How to adjust hidden layer weights
        'db1': db1,  # How to adjust hidden layer bias
        # Additional gradients for analysis
        'dZ2': dZ2,  # Gradient at output layer (before activation)
        'dA1': dA1,  # Gradient at hidden layer (after activation)
        'dZ1': dZ1   # Gradient at hidden layer (before activation)
    }

    return gradients


def update_parameters(W1, b1, W2, b2, gradients, learning_rate):
    """
    Update network parameters using computed gradients.

    This implements the GRADIENT DESCENT update rule:
    parameter_new = parameter_old - learning_rate × gradient

    Why subtract?
    - Gradient points in the direction of INCREASING loss
    - We want to DECREASE loss, so we move in the opposite direction

    Args:
        W1, b1: Hidden layer parameters
        W2, b2: Output layer parameters
        gradients: Dictionary with computed gradients
        learning_rate: Step size (how much to adjust parameters)

    Returns:
        Updated parameters: W1, b1, W2, b2
    """

    # Update hidden layer parameters
    # Move weights in the direction that reduces loss
    W1 = W1 - learning_rate * gradients['dW1']
    b1 = b1 - learning_rate * gradients['db1']

    # Update output layer parameters
    W2 = W2 - learning_rate * gradients['dW2']
    b2 = b2 - learning_rate * gradients['db2']

    return W1, b1, W2, b2


def compute_loss(Y_pred, Y_true):
    """
    Compute Mean Squared Error loss.

    Formula: MSE = (1/2n) * Σ(y_pred - y_true)²
    """
    n_samples = Y_true.shape[0]
    loss = np.sum((Y_pred - Y_true) ** 2) / (2 * n_samples)
    return loss


# ============================================================================
# COMPLETE TRAINING LOOP WITH BACKPROPAGATION
# ============================================================================

def train_network(X, Y, n_hidden, learning_rate=0.1, epochs=1000, print_every=100):
    """
    Train a two-layer neural network using backpropagation.

    This function combines:
    1. Forward propagation (make predictions)
    2. Loss computation (measure error)
    3. Backward propagation (compute gradients)
    4. Parameter update (improve the model)

    Args:
        X: Input data, shape (n_samples, n_features)
        Y: Target outputs, shape (n_samples, n_outputs)
        n_hidden: Number of neurons in hidden layer
        learning_rate: How much to adjust parameters each step
        epochs: Number of times to iterate through the data
        print_every: Print progress every N epochs

    Returns:
        Trained parameters and loss history
    """

    # ===== INITIALIZE PARAMETERS =====
    n_features = X.shape[1]
    n_outputs = Y.shape[1]

    # Initialize weights with small random values
    # (Random initialization breaks symmetry - important for learning!)
    W1 = np.random.randn(n_features, n_hidden) * 0.5
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_outputs) * 0.5
    b2 = np.zeros((1, n_outputs))

    loss_history = []

    print("=" * 70)
    print(f"Training Neural Network")
    print("=" * 70)
    print(f"Architecture: {n_features} inputs → {n_hidden} hidden → {n_outputs} outputs")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print("=" * 70)

    # ===== TRAINING LOOP =====
    for epoch in range(epochs):

        # 1. FORWARD PROPAGATION: Make predictions
        cache = forward_propagation_two_layer(X, W1, b1, W2, b2)
        Y_pred = cache['A2']

        # 2. COMPUTE LOSS: Measure how wrong we are
        loss = compute_loss(Y_pred, Y)
        loss_history.append(loss)

        # 3. BACKWARD PROPAGATION: Compute gradients
        gradients = backward_propagation_two_layer(cache, Y)

        # 4. UPDATE PARAMETERS: Improve the model
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, gradients, learning_rate)

        # Print progress
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    print("=" * 70)
    print("Training complete!")
    print("=" * 70)

    return W1, b1, W2, b2, loss_history


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BACKWARD PROPAGATION DEMONSTRATION")
    print("=" * 70)

    # ===== TRAIN THE NETWORK =====
    W1, b1, W2, b2, loss_history = train_network(
        X, Y,
        n_hidden=4,
        learning_rate=0.5,
        epochs=500,
        print_every=50
    )

    # ===== SHOW FINAL RESULTS =====
    print("\nFinal Results:")
    print("-" * 70)

    # Make final predictions
    cache = forward_propagation_two_layer(X, W1, b1, W2, b2)
    Y_pred = cache['A2']

    print("Input data:")
    print(X)
    print("\nTrue values:")
    print(Y)
    print("\nFinal predictions:")
    print(Y_pred)
    print("\nPrediction errors:")
    print(Y - Y_pred)

    # ===== DEMONSTRATE GRADIENT COMPUTATION =====
    print("\n" + "=" * 70)
    print("GRADIENT ANALYSIS (Final Epoch)")
    print("=" * 70)

    gradients = backward_propagation_two_layer(cache, Y)

    print("\nGradient magnitudes (how much each parameter should change):")
    print(f"  dW2 (output weights): mean absolute = {np.mean(np.abs(gradients['dW2'])):.6f}")
    print(f"  db2 (output bias):    mean absolute = {np.mean(np.abs(gradients['db2'])):.6f}")
    print(f"  dW1 (hidden weights): mean absolute = {np.mean(np.abs(gradients['dW1'])):.6f}")
    print(f"  db1 (hidden bias):    mean absolute = {np.mean(np.abs(gradients['db1'])):.6f}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("=" * 70)
    print("1. Backpropagation computes gradients using the chain rule")
    print("2. Gradients tell us how to adjust parameters to reduce loss")
    print("3. We update parameters by moving in the opposite direction of gradients")
    print("4. Repeating this process (forward → backward → update) trains the network")
    print("5. As training progresses, gradients become smaller (we're near optimal)")
    print("=" * 70)

