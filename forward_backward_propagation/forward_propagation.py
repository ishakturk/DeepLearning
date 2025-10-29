"""
Forward Propagation Implementation
===================================
This script demonstrates forward propagation in a simple neural network.
Forward propagation is the process of passing input data through the network
to produce an output prediction.
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
# ACTIVATION FUNCTIONS
# ============================================================================

def sigmoid(z):
    """
    Sigmoid activation function: Maps any value to a range between 0 and 1.

    Formula: σ(z) = 1 / (1 + e^(-z))

    Args:
        z: Input value or array (can be linear combination w*x + b)

    Returns:
        Output value(s) between 0 and 1

    Use case: Binary classification, output layer for probabilities
    """
    return 1.0 / (1.0 + np.exp(-z))


def relu(z):
    """
    ReLU (Rectified Linear Unit) activation function.

    Formula: ReLU(z) = max(0, z)

    Args:
        z: Input value or array

    Returns:
        Input if positive, 0 if negative

    Use case: Hidden layers in deep networks (prevents vanishing gradient)
    """
    return np.maximum(0, z)


def tanh(z):
    """
    Hyperbolic tangent activation function: Maps values to range between -1 and 1.

    Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

    Args:
        z: Input value or array

    Returns:
        Output value(s) between -1 and 1

    Use case: Hidden layers when you need outputs centered around 0
    """
    return np.tanh(z)


# ============================================================================
# FORWARD PROPAGATION FUNCTIONS
# ============================================================================

def forward_propagation_single_layer(X, W, b, activation='sigmoid'):
    """
    Perform forward propagation for a single-layer neural network.

    Architecture: Input → Single Layer → Output

    Steps:
        1. Compute linear transformation: Z = X * W + b
        2. Apply activation function: A = activation(Z)

    Args:
        X: Input data, shape (n_samples, n_features)
        W: Weight matrix, shape (n_features, n_outputs)
        b: Bias vector, shape (n_outputs,)
        activation: Activation function name ('sigmoid', 'relu', 'tanh')

    Returns:
        A: Activated output, shape (n_samples, n_outputs)
        Z: Linear output (before activation), shape (n_samples, n_outputs)

    Example:
        If X has 3 samples with 2 features each, and we want 1 output:
        X shape: (3, 2)
        W shape: (2, 1)
        b shape: (1,)
        Output A shape: (3, 1) - one prediction per sample
    """

    # Step 1: Linear transformation (weighted sum + bias)
    # Z = X @ W + b
    # This computes: z = w1*x1 + w2*x2 + ... + wn*xn + b for each sample
    Z = np.dot(X, W) + b

    # Step 2: Apply non-linear activation function
    # This introduces non-linearity, allowing the network to learn complex patterns
    if activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'relu':
        A = relu(Z)
    elif activation == 'tanh':
        A = tanh(Z)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return A, Z


def forward_propagation_two_layer(X, W1, b1, W2, b2,
                                  hidden_activation='relu',
                                  output_activation='sigmoid'):
    """
    Perform forward propagation for a two-layer (one hidden layer) neural network.

    Architecture: Input → Hidden Layer → Output Layer

    Steps:
        1. First layer: Compute Z1 = X * W1 + b1, then A1 = activation(Z1)
        2. Second layer: Compute Z2 = A1 * W2 + b2, then A2 = activation(Z2)

    Args:
        X: Input data, shape (n_samples, n_features)
        W1: First layer weights, shape (n_features, n_hidden)
        b1: First layer bias, shape (n_hidden,)
        W2: Second layer weights, shape (n_hidden, n_outputs)
        b2: Second layer bias, shape (n_outputs,)
        hidden_activation: Activation for hidden layer (typically 'relu' or 'tanh')
        output_activation: Activation for output layer (typically 'sigmoid')

    Returns:
        cache: Dictionary containing all intermediate values needed for backprop
            - A1: Hidden layer activations
            - Z1: Hidden layer linear outputs
            - A2: Final output activations
            - Z2: Output layer linear outputs
    """

    # ===== HIDDEN LAYER (Layer 1) =====
    # Linear transformation: combine inputs with weights
    Z1 = np.dot(X, W1) + b1

    # Non-linear activation: allows network to learn complex patterns
    if hidden_activation == 'relu':
        A1 = relu(Z1)
    elif hidden_activation == 'tanh':
        A1 = tanh(Z1)
    elif hidden_activation == 'sigmoid':
        A1 = sigmoid(Z1)
    else:
        raise ValueError(f"Unknown hidden activation: {hidden_activation}")

    # ===== OUTPUT LAYER (Layer 2) =====
    # Use hidden layer output (A1) as input to the next layer
    Z2 = np.dot(A1, W2) + b2

    # Final activation: typically sigmoid for binary classification
    if output_activation == 'sigmoid':
        A2 = sigmoid(Z2)
    elif output_activation == 'relu':
        A2 = relu(Z2)
    elif output_activation == 'tanh':
        A2 = tanh(Z2)
    else:
        raise ValueError(f"Unknown output activation: {output_activation}")

    # Store all values in a cache for later use in backpropagation
    cache = {
        'Z1': Z1,  # Hidden layer linear output
        'A1': A1,  # Hidden layer activation output
        'Z2': Z2,  # Output layer linear output
        'A2': A2   # Final prediction (output layer activation)
    }

    return cache


def compute_loss(Y_pred, Y_true):
    """
    Compute the loss (error) between predictions and true values.

    Uses Mean Squared Error (MSE) loss function.
    Formula: MSE = (1/n) * Σ(y_pred - y_true)²

    Args:
        Y_pred: Predicted values, shape (n_samples, n_outputs)
        Y_true: True values, shape (n_samples, n_outputs)

    Returns:
        loss: Scalar value representing the average error

    Note: Lower loss means better predictions
    """
    n_samples = Y_true.shape[0]

    # Compute squared differences and average them
    loss = np.sum((Y_pred - Y_true) ** 2) / (2 * n_samples)

    return loss


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FORWARD PROPAGATION DEMONSTRATION")
    print("=" * 70)

    # ===== SINGLE LAYER EXAMPLE =====
    print("\n1. SINGLE LAYER NETWORK")
    print("-" * 70)

    # Initialize random weights and bias
    np.random.seed(42)  # For reproducible results
    n_features = X.shape[1]  # 2 features
    n_outputs = 1            # 1 output

    W = np.random.randn(n_features, n_outputs) * 0.5
    b = np.zeros((n_outputs,))

    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")
    print(f"Bias shape: {b.shape}")
    print(f"\nInput data:\n{X}")
    print(f"\nWeights:\n{W}")
    print(f"Bias: {b}")

    # Perform forward propagation
    A, Z = forward_propagation_single_layer(X, W, b, activation='sigmoid')

    print(f"\nLinear output (Z):\n{Z}")
    print(f"\nActivated output (A - Predictions):\n{A}")
    print(f"\nTrue values (Y):\n{Y}")

    # Compute loss
    loss = compute_loss(A, Y)
    print(f"\nLoss (Mean Squared Error): {loss:.6f}")

    # ===== TWO LAYER EXAMPLE =====
    print("\n\n2. TWO LAYER NETWORK (WITH HIDDEN LAYER)")
    print("-" * 70)

    # Initialize random weights and biases for two layers
    np.random.seed(42)
    n_features = X.shape[1]  # 2 input features
    n_hidden = 4             # 4 neurons in hidden layer
    n_outputs = 1            # 1 output

    W1 = np.random.randn(n_features, n_hidden) * 0.5
    b1 = np.zeros((n_hidden,))
    W2 = np.random.randn(n_hidden, n_outputs) * 0.5
    b2 = np.zeros((n_outputs,))

    print(f"Input shape: {X.shape}")
    print(f"Hidden layer: {n_hidden} neurons")
    print(f"Output shape: {n_outputs}")
    print(f"\nLayer 1 weights shape: {W1.shape}")
    print(f"Layer 1 bias shape: {b1.shape}")
    print(f"Layer 2 weights shape: {W2.shape}")
    print(f"Layer 2 bias shape: {b2.shape}")

    # Perform forward propagation through both layers
    cache = forward_propagation_two_layer(X, W1, b1, W2, b2,
                                         hidden_activation='relu',
                                         output_activation='sigmoid')

    print(f"\nHidden layer output (A1) shape: {cache['A1'].shape}")
    print(f"Hidden layer activations:\n{cache['A1']}")
    print(f"\nFinal predictions (A2):\n{cache['A2']}")
    print(f"\nTrue values (Y):\n{Y}")

    # Compute loss
    loss = compute_loss(cache['A2'], Y)
    print(f"\nLoss (Mean Squared Error): {loss:.6f}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("Forward propagation takes input data and passes it through")
    print("the network layers to produce predictions. The loss measures")
    print("how far our predictions are from the true values.")
    print("=" * 70)

