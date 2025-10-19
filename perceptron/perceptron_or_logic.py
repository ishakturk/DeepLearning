import numpy as np  # Import the NumPy library for numerical operations


class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01, n_epochs=100):
        # Initialize the perceptron with the number of input dimensions, learning rate, and number of epochs
        self.lr = learning_rate  # Learning rate for weight updates
        self.n_epochs = n_epochs  # Number of training epochs
        # Initialize weights with random values and bias with 0
        self.weights = np.random.randn(input_dim)  # Random weights for each input dimension
        self.bias = 0.0  # Bias term initialized to 0

    def activation(self, x):
        """Step function."""
        # Activation function: returns 1 if input >= 0, otherwise 0
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        # Compute the linear combination of inputs and weights, then add bias
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply the activation function to the linear output
        return self.activation(linear_output)

    def fit(self, X, y):
        # Train the perceptron using the input data (X) and labels (y)
        for epoch in range(self.n_epochs):  # Loop over the number of epochs
            for xi, target in zip(X, y):  # Loop over each training example and its corresponding label
                # Forward pass: compute the linear output
                linear_output = np.dot(xi, self.weights) + self.bias
                # Apply the activation function to get the predicted output
                y_pred = self.activation(linear_output)

                # Compute the update value based on the difference between target and prediction
                update = self.lr * (target - y_pred)
                # Update the weights using the update value and the input
                self.weights += update * xi
                # Update the bias using the update value
                self.bias += update
            # Optional: calculate and print the accuracy after each epoch
            get_pred = self.predict(X)
            acc = np.mean(get_pred == y)  # Calculate accuracy as the mean of correct predictions
            print(f"Epoch {epoch + 1}/{self.n_epochs}, Accuracy: {acc:.2f}")  # Print epoch number and accuracy


# Example usage: AND logic gate
if __name__ == "__main__":
    # Input features for the AND gate (binary inputs)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Corresponding labels for the AND gate
    y = np.array([0, 1, 1, 1])

    # Create a Perceptron instance with 2 input dimensions, learning rate of 0.1, and 50 epochs
    perceptron = Perceptron(input_dim=2, learning_rate=0.1, n_epochs=50)
    # Train the perceptron on the AND gate data
    perceptron.fit(X, y)

    # Print predictions after training
    print("Predictions after training:")
    for xi in X:  # Loop over each input example
        print(f"{xi} -> {perceptron.predict(xi)}")  # Print the input and its predicted output
