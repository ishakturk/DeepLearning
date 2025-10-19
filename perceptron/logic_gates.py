import numpy as np


class Perceptron:
    def __init__(self, input_dim, learning_rate=0.1, n_epochs=20):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        # Initialize weights randomly to break symmetry
        self.weights = np.random.randn(input_dim)
        self.bias = 0.0

    def activation(self, x):
        """Step function."""
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

    def fit(self, X, y):
        print(f"--- Training for {self.n_epochs} epochs ---")
        for epoch in range(self.n_epochs):
            for xi, target in zip(X, y):
                # Make a prediction
                y_pred = self.predict(xi)
                # Calculate the update value (error * learning_rate)
                update = self.lr * (target - y_pred)
                # Update weights and bias
                self.weights += update * xi
                self.bias += update

            # Optional: print accuracy at each epoch to see progress
            acc = np.mean(self.predict(X) == y)
            print(f"Epoch {epoch + 1}/{self.n_epochs}, Accuracy: {acc:.2f}")
        print("--- Training complete ---")


# Example usage for different logic gates
if __name__ == "__main__":

    # --- AND Gate ---
    # Linearly Separable: Should succeed
    print("\n*** TESTING AND GATE ***")
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    perceptron_and = Perceptron(input_dim=2)
    perceptron_and.fit(X_and, y_and)
    print("Final Predictions for AND:")
    for xi in X_and:
        print(f"{xi} -> {perceptron_and.predict(xi)}")

    # --- OR Gate ---
    # Linearly Separable: Should succeed
    print("\n*** TESTING OR GATE ***")
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    perceptron_or = Perceptron(input_dim=2)
    perceptron_or.fit(X_or, y_or)
    print("Final Predictions for OR:")
    for xi in X_or:
        print(f"{xi} -> {perceptron_or.predict(xi)}")

    # --- XOR Gate ---
    # NOT Linearly Separable: Should fail
    print("\n*** TESTING XOR GATE ***")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    perceptron_xor = Perceptron(input_dim=2, n_epochs=100)  # More epochs to show it still won't work
    perceptron_xor.fit(X_xor, y_xor)
    print("Final Predictions for XOR:")
    for xi in X_xor:
        print(f"{xi} -> {perceptron_xor.predict(xi)}")