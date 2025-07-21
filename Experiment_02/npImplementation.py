import numpy as np

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

# XOR data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Seed and initialize weights
np.random.seed(42)
input_layer_neurons = X.shape[1]
hidden_neurons = 2
output_neurons = 1

# Weights and biases
wh = np.random.uniform(size=(input_layer_neurons, hidden_neurons))
bh = np.random.uniform(size=(1, hidden_neurons))
wo = np.random.uniform(size=(hidden_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))

# Training
epochs = 10000
lr = 0.1
for _ in range(epochs):
    # Forward Propagation
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)
    
    final_input = np.dot(hidden_output, wo) + bo
    final_output = sigmoid(final_input)
    
    # Backpropagation
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)
    
    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # Update weights and biases
    wo += hidden_output.T.dot(d_output) * lr
    bo += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Output after training
print("Final Output after training:")
print(final_output.round(3))
