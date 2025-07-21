import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

'''
First task is to add a Dense layer with 2 neurons and activation sigmoid.
Second task is to add a Dense layers with 1 neuron and activation sigmoid. 
'''

# XOR input and output
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Define the model
model = Sequential()
model.add()  # Hidden layer with 2 neurons
model.add()               # Output layer

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=5000, verbose=1)

# Evaluate the model
print("Predictions:")
predictions = model.predict(X)
for i, x in enumerate(X):
    print(f"Input: {x} => Output: {predictions[i][0]:.4f}")
