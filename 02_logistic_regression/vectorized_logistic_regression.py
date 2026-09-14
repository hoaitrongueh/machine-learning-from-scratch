
import numpy as np
songs = [
    [0.9, 0.4, 0.1],
    [0.8, 0.7, 0.1],
    [0.7, 0.3, 0.2],
    [0.2, 0.8, 0.9],
    [0.4, 0.5, 0.8],
    [0.3, 0.2, 0.7],
]

targets = [1, 1, 1, 0, 0, 0]

learning_rate=0.1


more_songs = [
    [0.85, 0.45, 0.75],
    [0.95, 0.55, 0.90],
    [0.75, 0.65, 0.70],
]

more_targets = [1, 1, 1]

songs += more_songs
targets += more_targets




features=np.array(songs,dtype=float)
targets=np.array(targets,dtype=float)
weights=np.zeros(features.shape[1])
bias=0


logits=features @ weights + bias
def sigmoid(logits):
    return 1/(1+np.exp(-logits))
probabilities=sigmoid(logits)
print(probabilities)
print(logits)

errors=probabilities-targets
print(errors)
weight_gradients = features.T @ errors / len(targets)
print(weight_gradients)

bias_gradient = np.mean(errors)
print(bias_gradient)

weights-=learning_rate*weight_gradients
bias-=learning_rate*bias_gradient

print(weights,bias)