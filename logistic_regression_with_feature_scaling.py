import numpy as np

features = np.array([
    [180, 0.2],
    [240, 0.8],
    [300, 0.3],
    [210, 0.9],
    [270, 0.4],
    [160, 0.7],
    [320, 0.2],
    [200, 0.6],
    [250, 0.3],
    [290, 0.8],
], dtype=float)

targets = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)

learning_rate=0.1

features_shape=np.shape(features)
targets_shape=np.shape(targets)

weights=np.zeros(features_shape[1])
bias=0
def loss_calc(probabilities,targets):
    return -(targets*np.log(probabilities)+(1-targets)*(np.log(1-probabilities)))
rng=np.random.default_rng(42)
indices=rng.permutation(len(features))

split=int(len(features)*0.8)

train_indices=indices[:split]
test_indices=indices[split:]

train_features=features[train_indices]
train_targets=targets[train_indices]

test_features=features[test_indices]
test_targets=targets[test_indices]

means=np.mean(train_features,axis=0)
stds=np.std(train_features,axis=0)

scaled_train=(train_features-means)/stds
scaled_test=(test_features-means)/stds
for epoch in range (1000):
    logits=scaled_train @ weights + bias
    probabilities=1/(1+np.exp(-logits))


    loss=np.mean(loss_calc(probabilities,train_targets))

    errors=probabilities-train_targets    

    weight_gradients=scaled_train.T @ errors/len(scaled_train)
    bias_gradient=np.mean(errors)
    weights-=learning_rate*weight_gradients
    bias-=learning_rate*bias_gradient

test_logits=scaled_test @ weights + bias
test_probabilities=1/(1+np.exp(-test_logits))

tf=test_probabilities>=0.5
nb=tf.astype(int)

accuracy = np.mean(nb == test_targets)
print(f"Test accuracy: {accuracy:.0%}")

print("Predictions:", nb)
print("Actual:", test_targets)

print("Weights:", weights)
print("Bias:", bias)