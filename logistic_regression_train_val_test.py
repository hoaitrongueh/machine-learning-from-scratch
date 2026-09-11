import numpy as np

features = np.array([
    [180, 0.2],
    [240, 0.8],
    [300, 0.3],
    [210, 0.9],
    [270, 0.4],
    [160, 0.7],
    [320, 0.2],
    [230, 0.6],
    [190, 0.3],
    [280, 0.9],
    [250, 0.5],
    [170, 0.8],
    [310, 0.4],
    [220, 0.7],
    [200, 0.1],
], dtype=float)

targets = np.array([
    0, 1, 0, 1, 0,
    1, 0, 1, 0, 1,
    0, 1, 0, 1, 0,
], dtype=float)


def sigmoid(logits):
    return 1/(1+np.exp(-logits))

def loss_calc(probabilities,targets):
    return -(targets*np.log(probabilities)+(1-targets)*np.log(1-probabilities))

def binary_converter(probabilities):
    binary_list = []

    for probability in probabilities:
        binary = int(probability >= 0.5)
        binary_list.append(binary)

    return np.array(binary_list)

def evaluate(probabilities,targets):
    converted=binary_converter(probabilities)
    accuracy=np.mean(converted==targets)
    average_loss=np.mean(loss_calc(probabilities,targets))
    return average_loss,accuracy
def predict_proba(scaled_features, weights, bias):
    logits=scaled_features @ weights + bias
    probabilities=sigmoid(logits)
    return probabilities
        




features_shape=np.shape(features)
targets_shape=np.shape(targets)

weights=np.zeros(features.shape[1])
bias=0
learning_rate=0.1

rng=np.random.default_rng(42)
indices=rng.permutation(len(features))

features=features[indices]
targets=targets[indices]

train_features=features[:9]
train_targets=targets[:9]

val_features=features[9:12]
val_targets=targets[9:12]

test_features=features[12:]
test_targets=targets[12:]

means=train_features.mean(axis=0)
stds=train_features.std(axis=0)

train_features_scaled=(train_features-means)/stds
test_features_scaled=(test_features-means)/stds
val_features_scaled=(val_features-means)/stds

for epoch in range(1000):
    probabilities = predict_proba(train_features_scaled, weights, bias)
    average_loss=np.mean(loss_calc(probabilities,train_targets))
    errors=probabilities-train_targets

    val_logits=val_features_scaled @ weights + bias
    val_probabilities=sigmoid(val_logits)
    average_val_loss=np.mean(loss_calc(val_probabilities,val_targets))

    weight_gradients=(train_features_scaled.T @ errors)/len(train_features_scaled)
    bias_gradient=np.mean(errors)
    weights-=learning_rate*weight_gradients
    bias-=learning_rate*bias_gradient

   
    
    if epoch % 100==0:
        print("Epoch:",epoch,"| Average val loss:",average_val_loss,"| Average loss",average_loss)

val_probabilities = predict_proba(val_features_scaled,weights,bias)



test_probabilities=predict_proba(test_features_scaled,weights,bias)

test_loss, test_accuracy = evaluate(test_probabilities, test_targets)
val_loss,val_accuracy=evaluate(val_probabilities,val_targets)

print("Average test loss",test_loss)
print("Average_val_loss",val_loss)
print(f"Val accuracy: {val_accuracy:.0%}")
print(f"Test accuracy: {test_accuracy:.0%}")


print("New weights:",weights)
print("New bias:",bias)



