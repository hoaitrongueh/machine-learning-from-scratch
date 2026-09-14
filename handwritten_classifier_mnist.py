import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

x,y=fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False
)

y=y.astype(int)
x=x/255

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

weights=np.zeros((x.shape[1],10))
bias=np.zeros(10)
learning_rate=0.1

def logits_calc(x,weights,bias):
    return x @ weights + bias

def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    exp_logits=np.exp(shifted_logits)
    probabilities=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
    return probabilities

def loss_calc(probabilities,targets):
    correct_probabilities=probabilities[np.arange(len(targets)),targets]
    losses=-np.log(correct_probabilities)
    return np.mean(losses)

for epoch in range(1000):
    logits=logits_calc(x_train,weights,bias)
    probabilities=softmax(logits)
    loss=loss_calc(probabilities,y_train)
    errors=probabilities.copy()
    errors[np.arange(len(y_train)),y_train]-=1
    weight_gradients=(x_train.T @ errors)/len(x_train)
    bias_gradients=np.mean(errors,axis=0)

    weights-=learning_rate * weight_gradients
    bias-=learning_rate * bias_gradients

    if epoch % 100==0:
        print("Epoch:", epoch, "Loss:", loss)

test_logits=logits_calc(x_test,weights,bias)
test_probabilities=softmax(test_logits)

train_predictions = np.argmax(softmax(logits_calc(x_train, weights, bias)),axis=1)

test_predictions = np.argmax(softmax(logits_calc(x_test, weights, bias)),axis=1)

train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)