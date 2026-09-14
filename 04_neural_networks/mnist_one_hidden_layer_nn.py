"""MNIST classifier: 784 inputs, 128 ReLU units, and 10 softmax outputs."""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

x,y=fetch_openml("mnist_784",version=1,return_X_y=True,as_frame=False)

y=y.astype(int)
x=(x/255).astype(np.float32)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

rng=np.random.default_rng(42)

weights_1=rng.normal(0,0.01,size=(784,128))
bias_1=np.zeros(128)

weights_2=rng.normal(0,0.01,size=(128,10))
bias_2=np.zeros(10)

learning_rate=0.01

def relu(logits):
    return np.maximum(0,logits)

def logits_calc(x,weights,bias):
    return x @ weights + bias

def softmax(logits):
    shifted_logits=logits-np.max(logits,axis=1,keepdims=True)
    logits_exp=np.exp(shifted_logits)
    probabilities=logits_exp/np.sum(logits_exp,axis=1,keepdims=True)
    return probabilities

def loss_calc(probabilities,targets):
    correct_probabilities=probabilities[np.arange(len(targets)),targets]
    correct_probabilities=np.clip(correct_probabilities,1e-15,1)
    loss=-np.log(correct_probabilities)
    return np.mean(loss)


batch_size=256
for epoch in range(100):
    indices=rng.permutation(len(x_train))
    x_shuffled=x_train[indices]
    y_shuffled=y_train[indices]
    epoch_loss=0
    for start in range(0,len(x_train),batch_size):
        end=start+batch_size

        x_batch=x_shuffled[start:end]
        y_batch=y_shuffled[start:end]

        hidden_logits=logits_calc(x_batch,weights_1,bias_1)
        hidden_activations=relu(hidden_logits)
        output_logits=logits_calc(hidden_activations,weights_2,bias_2)
        probabilities=softmax(output_logits)

        output_errors=probabilities.copy()
        output_errors[np.arange(len(y_batch)),y_batch]-=1

        loss=loss_calc(probabilities,y_batch)

        epoch_loss+=loss * len(x_batch)

        weights_2_gradients=(hidden_activations.T @ output_errors)/len(x_batch)
        bias_2_gradients=np.mean(output_errors,axis=0)

        hidden_errors=output_errors @ weights_2.T
        hidden_errors[hidden_logits<=0]=0

        weight_1_gradients=(x_batch.T @ hidden_errors)/len(x_batch)
        bias_1_gradients=np.mean(hidden_errors,axis=0)

        weights_1-=learning_rate * weight_1_gradients
        weights_2-=learning_rate * weights_2_gradients
        bias_1-=learning_rate*bias_1_gradients
        bias_2-=learning_rate*bias_2_gradients

    epoch_loss/=len(x_train)

    if epoch%5==0:
        print('Epoch:',epoch,'Loss:',epoch_loss)


train_hidden = relu(logits_calc(x_train, weights_1, bias_1))

train_logits = logits_calc(train_hidden,weights_2,bias_2)

train_predictions = np.argmax(
softmax(train_logits),axis=1)
test_hidden = relu(logits_calc(x_test, weights_1, bias_1))

test_logits = logits_calc(test_hidden,weights_2,bias_2)

test_predictions = np.argmax(softmax(test_logits),axis=1)

print("Train accuracy:",np.mean(train_predictions == y_train))

print("Test accuracy:",np.mean(test_predictions == y_test))
