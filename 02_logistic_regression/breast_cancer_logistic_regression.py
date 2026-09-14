import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
def logits_calc(features,weights,bias):
    return features @ weights +bias

def sigmoid(logits):
    return 1/(1+np.exp(-logits))

def loss_calc(probabilities,targets):
    return -(targets*np.log(probabilities)+(1-targets)*np.log(1-probabilities))

def evaluate(predictions,targets,probabilities):
    mistakes=0
    mistake_location={}
    for i in range(len(predictions)):
        if predictions[i]!=targets[i]:
            mistakes+=1
            mistake_location['Mistake',i]={
                "Prediction":int(predictions[i]),"Target":int(targets[i]),"Probability":float(probabilities[i])}
    return (mistakes/len(targets))*100,mistake_location

bias=0
learning_rate=0.1
data = load_breast_cancer()

x=data['data']
y=data['target']

weights=np.zeros(x.shape[1])

feature_names=data['feature_names']
target_names=data['target_names']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

means=np.mean(x_train,axis=0)
stds=np.std(x_train,axis=0)

train_features_scaled=(x_train-means)/stds
test_features_scaled=(x_test-means)/stds

for epoch in range(1000):
    logits=logits_calc(train_features_scaled,weights,bias)
    probabilities=sigmoid(logits)
    average_loss=np.mean(loss_calc(probabilities,y_train))

    errors=probabilities-y_train
    weight_gradients=(train_features_scaled.T @ errors)/len(train_features_scaled)
    bias_gradient=np.mean(errors)

    weights-=learning_rate*weight_gradients
    bias-=learning_rate*bias_gradient

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss = {average_loss}")


test_logits=logits_calc(test_features_scaled,weights,bias)
test_probabilities=sigmoid(test_logits)

test_predictions=(test_probabilities >= 0.5).astype(int)

accuracy=np.mean(test_predictions==y_test)

print("Test accuracy:",accuracy)

error_rate, mistakes = evaluate(test_predictions, y_test,test_probabilities)
print("Error rate:", error_rate)
print("Mistakes:", mistakes)