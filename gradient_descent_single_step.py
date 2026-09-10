features = [0.8, 0.6, 0.2]
weights = [8.0, 1.0, -1.0]
target = 6.0
learning_rate = 0.05

def predict(features,weights):
    prediction=0
    for i in range(len(weights)):
        prediction+=features[i]*weights[i]
    return prediction
  
prediction=predict(features,weights)
error=prediction-target
loss=error**2

gradient=[]
for i in range(len(features)):
    gradient_i=2*error*features[i]
    gradient.append(gradient_i)
print(gradient)

new_weights=[]
for i in range(len(weights)):
    new_weight_i=weights[i]-learning_rate*gradient[i]
    new_weights.append(new_weight_i)
print(new_weights)
weights=new_weights

new_prediction=predict(features,new_weights)
print(new_prediction)