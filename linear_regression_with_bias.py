dataset_features = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
]
targets = [7.0, 5.0, 10.0, 6.0]

weights = [0.0, 0.0] 

bias=0

learning_rate=0.5

def predict(features,weights):
    prediction=0
    for i in range(len(weights)):
        prediction+=features[i]*weights[i]
    prediction+=bias
    return prediction
        

for epoch in range(99):
    gradient_sums=[0,0]
    total_loss=0
    bias_gradient_real=0
    for i in range(len(dataset_features)):
        current_features=dataset_features[i]
        current_target=targets[i]
        prediction=predict(current_features,weights)
        error=prediction-current_target
        loss=error**2
        total_loss+=loss

        gradients=[]
        for i in range(len(weights)):
                    gradients_i=2*error*current_features[i]
                    gradients.append(gradients_i)
                    gradient_sums[i]+=gradients_i
                    bias_gradient=2*error
        bias_gradient_real+=bias_gradient

                
        
        average_loss=total_loss/len(dataset_features)
        average_gradient=[x / len(dataset_features) for x in gradient_sums]
        average_bias_gradient=bias_gradient_real/len(dataset_features)

        
        new_weights=[]
    for i in range(len(weights)):
        new_weight_i=weights[i]-learning_rate*average_gradient[i]
        new_weights.append(new_weight_i)

        new_bias=bias - learning_rate * average_bias_gradient
    bias=new_bias
    weights=new_weights

        
print("Final weights:", weights)
print("Final bias:", bias)
   

