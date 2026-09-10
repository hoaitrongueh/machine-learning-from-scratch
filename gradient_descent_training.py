
features =[0.8,0.6,0.2]
weights=[8.0,1.0,-1.0]
target=6.0
learning_rate=0.05

for step in range(6):

    def predict(features,weights):
        prediction=0
        for i in range(len(features)):
            prediction+=(features[i]*weights[i])
        return prediction


    prediction=predict(features,weights)
    error=prediction - target
    loss=error**2

        
    gradients=[]
    for i in range(len(weights)):
        gradient_i=2*error*features[i]
        gradients.append(gradient_i)

    

    new_weights=[]
    for i in range(len(weights)):
        new_weight=weights[i]-learning_rate*gradients[i]
        new_weights.append(new_weight)

    weights=new_weights
    print(prediction)






                    