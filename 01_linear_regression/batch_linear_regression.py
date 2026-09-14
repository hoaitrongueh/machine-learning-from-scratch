dataset_features = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0]
]

targets = [4.0, 2.0, -3.0, 6.0, 1.0]

weights = [0.0, 0.0, 0.0]

unseen_song = [0.8, 0.6, 0.2]   

learning_rate=0.5
def predict(features,weights):
    prediction=0
    for i in range(len(weights)):
        prediction+=features[i]*weights[i]
    return prediction

for epoch in range(99):
    total_loss=0
    gradient_sums=[0,0,0]


    for song_i in range(len(dataset_features)):
        current_features=dataset_features[song_i]
        current_target = targets[song_i]
        prediction=predict(current_features,weights)
        error = prediction-current_target
        loss = error**2
        total_loss+=loss
        current_gradient=[]
        for i in range(len(weights)):
            current_gradient_i=2*error*current_features[i]
            current_gradient.append(current_gradient_i)
            gradient_sums[i]+=current_gradient_i
        

    average_loss=total_loss/len(dataset_features)
    average_gradient=[x / len(dataset_features) for x in gradient_sums]

    new_weights=[]
    for i in range(len(weights)):
        new_weight_i=weights[i]-learning_rate*average_gradient[i]
        new_weights.append(new_weight_i)
    weights=new_weights
print(weights)

print(predict(unseen_song,weights))


