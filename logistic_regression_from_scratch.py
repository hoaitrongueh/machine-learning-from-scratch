import math
songs = [
    [0.9, 0.4, 0.1],
    [0.8, 0.7, 0.1],
    [0.7, 0.3, 0.2],
    [0.2, 0.8, 0.9],
    [0.4, 0.5, 0.8],
    [0.3, 0.2, 0.7],
]

targets = [1, 1, 1, 0, 0, 0]

weights = [0.0, 0.0, 0.0]
bias = 0.0
learning_rate=0.1


more_songs = [
    [0.85, 0.45, 0.75],
    [0.95, 0.55, 0.90],
    [0.75, 0.65, 0.70],
]

more_targets = [1, 1, 1]

songs += more_songs
targets += more_targets








def logit_calc(features, weights,bias):
    logit=0
    for i in range(len(features)):
        logit+=features[i]*weights[i]
    return logit+bias

def sigmoid(logit):
    return 1/(1+math.exp(-logit))

def loss_calc(target, probability):
    return -(target * math.log(probability)
             + (1 - target) * math.log(1 - probability))
def classify(probability):
    if probability>=0.5:
        return 1
    else:
        return 0
def predict(features,weights,bias):
    logit=logit_calc(features,weights,bias)
    probability=sigmoid(logit)
    prediction=classify(probability)
    return probability, prediction

def evaluate(songs, targets, weights, bias):
    total_loss=0
    correct_prediction=0
    for i in range(len(songs)):
        features=songs[i]
        target=targets[i]
        probability,prediction=predict(features,weights,bias)
        if prediction==target:
            correct_prediction+=1
       

            
        
       
        loss=loss_calc(target,probability)
        total_loss+=loss
    return total_loss/len(songs),correct_prediction/len(targets)
        




for epoch in range(1000):
    loss=0
    weight_gradients_sums=[0,0,0]
    
    bias_gradient=0
    for i in range(len(targets)):
        features=songs[i]
        logit=logit_calc(features,weights,bias)
        probability=sigmoid(logit)
        loss += loss_calc(targets[i], probability)

    
        error_signal=probability-targets[i]
        
        for yo in range(len(weights)):
            weight_gradient_i=features[yo]*error_signal
            
            weight_gradients_sums[yo]+=weight_gradient_i

        bias_gradient+=error_signal
    average_weight_gradients=[x/len(targets) for x in weight_gradients_sums]
    average_bias_gradient=bias_gradient/len(targets)
    average_loss=loss/len(targets)
    new_weights=[]
    for i in range(len(weights)):
        new_weight_i=weights[i]-learning_rate*average_weight_gradients[i]
        new_weights.append(new_weight_i)

    new_bias=bias-learning_rate*average_bias_gradient
    weights=new_weights
    bias=new_bias
    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", average_loss)

   
   

print("Final weights:", weights)
print("Final bias:", bias)
test_songs = [
    [0.85, 0.6, 0.15],
    [0.15, 0.6, 0.85],
    [0.90, 0.4, 0.80],  # unusual combination
    [0.10, 0.4, 0.15],
]

test_targets = [1, 0, 1, 0]
for song_index in range(len(test_songs)):
    features = test_songs[song_index]

    probability, prediction = predict(features, weights, bias)

    print(
        "Target:", test_targets[song_index],
        "Probability:", probability,
        "Prediction:", prediction
    )

evaluate_training_data=evaluate(songs,targets,weights,bias)
evaluate_validation_data=evaluate(test_songs,test_targets,weights,bias)

print(evaluate_training_data,evaluate_validation_data)