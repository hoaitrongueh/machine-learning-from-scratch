import math
features = [-2.0, -1.0, 1.0, 2.0]
targets  = [ 0.0,  0.0, 1.0, 1.0]
weight=0
learning_rate=0.1
bias=0

def logit_calc(features,weight,bias):
    return features*weight+bias


def sigmoid(logit):
    return 1/(1+math.exp(-logit))

def loss(probability,targets):
    return -(targets*math.log(probability)+(1-targets)*math.log(1-probability))
for epoch in range(100):
    weight_gradient_sum=0
    bias_gradient_sum=0
    total_loss=0
    for i in range(len(features)):
        logit=logit_calc(features[i],weight,bias)
        probability=sigmoid(logit)
        total_loss+=loss(probability,targets[i])

        error_signal=probability-targets[i]

        weight_gradient=error_signal*features[i]
        bias_gradient=error_signal
        weight_gradient_sum+=weight_gradient
        bias_gradient_sum+=bias_gradient
    average_weight_gradient=weight_gradient_sum/len(features)
    average_bias_gradient=bias_gradient_sum/len(features)
    new_weight=weight-learning_rate*average_weight_gradient
    new_bias=bias-learning_rate*average_bias_gradient
    weight=new_weight
    bias=new_bias
    print(epoch,bias,weight)

for feature, target in zip(features, targets):
    probability = sigmoid(logit_calc(feature, weight, bias))
    predicted_class = 1 if probability >= 0.5 else 0
    print(feature, probability, predicted_class, target)