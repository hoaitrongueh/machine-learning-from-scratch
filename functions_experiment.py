import numpy as np

probabilities = np.array([0.12, 0.76, 0.48, 0.91, 0.31, 0.67])
targets = np.array([0, 1, 1, 1, 0, 0])

def binary_converter(probabilities,threshold=0.5):
    tf=probabilities>=threshold
    predictions=tf.astype(int)
    return predictions

def accuracy(predictions,targets):
    accuracy=(np.mean(predictions==targets))*100
    return accuracy

def find_mistakes(probabilities,targets,predictions,confidence_list):
    mistake_location={}
    for i in range(len(probabilities)):
        if predictions[i]!=targets[i]:
            mistake_location[i]={'Prediction':int(predictions[i]),'Target':int(targets[i]),'Probability':float(probabilities[i]),'Confidence':confidence_list[i]}
    return mistake_location

def confidence_evaluation(probabilities):
    confidence_list=[]
    for probability in probabilities:
        if 0.4<=probability<=0.6:
            confidence_list.append('Low')
        elif 0.2<=probability<=0.4 or 0.6<=probability<=0.8:
            confidence_list.append('Medium')
        else:
            confidence_list.append('High')
    return confidence_list

def prediction_counter(predictions):
    positives=0
    negatives=0
    for prediction in predictions:
        if prediction==0:
            negatives+=1
        else:
            positives+=1
    return positives,negatives

def average_confidence(probabilities):
    average_confidence = 0
    for probability in probabilities:
        average_confidence+=(abs(probability-0.5))/len(probabilities)
    return average_confidence

def most_uncertain_probabilities(probabilities):
    index=[]
    uncertain_probabilities=[]
    for probability in probabilities:
        value=abs(probability-0.5)
        index.append(float(value))
    value_sorted=sorted(index)[:3]
    for distance,probability in zip(index,probabilities):
        if distance in value_sorted:
            uncertain_probabilities.append(float(probability))
    return uncertain_probabilities
            



def most_confident(probabilities):
    index=[]
    most_confident=[]
    for probability in probabilities:
        value=abs(probability-0.5)
        index.append(float(value))
    sorted_value=sorted(index,reverse=True)[:3]
    for distance,probability in index,probabilities:
        if distance in zip(sorted_value,probabilities):
            most_confident.append(float(probability))
    return most_confident

def false_diagnosis_counter(predictions,targets):
    false_positives_counter=0
    false_negatives_counter=0
    for prediction, target in zip(predictions, targets):
        if prediction==1 and target==0:
            false_positives_counter+=1
        if prediction==0 and target==1:
            false_negatives_counter+=1
    return false_negatives_counter,false_positives_counter
predictions = binary_converter(probabilities)
confidence_list=confidence_evaluation(probabilities)
print("Predictions:", predictions)
print("Accuracy:", accuracy(predictions, targets))
print("Mistakes:", find_mistakes(probabilities,targets,predictions,confidence_list))


print('Predictions:',predictions)
print('Positives:',prediction_counter(predictions)[0])
print('Negatives:',prediction_counter(predictions)[1])
print('Most uncertain:',most_uncertain_probabilities(probabilities))