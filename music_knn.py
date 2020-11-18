from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np

# from tempfile import TemporaryFile
import os
import pickle
import random
import operator
# import math

# distance algorithm
def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1)
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance

# k nearest neighbors for given test instance
def get_neighbors(train_set, instance, k):
    distances = []
    for i in range(len(train_set)):
        dist = distance(train_set[i], instance, k) + distance(instance, train_set[i], k)
        distances.append((train_set[i][2], dist))
    distances.sort(key=operator.itemgetter(1))
    
    neighbors = []
    for j in range(k):
        neighbors.append(distances[j][0])
    return neighbors

# predict response based on neighbors
def predict_class(neighbors):
    class_vote = {}
    for i in range(len(neighbors)):
        response = neighbors[i]
        if response in class_vote:
            class_vote[response] += 1
        else:
            class_vote[response] = 1
    
    sorted_votes = sorted(class_vote.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]

# basic accuracy calculation
def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct+=1
    return (1.0*correct)/len(test_set)

# extract features and save
directory = "genres"
f=open("my.dat", 'wb')
i=0

for folder in os.listdir(directory):
    i+=1
    if i==11:
        break
    for file in os.listdir(directory+"/"+folder):
        (rate,sig) = wav.read(directory+"/"+folder+"/"+file)
        mfcc_feat = mfcc(sig, rate, winlen=0.01, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)
        pickle.dump(feature, f)

f.close()

dataset = []
def load_dataset(filename, split, train_set, test_set):
    with open("my.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    
    for i in range(len(dataset)):
        if random.random() < split:
            train_set.append(dataset[i])
        else:
            test_set.append(dataset[i])

train_set = []
test_set = []
load_dataset("my.dat", 0.66, train_set, test_set)

predictions = []
for i in range(len(test_set)):
    predictions.append(predict_class(get_neighbors(train_set, test_set[i], 5)))

accuracy = get_accuracy(test_set, predictions)
print(accuracy)