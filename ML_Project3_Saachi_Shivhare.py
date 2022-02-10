#!/usr/bin/env python
# coding: utf-8

# In[250]:


import numpy as np
import random
import math
import operator
from collections import Counter


# In[251]:


# Read the data from file Feature data and label data

with open('iris.data') as f:
    read_data = f.read()

read_data    = read_data.split("\n")
temp_data    = []
label        = []
data         = read_data[0 : len(read_data) - 2]
feature_data = []

for line in data:
    line = line.split(",")
    for indx in range(4):
        line[indx] = float(line[indx])
    feature_data.append(line[0 : 4])
    temp_data.append(line[len(line) - 1])

# Converting string labels into intseger labels

for line in temp_data:
    if line == 'Iris-versicolor':
        label.append(1.0)
    elif line == 'Iris-virginica':
        label.append(2.0)
    else:
        label.append(0.0)

##################################################################################################

# Shuffle the data and labels

numbers = [i for i in range(150)]

random.seed(10)
random.shuffle(numbers)

# After the data is shuffled, store the data in processed_data and processed_label

processed_data  = []
processed_label = []

for i in range(150):
    processed_data.append(feature_data[numbers[i]])  
    processed_label.append(label[numbers[i]])  
      
# Split the processed data into 80% train and 20% test data

split1 = int(0.7 * len(feature_data))

train_data = processed_data[:split1]
test_data  = processed_data[split1:]

# Split the processed_label into 80% train and 20% test label

train_label = processed_label[:split1]
test_label  = processed_label[split1:]

print("Number of TRAINING samples are", len(train_data))
print("Number of TESTING samples are", len(test_data))


# In[252]:


#Function to calculate euclidean distance between two points

def euclideanDistance(point1, point2):
    distance = 0.0
    val = 0.0
    for i in range(len(point1)):
        val = pow((point1[i] - point2[i]), 2)
        #print("point1 point2 distance")
        #print(point1 , point2, val)
        distance += val
        #print("here")
    return math.sqrt(distance)


# In[253]:


#Function to calculate K nearest neighbours

def findNeighbors(train_data, test_data, k):
    
    neighbors_labels = np.zeros(k)
    value = np.zeros(len(train_data))
    (rows,cols) = (len(test_data),2)
    most_common_actual_label = [rows,cols]
    most_common_label = []
    idx = 0
    correct = 0
    accuracy = 0.0
    
    for j in range(len(test_data)):
        test_train_distance_list = []
        print("Calculating Euclidean Distance for Test data:", test_data[j])
        
#Sort the samples in ascending order according to the euclidean distance
        
        for i in range(len(train_data)):             
            value[i] = euclideanDistance(test_data[j], train_data[i])
            test_train_distance_list.append([test_data[j], train_data[i], value[i], test_label[j], train_label[i]])
            test_train_distance_list.sort(key = lambda x : x[2])
            
        print("--TEST DATA                 TRAIN DATA           EUC. DISTANCE         LABEL  PREDICTED LABEL--")
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in test_train_distance_list[:k]]))
        
#Predict label by considering k nearest neighbours
            
        for i in range(k):
            neighbors_labels[i] = test_train_distance_list[i][4]
        most_common_label = Counter(neighbors_labels).most_common(1)
        
        print("Considering K nearest neighbour", k)
        print("Most Common LABEL & its frequency:", most_common_label)
        print("\n")
        
#Compute number of correctly classified samples by compairing predicted labels and actual labels
        
        for i in range(len(test_label)):
            most_common_actual_label.append([most_common_label,test_label[idx]])
            if (most_common_label[0][0] == test_label[idx]):
                correct = correct + 1
            idx = idx + 1
            break                               
    
    most_common_actual_label.pop(0)
    most_common_actual_label.pop(0)  
    print("\n")
    
    print("+++++++++++++++++++++++++++++PREDICTED LABELS AND ACTUAL LABELS +++++++++++++++++++++++++++++")
    print("Most common labels with their frequency and actual labels for the training samples are displayed below:")
    print("\n")
    print("PREDICTED      ACTUAL")
    print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in most_common_actual_label]))
    
    print("\n")
    print("Number of correctly classified labels are :", correct)
   
# Compute accuracy of the model

    accuracy = correct/len(test_label)
    print("Accuracy of the model when k is set to", k ,"is given by", accuracy*100)
    
    return(neighbors_labels)               


# In[254]:


#KNN using K = 3

predicted_neigh_labels = findNeighbors(train_data,test_data, 3)


# In[255]:


#KNN using k = 5

predicted_neigh_labels = findNeighbors(train_data,test_data, 5)


# In[ ]:




