from math import log
import csv


def calc_entropy(data):
    samples = len(data)
    labels = {}
    for rec in data:
        label = rec[-1]
        if label not in labels .keys():
            labels[label] = 0
        labels[label] += 1
    entropy = 0.0
    for key in labels:
        prob = float(labels[key])/samples
        entropy -= prob * log(prob, 4)
    return entropy


def attribute_selection(data):
    features = len(data[0]) - 1
    base_entropy = calc_entropy(data)
    max_info_gain = 0.0;
    best_attr = -1
    for i in range(features):
        attr_list = [rec[i] for rec in data]
        unique_vals = set(attr_list)
        new_entropy = 0.0
        attr_entropy = 0.0
        for value in unique_vals:
            new_data = dataset_split(data, i, value)
            prob = len(new_data)/float(len(data))
            new_entropy = prob * calc_entropy(new_data)
            attr_entropy += new_entropy
        info_gain = base_entropy - attr_entropy
        if (info_gain > max_info_gain):
            max_info_gain = info_gain
            best_attr = i
            entropy = attr_entropy
    return best_attr,entropy


def dataset_split(data, arc, val):
    split_data = []
    for rec in data:
        if rec[arc] == val:
            reduced_set = list(rec[:arc])
            reduced_set.extend(rec[arc+1:])
            split_data.append(reduced_set)
    return split_data

# Function to build the decision tree
def main(data, labels) :
    # list variable to store the class-labels (terminal nodes of decision tree)
    classList = [rec[-1] for rec in data]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # functional call to identify the attribute for split
    maxGainNode, entropy = attribute_selection(data)
    # variable to store the class label value
    treeLabel = labels[maxGainNode]
    # dict object to represent the nodes in the decision tree
    theTree = {treeLabel:{}}
    del(labels[maxGainNode])
    # get the unique values of the attribute identified
    nodeValues = [rec[maxGainNode] for rec in data]
    uniqueVals = set(nodeValues)
    for value in uniqueVals:
        subLabels = labels[:]
        # update the non-terminal node values of the decision tree
        data_set1 = dataset_split(data,maxGainNode,value)
        entropy1 = calc_entropy(data_set1)
        print(treeLabel,value,entropy)
        theTree[treeLabel][value] = main(dataset_split(data, maxGainNode, value),subLabels)
    #return the decision tree (dict object)
    return theTree

def readcsv():
    data = list(csv.reader(open('car.csv')))
    return data

import pprint
labels = ['att0','att1','att2','att3','att4','att5']
data=readcsv()
pprint.pprint(main(data,labels))