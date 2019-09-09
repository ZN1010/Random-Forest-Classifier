from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        self.tree = {}
        

    def learn(self, X, y):
        helper(self.tree, X, y)
        

    def classify(self, record):
        temp = self.tree
        while temp["rst"] == None:
            if record[temp["index"]] <= temp["split_val"]:
                temp = temp["left"]
            else:
                temp = temp["right"]
        return temp["rst"]


def checkEqual(iterator):
   return len(set(iterator)) <= 1

def majority(arr):
    return max(set(arr), key=arr.count)

def isDiscrete(arr):
    for element in arr:
        if element is not int:
            return False
    return True

def helper(tree, X, y):
    if checkEqual(y):
        if len(y) == 0:
            tree["rst"] = 1
        else:
            tree["rst"] = y[0]
        tree["index"] = None
    else:
        size = len(X[0])
        gain, index, val = 0, 0, 0
        X_left, X_right, y_left, y_right = [], [], [], []
        for i in range(size):
            X_i = [item[i] for item in X]
            if isDiscrete(X_i):
                m = majority(X_i)
            else:
                m = np.mean(X_i)
            X_l, X_r, y_l, y_r = partition_classes(X, y, i, m)
            temp = information_gain(y, [y_l, y_r])
            if temp > gain:
                index, gain, val = i, temp, m
                X_left, X_right, y_left, y_right = X_l, X_r, y_l, y_r
        tree["index"] = index
        tree["split_val"] = val
        tree["rst"] = None        
        tree["left"] = {}
        helper(tree["left"], X_left, y_left)
        tree["right"] = {}
        helper(tree["right"], X_right, y_right)


        
