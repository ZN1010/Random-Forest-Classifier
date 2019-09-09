from decision_tree import DecisionTree
import csv
import numpy as np 
import ast


"""
Here, 
1. X is assumed to be a matrix with n rows and d columns where n is the
number of total records and d is the number of features of each record. 
2. y is assumed to be a vector of labels of length n.
3. XX is similar to X, except that XX also contains the data label for each
record.
"""



class RandomForest(object):
    num_trees = 0
    decision_trees = []

    # the bootstrapping datasets for trees
    # bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
    bootstraps_datasets = []

    # the true class labels, corresponding to records in the bootstrapping datasets
    # bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in 
    # the 'i'th bootstrapped dataset.
    bootstraps_labels = []

    def __init__(self, num_trees):
        # Initialization done here
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]


    def _bootstrapping(self, XX, n):
        # Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

        samples = [] # sampled dataset
        labels = []  # class labels for the sampled records
        size = len(XX[0])
        pick = np.random.randint(low=0, high=(len(XX)-1), size=n)
        for i in pick:
            samples.append(XX[i][:(size-1)])
            labels.append(XX[i][size-1])
        return (samples, labels)


    def bootstrapping(self, XX):
        # Initializing the bootstap datasets for each tree
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)


    def fitting(self):
        # Train `num_trees` decision trees using the bootstraps datasets
        # and labels by calling the learn function from your DecisionTree class.
        for i in range(self.num_trees):
            features = self.bootstraps_datasets[i]
            labels = self.bootstraps_labels[i]
            tree = self.decision_trees[i]
            tree.__init__()
            tree.learn(features, labels)
            


    def voting(self, X):
        y = []

        for record in X:
            # Following steps have been performed here:
            #   1. Find the set of trees that consider the record as an 
            #      out-of-bag sample.
            #   2. Predict the label using each of the above found trees.
            #   3. Use majority vote to find the final label for this recod.
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                if record not in dataset:
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)


            counts = np.bincount(votes)
            
            if len(counts) == 0:
                # Special case:
                #  Handle the case where the record is not an out-of-bag sample
                #  for any of the trees. 
                vote_special = []
                for j in range(len(self.decision_trees)):
                    temp_tree = self.decision_trees[j]
                    v = temp_tree.classify(record)
                    vote_special.append(v)
                y = np.append(y, np.argmax(np.bincount(vote_special)))
            else:
                y = np.append(y, np.argmax(counts))

        return y


def main():
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels
    numerical_cols = numerical_cols=set([i for i in range(0,43)]) # indices of numeric attributes (columns)

    # Loading data set
    with open("hw4-data.csv") as f:
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                if i in numerical_cols:
                    xline.append(ast.literal_eval(line[i]))
                else:
                    xline.append(line[i])

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])

    # Initialize according to your implementation
    # VERY IMPORTANT: Minimum forest_size should be 10
    forest_size = 10
    
    # Initializing a random forest.
    randomForest = RandomForest(forest_size)

    # Creating the bootstrapping datasets
    randomForest.bootstrapping(XX)

    # Building trees in the forest 
    randomForest.fitting()

    # Calculating an unbiased error estimation of the random forest
    # based on out-of-bag (OOB) error estimate.
    y_predicted = randomForest.voting(X)

    # Comparing predicted and true labels
    results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    print('The accuracy is:', accuracy)


if __name__ == "__main__":
    main()
