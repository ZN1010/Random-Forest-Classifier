# Random-Forest-Classifier
This project serves as a part of the homework from CSE 6242 (Data and Visual Analytics at Georgia Tech, Spring 2019) at https://poloclub.github.io/cse6242-2019spring-campus/. \
I implement a **random forest classifier** in Python3 **without using existing machine learning or random forest libraries like scikit-learn**.
## Description
By using information gain to perform the splitting in the decision tree, a random forest classifier is built to classify if a molecule is biodegradable or not based on given attributes of the molecule. The performance of the classifier is evaluated via the out-of-bag (OOB) error estimate. To shorten the running time and obtain a quick demonstration, the random forest is initialized with a size of 10. Users can change the size if they want more accurate prediction.
## Data
The provided dataset is called 'hw4-data.csv'. The dataset is extracted from the [QSAR biodegradation Data Set](http://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation), and it has been cleaned to remove missing attributes. Specifically, the first 40 columns in the dataset represent attributes of a certain molecule, and the last one is the ground truth whether it is biodegradable or not.
## How to run
To run the classifier, use:

```console
$ python random_forest.py
```
