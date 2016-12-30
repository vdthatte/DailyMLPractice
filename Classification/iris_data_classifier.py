# IMPORT DATASET

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
import numpy as np
iris = load_iris()

# prepare data by splitting into testing and learning data
test_indexes = [0,50,100]
train_target = np.delete(iris.target, test_indexes)
train_data = np.delete(iris.data, test_indexes, axis=0)
test_target = iris.target[test_indexes]
test_data = iris.data[test_indexes]

# train a classifier
classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_target)


# predict label for a new flower
print classifier.predict(test_data)






