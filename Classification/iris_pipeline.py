# load the data
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data #features
y = iris.target #labels


# split the data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# train the classifier - decision tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

# train the classifier - knearest neighbors
from sklearn.neighbors import KNeighborsClassifier
my_other_classifier = KNeighborsClassifier()
my_other_classifier.fit(X_train, y_train)

# make a prediction using test data
predictions = my_classifier.predict(X_test)
other_predictions = my_other_classifier.predict(X_test)
# calculate accuracy of preciction using Y test value
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
print accuracy_score(y_test, other_predictions)
