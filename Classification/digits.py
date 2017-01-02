from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

digits = load_digits()

x = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print accuracy_score(y_test, predictions)