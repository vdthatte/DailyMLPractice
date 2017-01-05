from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model

boston = load_boston()

y = boston.target
x = boston.data

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print regr.predict(X_test)
