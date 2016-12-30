# k nearest neighbors
import random
from scipy.spatial import distance

def euc(a,b):
	return distance.euclidean(a,b)

class ScrappyKNN():
	def fit(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train

	def predict(self, x_test):
		predictions = []

		for row in x_test:
			label = self.closest(row)
			predictions.append(label)
		
		return predictions

	def closest(self, row):
		best_dist = euc(row, self.x_train[0])
		best_index = 0
		for i in range(1, len(self.x_train)):
			dist = euc(row, self.x_train)
			if dist < best_dist:
				best_dist = dist
				best_index = i

		return self.y_train[best_index]


# load the data
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data #features
y = iris.target #labels

# split the data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

# train the classifier - knearest neighbors
#from sklearn.neighbors import KNeighborsClassifier

my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

# make a prediction using test data
predictions = my_classifier.predict(X_test)
# calculate accuracy of preciction using Y test value
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
