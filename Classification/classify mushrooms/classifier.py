# load the data
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import csv

with open('mushrooms.csv', 'rb') as csvfile:
	rows = []
	mushrooms = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in mushrooms:
		rows.append(row[0].split(','))

	#print rows[1][1:] 

	x = []
	y = []

	for row in rows:
		y.append(row[0])
		x.append(row[1:])
	
	y = y[1:]
	x = x[1:]

	#print encoded_y
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
	# train the classifier - decision tree
	my_classifier = tree.DecisionTreeClassifier()
	my_classifier.fit(X_train, y_train)

	# make a prediction using test data
	predictions = my_classifier.predict(X_test)
	print predictions
