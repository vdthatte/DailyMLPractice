from sklearn import tree

# bumpy - 1
# orange - 1

features = [[150,1],[170,1],[140,0],[130,0]]
labels = [1,1,0,0]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

print clf.predict([[180,0]])
