from sklearn import tree
X = [[0, 0], [1, 1],[0, 1]]
Y = [0, 1, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

x = clf.predict([[0,2]])
p = clf.predict_proba([[0,2]])
print(x)
print(p)