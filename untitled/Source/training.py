import Directory
from Dataset import Dataset

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from math import sqrt
import matplotlib.pyplot as plt


if __name__ == "__main__":
	print("go")
	db = Dataset("E:/Clustered/")
	db.info()
	print("total: "+str(db.total))

	X_train, X_test, y_train, y_test \
		= train_test_split(db.dataset, db.target, test_size=1/5, random_state=1)
	print("knn")
	neigh = KNeighborsClassifier(n_neighbors= int(sqrt(db.max)))
	neigh.fit(X_train, y_train)
	print("save")
	dump(neigh, 'E:\kneigh.joblib')

	print("score")
	neigh = load('E:\kneigh.joblib')
	score = neigh.score(X_test, y_test)
	print(score)