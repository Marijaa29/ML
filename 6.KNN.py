"""an example of KNeighborsClassifier"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

iris = load_iris()
#print(iris.DESCR)

X = iris.data
y = iris.target

class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cnf_matrix = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)

print(clf_report)
confusion_matrix_plot = ConfusionMatrixDisplay(cnf_matrix, display_labels=class_names)
confusion_matrix_plot.plot()
plt.show()