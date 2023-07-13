"""Simple Logistic Regression on Scikit learn Iris Dataset"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

#Load the iris dataset
iris = load_iris()

X = iris.data
y = iris.target

#Descriptive info of iris dataset
print(iris.DESCR)

#Define classes
class_names = iris.target_names

#Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#Train a logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Predict probabilities for the test set
y_prob = classifier.predict_proba(X_test)[:,1]

#Plot the predicted probabilities
plt.figure(figsize=(10,6))
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
plt.scatter(range(len(y_test)), y_prob, color="red", label="Predicted")
plt.axhline(0.5, color="green", linestyle="--", label="Decision Boundary")
plt.xlabel("Sample Index")
plt.ylabel("Probability")
plt.title("Logistic regression predicted probabilities")
plt.legend()
plt.show()

#Prediction for classification report and confusion matrix
y_pred = classifier.predict(X_test)

#Classification report and confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
clf_report = classification_report(y_test, y_pred)
print(clf_report)

#Confusion Matrix Display
confusion_matrix_plot = ConfusionMatrixDisplay(cnf_matrix, display_labels=class_names)
confusion_matrix_plot.plot()
plt.show()