"""Simple presentation of Logistic Regression - model's performance using classification metrics and confusion matrix"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Dataset
X_train = np.array([[1], [2], [3], [4], [5], [6]]) #input variables - 2D numpy array
y_train = np.array([0, 0, 0, 1, 1, 1]) #target labels - 1D numpy array

# Visualization of training data
# for understanding the distribution and separation of the two classes
plt.scatter(X_train, y_train, c=y_train, cmap='bwr') 
plt.plot(X_train, y_train)
plt.title("Training data")
plt.xlabel("X_train")
plt.ylabel("y_train")
plt.colorbar(ticks=[0, 1], label='Class')
plt.legend()
plt.show()

#Initialization of the logistic regression and training model on a training set 
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Test data 
X_test = np.array([[2.5], [3.51]])
y_test = np.array([0, 1])

#Predicting labels for test data
y_pred = classifier.predict(X_test)

#Visualization of test data
plt.scatter(X_test, y_pred, c=y_pred, cmap='bwr') 
#plt.plot(X_test, y_pred)
plt.title("Test data")
plt.xlabel("X_test")
plt.ylabel("y_predict")
plt.colorbar(ticks=[0, 1], label='Class')
plt.legend()
plt.show()

#Classification report and confusion matrix
report = classification_report(y_test, y_pred) 
cnf_matrix = confusion_matrix(y_test, y_pred) 

print(report)
print(cnf_matrix)

#Confusion Matrix Display
confusion_matrix_plot = ConfusionMatrixDisplay(cnf_matrix)
confusion_matrix_plot.plot()
plt.show()