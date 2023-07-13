"""1. Task
Build a classifier using logistic regression whose output is a binary value that denotes
is there a chance of a heart attack based on the information / findings of the patient recorded in heart.csv
files.
Description of input sizes:
1) ages
2) sex
3) chest pain type (4 values)
4) resting blood pressure
5) serum cholesterol in mg/dl
6) fasting blood sugar > 120 mg/dl
7) resting electrocardiographic results (values 0,1,2)
8) maximum heart rate achieved
9) exercise-induced angina
10) old peak = ST depression induced by exercise relative to rest
11) the slope of the peak exercise ST segment
12) number of major vessels (0-3) colored by flourosopy
13) thal: 0 = normal; 1 = fixed defect; 2 = reversible defect
14) target: 0= less chance of heart attack 1= more chance of heart attack
Using the test data, create / determine:
1. The confusion matrix
2. Precision
3. Response
4. F1 measure"""

"""Logistic Regression for a binary classification task using the heart disease dataset"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

#Loading the dataset
data = pd.read_csv('data/heart.csv')

#Splitting the data into features(X) and target variable(y)
X = data.iloc[:, :-1] #all columns except the last one
y = data.iloc[:, -1]   #just last column


#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Creating and training the logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Making predictions on the test set
y_pred = classifier.predict(X_test)

#Evaluating the models performance
confusion = confusion_matrix(y_test, y_pred)
class_names = ['Class 0 - Less chance', 'Class 1 - More chance']  
clf_report = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#Confusion Matrix Display
confusion_matrix_plot = ConfusionMatrixDisplay(confusion, display_labels=class_names)
confusion_matrix_plot.plot()
plt.show()

#Print results
print("Confusion matrix:", confusion)
print("Precision score:", precision)
print("Recall score:", recall)
print("F1 score:", f1)