"""an example of KNeighborsClassifier on "heart.csv" dataset"""

#Repeat the previous task using the K nearest neighbors method. 
#Use multiple input sizes, interpret the results

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

#Loading the dataset
data = pd.read_csv('data/heart.csv')

#Splitting the data into features(X) and target variable(y)
X = data.iloc[:, :-1] #all columns except the last one
y = data.iloc[:, -1]   #just last column


#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Kreating and training KNN
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

#Making predictions on the test set
y_pred = classifier.predict(X_test)

#Evaluating the models performance
confusion = confusion_matrix(y_test, y_pred)
class_names = ['Class 0 - Less chance', 'Class 1 - More chance']  
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
