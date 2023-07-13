""""Heart Disease Classification and Evaluation with KNN: Confusion Matrix, Performance Metrics and Scatter Plot"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Loading the dataset
data = pd.read_csv('data/heart.csv')

# Choosing two features for the scatter plot
feature1 = 'age'
feature2 = 'chol'

# Extracting the chosen features and the target variable
X = data[[feature1, feature2]]
y = data.iloc[:, -1]   # just the last column

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Creating and training KNN
classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train)

# Making predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluating the model's performance
confusion = confusion_matrix(y_test, y_pred)
class_names = ['Class 0 - Less chance', 'Class 1 - More chance'] 
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix Display
confusion_matrix_plot = ConfusionMatrixDisplay(confusion, display_labels=class_names)
confusion_matrix_plot.plot()
plt.title('Confusion Matrix')
plt.show()

# Plotting Precision, Recall, and F1 Score
x_labels = ['Precision', 'Recall', 'F1 Score']
y_values = [precision, recall, f1]

plt.bar(x_labels, y_values)
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.show()

# Scatter plot
plt.scatter(X_test[feature1], X_test[feature2], c=y_pred, cmap='viridis')
plt.title('Scatter Plot')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.show()

# Print results
print("Confusion matrix:", confusion)
print("Precision score:", precision)
print("Recall score:", recall)
print("F1 score:", f1)
