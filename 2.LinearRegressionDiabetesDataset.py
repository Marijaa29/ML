"""Linear Regression Analysis on the Diabetes Dataset using Scikit learn"""

from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def show_feature_relations(X_values, y_values): 
    nrows, ncols = 2, 5 #determined by how many input columns x we have
    fig = plt.figure()  #displays a graph for each feature
    fig.suptitle("Feature relations")
    for i, column_name in enumerate(X_values):   
        ax = fig.add_subplot(nrows, ncols, i+1)  
        ax.scatter(X_values[column_name], y_values) 
        ax.set_title(column_name)
    plt.show()
#this function visualizes the relationship between each feature in X_values (the input data)
#and the target variable y_values using scatter plots
#it creates a grid of subplots (2 rows and 5 columns) 
#and plots the scatter plot for each feature against the target variable.
    
def fit_and_visualize_model(diabetes_X, diabetes_y): 
    diabetes_y = diabetes_y.values  #converts target values to numpy array
    nrows, ncols = 2,5
    fig = plt.figure()
    fig.suptitle("Linear regression results")
    for i, feature_name in enumerate(diabetes_X):
        diabetes_X_feature = diabetes_X[feature_name].values 
        diabetes_X_feature = np.expand_dims(diabetes_X_feature, axis=1)  
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_X_feature, diabetes_y, test_size=0.2, random_state=1)

        lm = linear_model.LinearRegression()
        lm.fit(diabetes_X_train, diabetes_y_train)
        diabetes_y_pred = lm.predict(diabetes_X_test) 
        mse = mean_squared_error(diabetes_X_test, diabetes_y_pred)  #average squared difference between actual and predicted values 
        r2_result = r2_score(diabetes_X_test, diabetes_y_pred)  #how well the model fits the data from 0 to 1
        
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.set_title("{}\nmse: {:.2f}\nr2: {:.2f}.".format(feature_name, mse, r2_result))
        ax.scatter(diabetes_X_test, diabetes_y_test, color="black", label = "Real values") 
        ax.plot(diabetes_X_test, diabetes_y_pred, color="blue", marker ="o", linewidth = 3, label ="Predictive values")
    plt.show()

#this function fits a linear regression model for each feature in diabetes_X (the input data) and visualizes the results for each feature,
#it splits the data into training and testing sets,
#fits a linear regression model using the training data,
#predicts the target variable for the test data,
#calculates the mean squared error (mse) and coefficient of determination (r2) between the predicted and actual values
#plots the scatter plot of the test data, the predicted values, and displays the mse and r2 values in the subplot.


diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)
#print(diabetes_X.shape)
#print(diabetes_y)

show_feature_relations(diabetes_X, diabetes_y)

fit_and_visualize_model(diabetes_X, diabetes_y)