"""Simple Presentation of Linear Regression with scikit learn"""

import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt

#Dataset
Xtrain = np.array([[0], [1], [2]]) # input -2D array
ytrain = np.array([0, 1, 2])  #expected output -1D array

#Visualization of the training set
plt.scatter(Xtrain, ytrain)    
plt.title("Trening set")       
plt.xlabel("X_train")          
plt.ylabel("Y_train")          
plt.show()                     

#Creating linear model object
linear_model = lm.LinearRegression()

linear_model.fit(Xtrain, ytrain) 
#fit - training of the model, receives input data and target values
#uses them to learn the parameters of the model to achieve the
#best possible approximation of target values based on input data

#Print intercept and coefficient 
print(linear_model.intercept_)

print(linear_model.coef_)


#Test data & predicting label
Xtest = np.array([[0.5], [3]])          #test data -2D array
ypredict = linear_model.predict(Xtest)
#works based on learned parameters prediction of target values based on new input data (Xtest)
print(ypredict)

#Visualization
plt.scatter(Xtest, ypredict) 
plt.plot(Xtest, ypredict)      
plt.title("Prediction")        
plt.xlabel("X_train")          
plt.ylabel("Y_train")         
plt.show()                     