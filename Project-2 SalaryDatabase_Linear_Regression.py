import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('9.1 Salary_Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 1/3)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

y_pred = regression.predict(x_test)

#Plotting a graph for x-train and y-train
#plt.scatter(x_train, y_train, color='red', edgecolors='blue')
#plt.plot(x_train, regression.predict(x_train), color='blue')
#plt.title("Salary vs Experience (Training-Set)")
#plt.ylabel("Salary")
#plt.xlabel("Experience (years)")

#Plotting the graph for x-test and y-test
plt.scatter(x_test,y_test, color= 'red')
plt.plot(x_test, y_pred, color= 'blue')
plt.title("Salary vs Experience (Testing-Set)")
plt.ylabel("Salary")
plt.xlabel("Experience (years)")
plt.show()