import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
car_data=pd.read_csv("fuel_efficiency_data.csv")
car_data.head(10)
indp_var=car_data[['Engine_Size_Liters']]
dep_var=car_data[['Fuel_Efficiency_MPG']]
plt.scatter(indp_var,dep_var)
plt.xlabel('Engine_Size_Liters')
plt.ylabel('Fuel_Efficiency_MPG')
plt.title ('Indp vs Dep for linearly verification')
train_x, test_x, train_y, test_y=train_test_split(indp_var,dep_var,test_size=0.2,random_state=8)
lr_model=LinearRegression()
lr_model.fit(train_x,train_y)
y_hat=lr_model.predict(test_x)
mse=mean_squared_error(y_hat,test_y)
print(f"the MSE for this model on the testing set is {mse}")
plt.scatter (train_x,train_y, color='blue', label="training set")
plt.scatter (test_x,test_y,color='green', label='testing set')
plt.plot(test_x,y_hat,color='salmon',label='regression line')
plt.legend()
print(f" the m for this model is = {lr_model.coef_}")
print(f" the b for this model is = {lr_model.intercept_}")
new_entry=input ("please enter a new size value:")
new_entry=np.array([[float(new_entry)]])
new_prediction=lr_model.predict(new_entry)
print(f" the mpg for this car is {new_prediction} MPG")
