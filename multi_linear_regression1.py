# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:32:38 2019

@author: Ahmed Khaled
steps for multi_linear regression are :
    step 1 :import libararies
    step 2 :Get data set
    step 3 :split data into input & output
    step 4 :check missing data
    step 5: check categeorical data
    step 6 :split data into training data  & test data 
    step 7 :Build your model
    step 8 :plot best line 
    step 9 :Estimate Error 
"""
#feature scalling 
"""from sklearn.preprocessing import StandardScaler
sc_x =StandardScaler()
x_train =sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test )"""    

#Data processing
#step 1 :import libararies
import numpy as np   # to make  mathmatical operation on metrices
import pandas as pd  #to read data
import matplotlib.pyplot as plt   #to show some graghs
import seaborn as sns    #for plot data
from sklearn.cross_validation import  train_test_split #to split data to train & test
from sklearn.linear_model import LinearRegression    #to import linear model 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder   #for categeorical data
from sklearn.metrics import mean_squared_error #to calculate MSE ,MAE ,RMSE


#step 2 :Get data set, importing the data set 
path='C:\\Users\\Ahmed Khaled\\Downloads\\my work (regression)\\4)multilinear regression\\50_Startups.csv' 
data  = pd.read_csv(path)
print('data : \n',data)
print('data.head : \n',data.head())
print('data.shape : \n',data.shape)
print('names of columns :\n',data.columns)
print('data.imnformation: \n ' ,data.info())
print('data.describe: \n ' ,data.describe())
sns.pairplot(data)      #to visulize data in pair
sns.distplot(data['Profit']) #distrebution
data.corr()    #corrolations
sns.heatmap(data.corr()) 
sns.heatmap(data.corr(),annot=True) #relation with number


#step 3 :split data into input & output
cols = data.shape[1]
x = data.iloc[:, :-1].values       #note data type is object
y = data.iloc[: , cols-1].values

#step 4 :check missing data 
# there is no missing data 

#step 5: check categeorical data
# there is  categeorical data

#Encoding categrical data
#from sklearn.preprocessing import LabelEncoder ,OneHotEncoder 
labelencoder_x = LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3]) #encode the fourth column to 0,1,2 ,convert the text to int values 
#dummy variable to encode text without any piority 
onehotencoder = OneHotEncoder(categorical_features = [3] )  
x = onehotencoder.fit_transform(x).toarray()
# Avoiding the Dummy variable trap
x=x[:, 1:]


#step 6 :split data into training data  & test data 
#from sklearn.cross_validation import train_test_split
x_train, x_test ,y_train,y_test =train_test_split(x,y,test_size = 0.2  ,random_state = 0 )


#step 7 :Build your model fitting multileaner regression training set 
#from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(x_train,y_train)
# print the intercept
print(model.intercept_) 

#step 8 :plot best line accordding to your prediction
from sklearn.datasets import load_boston
boston  = load_boston()
print('boston.Keys\n',boston.keys()) 
print(boston['target'])  # target from keys of dictionary of boston
print(boston['feature_names'])  # feature_names from keys of dictionary of boston

y_pred = model.predict(x_test)
plt.scatter(y_test,y_pred)
sns.distplot((y_test-y_pred),bins=50) 


#step 9 :Estimate Error 
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test,y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
x=np.append(arr = np.ones((50, 1)).astype(int), values = x ,axis=1)
x_opt = x[:,  [0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog = y ,exog = x_opt).fit()
regressor_OLS.summary()
#############################################
x_opt = x[:,  [0,2,1,3,4]]
regressor_OLS=sm.OLS(endog = y ,exog = x_opt).fit()
regressor_OLS.summary() 
##############################################
x_opt = x[:,  [0,1,2,3]]
regressor_OLS=sm.OLS(endog = y ,exog = x_opt).fit()
regressor_OLS.summary() 
 






 




