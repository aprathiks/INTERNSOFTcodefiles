# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 18:30:41 2020

@author: 
"""


#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf




#Reading the data from our files
Data = pd.read_csv("advertising.csv")
Data.head()


#To visualise data
fig, axs = plt.subplots(1, 3,sharey = True)
Data.plot(kind='scatter',x='TV',y='Sales',ax=axs[0],figsize=(14,7))
Data.plot(kind='scatter',x='Radio',y='Sales',ax=axs[1],figsize=(14,7))
Data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axs[2],figsize=(14,7))



#Creating X and Y for linear regression
feature_cols = ['TV']
X = Data[feature_cols]
y = Data.Sales




#Importing Linear Regression Algorithm for simple linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)

#y = a+bx
#Where y=output or result(sales)
#a=intercept , b=coefficient and x=input ie.., investment


result = 6.9748214882298925 + 0.05546477 * 50
print(result)



#CREATING DATAFRAME FOR MIN AND MAX VALUE OF THE TABLE
X_minmax = pd.DataFrame({'TV':[Data.TV.min(),Data.TV.max()]})
X_minmax.head()


pred = lr.predict(X_minmax)
pred


Data.plot(kind= 'scatter',x='TV',y='Sales')

plt.plot(X_minmax,pred,c='red',linewidth = 4)


lm = smf.ols(formula = 'Sales ~ TV',data = Data).fit()
lm.conf_int()


#FINDING THE PROBABLITY VALUES
lm.pvalues

#FINDING THE R-SQUARED VALUES
lm.rsquared



# MULTI LINEAR REGRESSION
feature_cols = ['TV','Radio','Newspaper']
X = Data[feature_cols]
y = Data.Sales


lr = LinearRegression()
lr.fit(X,y)

print(lr.intercept_)
print(lr.coef_)



lm = smf.ols(formula='Sales ~ TV+Radio+Newspaper',data=Data).fit()
lm.conf_int()
lm.summary()





