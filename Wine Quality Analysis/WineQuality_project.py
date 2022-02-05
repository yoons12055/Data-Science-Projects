#!/usr/bin/env python
# coding: utf-8

# # Prediction on Wine Quality Project

# ###  Python(NumPy, Pandas, matplotlib, sklearn, GridSearchCV),  Machine Learning models(OLS Regression, Ridge Regression, Lasso Regression), Jupyter Notebook

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def Insert_row_(row_number, df, row_value):
    df1 = df[0:row_number]
    df2 = df[row_number:]
    df1.loc[row_number]=row_value
    df_result = pd.concat([df1, df2])
    df_result.index = [*range(df_result.shape[0])]
    return df_result

def OSR2(model, X_test, y_test, y_train):
    y_pred = model.predict(X_test)
    SSE = np.sum((y_test - y_pred)**2)
    SST = np.sum((y_test - np.mean(y_train))**2)   
    return (1 - SSE/SST)

data = pd.read_csv("winequality-red.csv")
data.head()


# With a given dataset containing fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and quality of wines, using a python language, I assigned y as a quality which is a subjective variable of wine dataset and X as a dataset of the other objective variables of wine and built following three kinds of models (X vs y): OLS regression, ridge regression, and lasso regression. First of all, I split the dataset into 70% of the training set and 30% of the test set with an intercept.

# # 1. OLS Regression Model

# In[2]:


cols = ["fixed acidity", "volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
X = data[cols]
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
olsmodel = linear_model.LinearRegression(fit_intercept=True)
olsmodel.fit(X_train, y_train)


# For the OLS regression model, I built a linear regression model with fit_intercept = True and fit to training sets of X and y. I got a coefficient table for the OLS model but no constant. By using a function Insert_row_, I inserted a constant variable to index 0 of the coefficient table for the OLS model. Based on the model, I found the OSR^2, MSE, and MAE of OLS model through a function of OSR2, mean_squared_error, and mean_absolute_error from sklearn.metrics.

# # 1.1 Table of Coefficients

# In[3]:


# OLS Regression, Table of Coefficients
OLS_coeftable=pd.DataFrame(columns=["variable", "coefficient"])
OLS_coeftable["variable"]=cols
OLS_coeftable["coefficient"]= olsmodel.coef_
OLS_intercept = ['constant', olsmodel.intercept_]
OLS_coeftable = Insert_row_(0, OLS_coeftable, OLS_intercept)
OLS_coeftable


# # 1.2 OSR^2, MSE, MAE

# In[4]:


# OLS OSR^2, MSE, MAE
ols_osr2=OSR2(olsmodel, X_test, y_test, y_train)

predict_train = olsmodel.predict(X_train)
predict_test = olsmodel.predict(X_test)

ols_train_mse = mse(y_train, predict_train)
ols_test_mse = mse(y_test, predict_test)

ols_train_mae = mae(y_train, predict_train)
ols_test_mae = mae(y_test, predict_test)

print(ols_osr2)
print(ols_train_mse, ols_test_mse)
print(ols_train_mae, ols_test_mae)


# # 2. Ridge Regression Model

# Since each ridge regression and lasso regression has one tuning parameter, for those two models, using GridSearchCV with 10 splits, I got a graph of CV error then found a best tuning parameter which results in the lowest CV error. With the best parameter, I made a table of coefficient for those two models. 
# For the ridge model, I assigned an alpha_grid, 88 random states, and a scoring method as a negative mean squared error then fit to X and y. Later in order to find CV error score, I multiplied -1 since I used negative mean squared error. With those, I found a CV error graph of Ridge regression(log Alpha vs CV Error) model. I also found OSR2, MSE, and MAE for the ridge model. 

# In[5]:


alpha_grid = {'alpha': np.logspace(-1, 5, num=50, base=10)}
rr = Ridge(random_state=88)
rr_cv = GridSearchCV(rr, alpha_grid, scoring='neg_mean_squared_error', cv=10)
rr_cv.fit(X_train, y_train)


# In[6]:


range_alpha = rr_cv.cv_results_['param_alpha'].data
CV_scores = rr_cv.cv_results_['mean_test_score']*(-1)
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_xscale('log')
plt.xlabel('Alpha', fontsize=16)
plt.ylabel('CV Error', fontsize=16)
plt.scatter(range_alpha, CV_scores, s=30)
plt.tight_layout()
plt.show()


# In[7]:


print(rr_cv.best_params_)


# # 2.1 Table of Coefficient

# In[8]:


real_ridge = Ridge(alpha=0.9540954763499939, fit_intercept=True)
real_ridge.fit(X_train,y_train)
ridge_intercept = ['constant', real_ridge.intercept_]
ridge_coeftable = pd.DataFrame(columns=["variable", "coefficient"])
ridge_coeftable["variable"]=cols
ridge_coeftable["coefficient"]=real_ridge.coef_
ridge_coeftable = Insert_row_(0, ridge_coeftable, ridge_intercept)
ridge_coeftable


# # 2.2 OSR^2, MSE, MAE

# In[9]:


ridge_osr2=OSR2(real_ridge, X_test, y_test, y_train)

rpredict_train=real_ridge.predict(X_train)
rpredict_test=real_ridge.predict(X_test)

ridge_train_mse=mse(y_train, rpredict_train)
ridge_test_mse=mse(y_test, rpredict_test)

ridge_train_mae=mae(y_train, rpredict_train)
ridge_test_mae=mae(y_test, rpredict_test)

print(ridge_osr2)
print(ridge_train_mse, ridge_test_mse)
print(ridge_train_mae, ridge_test_mae)


# Same as the ridge regression model, I used GridSearchCV with 10 splits, I got a graph of CV error then found a best tuning parameter which results in the lowest CV error. With the best parameter, I made a table of coefficient for the lasso regression model. Also for the lasso model, I assigned an alpha_grid, 88 random states, and a scoring method as a negative mean squared error then fit to X and y. Later in order to find CV error score, I multiplied -1 since I used negative mean squared error. With those, I found a CV error graph of lasso regression(log of Alpha vs CV Error) model. I also found OSR2, MSE, and MAE for the lasso model.

# # 3. Lasso Regression CV

# In[10]:


alphas = np.logspace(-8, 1 , num=50, base=10)

for a in alphas:
    lasso = Lasso(alpha=a, random_state=88)

alpha_grid = {'alpha': np.logspace(-8, -1, num=50, base=10)}
lasso_cv = GridSearchCV(lasso, alpha_grid, scoring='neg_mean_squared_error', cv=10)
lasso_cv.fit(X_train, y_train)
range_alpha = lasso_cv.cv_results_['param_alpha'].data
CV_scores = lasso_cv.cv_results_['mean_test_score']*(-1)
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.set_xscale('log')
plt.xlabel('Alpha', fontsize=16)
plt.ylabel('CV Error', fontsize=16)
plt.scatter(range_alpha, CV_scores, s=30)
plt.tight_layout()
plt.show()


# In[11]:


print(lasso_cv.best_params_)


# With this best parameter, I found a table of coefficients of the lasso regression model below. The coefficients of the lasso regression model (some coefficients are zero) seems to be much closer to 0 than those of OLS and ridge regression models. 

# # 3.1 Table of Coefficients

# In[12]:


real_lasso = Lasso(alpha=0.0001, fit_intercept=True)
real_lasso.fit(X_train,y_train)
lasso_intercept=['constant', real_lasso.intercept_]
lasso_coeftable = pd.DataFrame(columns=["variable", "coefficient"])
lasso_coeftable["variable"]=cols
lasso_coeftable["coefficient"]=real_lasso.coef_
lasso_coeftable = Insert_row_(0, lasso_coeftable, lasso_intercept)
lasso_coeftable


# # 3.2 OSR^2, MSE, MAE

# In[13]:


lasso_osr2=OSR2(real_lasso, X_test, y_test, y_train)

lpredict_train=real_lasso.predict(X_train)
lpredict_test=real_lasso.predict(X_test)

lasso_train_mse=mse(y_train, lpredict_train)
lasso_test_mse=mse(y_test, lpredict_test)

lasso_train_mae=mae(y_train, lpredict_train)
lasso_test_mae=mae(y_test, lpredict_test)

print(lasso_osr2)
print(lasso_train_mse, lasso_test_mse)
print(lasso_train_mae, lasso_test_mae)


# Finally, I got a comparison table comparing values of OSR^2, RMSE, and MAE of those three previous models based on the test sets. According to the comparison table, Lasso Regression Model seems to be the most desirable because of the highest OSR^2 accuracy and lowest RMSE and MAE as well. And the ridge regression model seems to be the worst because of the lowest OSR^2 and highest RMSE and MAE. 

# # 4. Comparison Table

# In[14]:


comparison_data = {
    'Ridge Regression': ['{:.6f}'.format(ridge_osr2),
                         '{:.6f}'.format((ridge_test_mse)**(1/2)),
                         '{:.6f}'.format(ridge_test_mae)],
    'OLS Regression': ['{:.6f}'.format(ols_osr2),
                       '{:.6f}'.format((ols_test_mse)**(1/2)),
                       '{:.6f}'.format(ols_test_mae)],
    'Lasso Regression': ['{:.6f}'.format(lasso_osr2),
                         '{:.6f}'.format((lasso_test_mse)**(1/2)),
                         '{:.6f}'.format(lasso_test_mae)],
}
comparison_table = pd.DataFrame(data=comparison_data, index=['Out-of-sample R2', 'Out-of-sample RMSE', 'Out-of-sample MAE'])
comparison_table


# # 5. Predict Wine Quality using Lasso Regression

# In[21]:


lasso_pred0 = lasso_cv.predict(X_train)
lasso_pred1 = lasso_cv.predict(X_test)
lasso_pred0 = lasso_pred0.astype(int)
lasso_pred1 = lasso_pred1.astype(int)
lasso_pred = np.concatenate((lasso_pred0, lasso_pred1))
lasso_pred


# In[22]:


# Make a dataframe and convert from float to int
predicted_dataset = pd.DataFrame()
predicted_dataset['Quality'] = y_test
predicted_dataset['PRED Quality'] = lasso_pred1
predicted_dataset

