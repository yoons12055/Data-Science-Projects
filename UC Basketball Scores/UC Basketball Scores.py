#!/usr/bin/env python
# coding: utf-8

# # Prediction on Scores Scored by UC Basketball Players
# 

# ## Python(NumPy, Pandas, Seaborn, matplotlib, sklearn, OneHotEncoder, GridSearchCV ,KFold), Machine Learning Models(OLS Regression, Decision Tree Regressor, Random Forest Regressor, Lasso Regression, Ridge Regression), Jupyter Notebook

# In[1]:


import re
import nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[2]:


import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


# # Data Cleaning

# In[3]:


#function that reads position dataframe
def position_df(html):
    table = pd.read_html(html)
    table = table[2]
    return table


# In[4]:


ucb21_pos = position_df('https://calbears.com/sports/mens-basketball/roster/2020-21')
ucb20_pos = position_df('https://calbears.com/sports/mens-basketball/roster/2019-20')
ucb19_pos = position_df('https://calbears.com/sports/mens-basketball/roster/2018-19')
ucb18_pos = position_df('https://calbears.com/sports/mens-basketball/roster/2017-18')
ucb17_pos = position_df('https://calbears.com/sports/mens-basketball/roster/2016-17')
ucb16_pos = position_df('https://calbears.com/sports/mens-basketball/roster/2015-16')
ucb14_pos = position_df('https://calbears.com/sports/mens-basketball/roster/2013-14')

ucsd21_pos = position_df('https://ucsdtritons.com/sports/mens-basketball/roster/2020-21')
ucsd20_pos = position_df('https://ucsdtritons.com/sports/mens-basketball/roster/2019-20')
ucsd19_pos = position_df('https://ucsdtritons.com/sports/mens-basketball/roster/2018-19')
ucsd18_pos = position_df('https://ucsdtritons.com/sports/mens-basketball/roster/2017-18')
ucsd17_pos = position_df('https://ucsdtritons.com/sports/mens-basketball/roster/2016-17')

uci21_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2020-21')
uci20_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2019-20')
uci19_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2018-19')
uci18_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2017-18')
uci17_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2016-17')
uci16_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2015-16')
uci15_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2014-15')
uci14_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2013-14')
uci13_pos = position_df('https://ucirvinesports.com/sports/mens-basketball/roster/2012-13')

ucla21_pos = position_df('https://uclabruins.com/sports/mens-basketball/roster/2020-21')
ucla20_pos = position_df('https://uclabruins.com/sports/mens-basketball/roster/2019-20')
ucla19_pos = position_df('https://uclabruins.com/sports/mens-basketball/roster/2018-19')
ucla18_pos = position_df('https://uclabruins.com/sports/mens-basketball/roster/2017-18')
ucla17_pos = position_df('https://uclabruins.com/sports/mens-basketball/roster/2016-17')
ucla16_pos = position_df('https://uclabruins.com/sports/mens-basketball/roster/2015-16')

ucd21_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2020-21')
ucd20_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2019-20')
ucd19_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2018-19')
ucd18_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2017-18')
ucd17_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2016-17')
ucd16_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2015-16')
ucd15_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2014-15')
ucd14_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2013-14')
ucd13_pos = position_df('https://ucdavisaggies.com/sports/mens-basketball/roster/2012-13')


# In[5]:


#function that reads statistics dataframe
def stats_df(html):
    table = pd.read_html(html, match = 'Player Averages')
    table = table[0]
    table = table.droplevel(0, axis=1)
    table = table.drop(table.index[len(table)-2]).drop(table.index[len(table)-1])
    return table


# In[6]:


ucb21_st = stats_df('https://calbears.com/sports/mens-basketball/stats/2020-21#individual')
ucb20_st = stats_df('https://calbears.com/sports/mens-basketball/stats/2019-20#individual')
ucb19_st = stats_df('https://calbears.com/sports/mens-basketball/stats/2018-19#individual')
ucb18_st = stats_df('https://calbears.com/sports/mens-basketball/stats/2017-18#individual')
ucb17_st = stats_df('https://calbears.com/sports/mens-basketball/stats/2016-17#individual')
ucb16_st = stats_df('https://calbears.com/sports/mens-basketball/stats/2015-16#individual')
ucb14_st = stats_df('https://calbears.com/sports/mens-basketball/stats/2013-14#individual')

ucsd21_st = stats_df('https://ucsdtritons.com/sports/mens-basketball/stats/2020-21#individual')
ucsd20_st = stats_df('https://ucsdtritons.com/sports/mens-basketball/stats/2019-20#individual')
ucsd19_st = stats_df('https://ucsdtritons.com/sports/mens-basketball/stats/2018-19#individual')
ucsd18_st = stats_df('https://ucsdtritons.com/sports/mens-basketball/stats/2017-18#individual')
ucsd17_st = stats_df('https://ucsdtritons.com/sports/mens-basketball/stats/2016-17#individual')

uci21_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2020-21#individual')
uci20_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2019-20#individual')
uci19_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2018-19#individual')
uci18_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2017-18#individual')
uci17_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2016-17#individual')
uci16_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2015-16#individual')
uci15_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2014-15#individual')
uci14_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2013-14#individual')
uci13_st = stats_df('https://ucirvinesports.com/sports/mens-basketball/stats/2012-13#individual')

ucla21_st = stats_df('https://uclabruins.com/sports/mens-basketball/stats/2020-21#individual')
ucla20_st = stats_df('https://uclabruins.com/sports/mens-basketball/stats/2019-20#individual')
ucla19_st = stats_df('https://uclabruins.com/sports/mens-basketball/stats/2018-19#individual')
ucla18_st = stats_df('https://uclabruins.com/sports/mens-basketball/stats/2017-18#individual')
ucla17_st = stats_df('https://uclabruins.com/sports/mens-basketball/stats/2016-17#individual')
ucla16_st = stats_df('https://uclabruins.com/sports/mens-basketball/stats/2015-16#individual')

ucd21_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2020-21#individual')
ucd20_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2019-20#individual')
ucd19_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2018-19#individual')
ucd18_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2017-18#individual')
ucd17_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2016-17#individual')
ucd16_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2015-16#individual')
ucd15_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2014-15#individual')
ucd14_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2013-14#individual')
ucd13_st = stats_df('https://ucdavisaggies.com/sports/mens-basketball/stats/2012-13#individual')


# In[7]:


#function that change column name
def change_colname(df, old, new):
    df = df.rename(columns={old: new})
    return df


# In[8]:


#function that change the column names so that all the df have same column names
ucb21_st = change_colname(ucb21_st, "#", "No.")
ucb20_st = change_colname(ucb20_st, "#", "No.")
ucb19_st = change_colname(ucb19_st, "#", "No.")
ucb18_st = change_colname(ucb18_st, "#", "No.")
ucb17_st = change_colname(ucb17_st, "#", "No.")
ucb16_st = change_colname(ucb16_st, "#", "No.")
ucb14_st = change_colname(ucb14_st, "#", "No.")

ucsd21_st = change_colname(ucsd21_st, "#", "No.")
ucsd20_st = change_colname(ucsd20_st, "#", "No.")
ucsd19_st = change_colname(ucsd19_st, "#", "No.")
ucsd18_st = change_colname(ucsd18_st, "#", "No.")
ucsd17_st = change_colname(ucsd17_st, "#", "No.")

uci21_st = change_colname(uci21_st, "#", "No.")
uci20_st = change_colname(uci20_st, "#", "No.")
uci19_st = change_colname(uci19_st, "#", "No.")
uci18_st = change_colname(uci18_st, "#", "No.")
uci17_st = change_colname(uci17_st, "#", "No.")
uci16_st = change_colname(uci16_st, "#", "No.")
uci15_st = change_colname(uci15_st, "#", "No.")
uci14_st = change_colname(uci14_st, "#", "No.")
uci13_st = change_colname(uci13_st, "#", "No.")

ucla21_st = change_colname(ucla21_st, "#", "No.")
ucla20_st = change_colname(ucla20_st, "#", "No.")
ucla19_st = change_colname(ucla19_st, "#", "No.")
ucla18_st = change_colname(ucla18_st, "#", "No.")
ucla17_st = change_colname(ucla17_st, "#", "No.")
ucla16_st = change_colname(ucla16_st, "#", "No.")

ucd21_st = change_colname(ucd21_st, "#", "No.")
ucd20_st = change_colname(ucd20_st, "#", "No.")
ucd19_st = change_colname(ucd19_st, "#", "No.")
ucd18_st = change_colname(ucd18_st, "#", "No.")
ucd17_st = change_colname(ucd17_st, "#", "No.")
ucd16_st = change_colname(ucd16_st, "#", "No.")
ucd15_st = change_colname(ucd15_st, "#", "No.")
ucd14_st = change_colname(ucd14_st, "#", "No.")
ucd13_st = change_colname(ucd13_st, "#", "No.")


# In[9]:


ucsd19_pos = change_colname(ucsd19_pos, "#", "No.")
ucsd18_pos = change_colname(ucsd18_pos, "#", "No.")
ucsd17_pos = change_colname(ucsd17_pos, "#", "No.")

ucb21_pos = change_colname(ucb21_pos, "#", "No.")
ucb20_pos = change_colname(ucb20_pos, "#", "No.")
ucb19_pos = change_colname(ucb19_pos, "#", "No.")
ucb18_pos = change_colname(ucb18_pos, "#", "No.")
ucb17_pos = change_colname(ucb17_pos, "#", "No.")
ucb16_pos = change_colname(ucb16_pos, "#", "No.")
ucb14_pos = change_colname(ucb14_pos, "#", "No.")

uci21_pos = change_colname(uci21_pos, "#", "No.")
uci20_pos = change_colname(uci20_pos, "#", "No.")
uci19_pos = change_colname(uci19_pos, "#", "No.")
uci18_pos = change_colname(uci18_pos, "#", "No.")
uci17_pos = change_colname(uci17_pos, "#", "No.")
uci16_pos = change_colname(uci16_pos, "#", "No.")
uci15_pos = change_colname(uci15_pos, "#", "No.")
uci14_pos = change_colname(uci14_pos, "#", "No.")
uci13_pos = change_colname(uci13_pos, "#", "No.")

ucla21_pos = change_colname(ucla21_pos, "#", "No.")
ucla20_pos = change_colname(ucla20_pos, "#", "No.")
ucla19_pos = change_colname(ucla19_pos, "#", "No.")
ucla18_pos = change_colname(ucla18_pos, "#", "No.")
ucla17_pos = change_colname(ucla17_pos, "#", "No.")
ucla16_pos = change_colname(ucla16_pos, "#", "No.")

ucd21_pos = change_colname(ucd21_pos, "#", "No.")
ucd20_pos = change_colname(ucd20_pos, "#", "No.")
ucd19_pos = change_colname(ucd19_pos, "#", "No.")
ucd18_pos = change_colname(ucd18_pos, "#", "No.")
ucd17_pos = change_colname(ucd17_pos, "#", "No.")
ucd16_pos = change_colname(ucd16_pos, "#", "No.")
ucd15_pos = change_colname(ucd15_pos, "#", "No.")
ucd14_pos = change_colname(ucd14_pos, "#", "No.")
ucd13_pos = change_colname(ucd13_pos, "#", "No.")

ucsd21_pos = change_colname(ucsd21_pos, "Name", "Full Name")
ucsd20_pos = change_colname(ucsd20_pos, "Name", "Full Name")
ucsd19_pos = change_colname(ucsd19_pos, "Name", "Full Name")
ucsd18_pos = change_colname(ucsd18_pos, "Name", "Full Name")
ucsd17_pos = change_colname(ucsd17_pos, "Name", "Full Name")

ucsd21_pos = change_colname(ucsd21_pos, "Pos", "Pos.")
ucsd20_pos = change_colname(ucsd20_pos, "Pos", "Pos.")
ucsd19_pos = change_colname(ucsd19_pos, "Pos", "Pos.")
ucsd18_pos = change_colname(ucsd18_pos, "Pos", "Pos.")
ucsd17_pos = change_colname(ucsd17_pos, "Pos", "Pos.")

uci19_pos = change_colname(uci19_pos, "Pos", "Pos.")
uci18_pos = change_colname(uci18_pos, "Pos", "Pos.")
uci17_pos = change_colname(uci17_pos, "Pos", "Pos.")
uci16_pos = change_colname(uci16_pos, "Pos", "Pos.")
uci15_pos = change_colname(uci15_pos, "Pos", "Pos.")
uci14_pos = change_colname(uci14_pos, "Pos", "Pos.")
uci13_pos = change_colname(uci13_pos, "Pos", "Pos.")

ucb17_pos = change_colname(ucb17_pos, "Pos", "Pos.")
ucb16_pos = change_colname(ucb16_pos, "Pos", "Pos.")
ucb14_pos = change_colname(ucb14_pos, "Pos", "Pos.")

ucd17_pos = change_colname(ucd17_pos, "Pos", "Pos.")
ucd16_pos = change_colname(ucd16_pos, "Pos", "Pos.")
ucd15_pos = change_colname(ucd15_pos, "Pos", "Pos.")
ucd14_pos = change_colname(ucd14_pos, "Pos", "Pos.")
ucd13_pos = change_colname(ucd13_pos, "Pos", "Pos.")


# In[10]:


#function that merge df
#only players who are in both df will be return
def merge_df(df1, df2):
    merge = df1.merge(df2, left_on='No.', right_on='No.')
    return merge


# In[11]:


ucb21_merge = merge_df(ucb21_st, ucb21_pos)
ucb20_merge = merge_df(ucb20_st, ucb20_pos)
ucb19_merge = merge_df(ucb19_st, ucb19_pos)
ucb18_merge = merge_df(ucb18_st, ucb18_pos)
ucb17_merge = merge_df(ucb17_st, ucb17_pos)
ucb16_merge = merge_df(ucb16_st, ucb16_pos)
ucb14_merge = merge_df(ucb14_st, ucb14_pos)

ucsd21_merge = merge_df(ucsd21_st, ucsd21_pos)
ucsd20_merge = merge_df(ucsd20_st, ucsd20_pos)
ucsd19_merge = merge_df(ucsd19_st, ucsd19_pos)
ucsd18_merge = merge_df(ucsd18_st, ucsd18_pos)
ucsd17_merge = merge_df(ucsd17_st, ucsd17_pos)

uci21_merge = merge_df(uci21_st, uci21_pos)
uci20_merge = merge_df(uci20_st, uci20_pos)
uci19_merge = merge_df(uci19_st, uci19_pos)
uci18_merge = merge_df(uci18_st, uci18_pos)
uci17_merge = merge_df(uci17_st, uci17_pos)
uci16_merge = merge_df(uci16_st, uci16_pos)
uci15_merge = merge_df(uci15_st, uci15_pos)
uci14_merge = merge_df(uci14_st, uci14_pos)
uci13_merge = merge_df(uci13_st, uci13_pos)

ucla21_merge = merge_df(ucla21_st, ucla21_pos)
ucla20_merge = merge_df(ucla20_st, ucla20_pos)
ucla19_merge = merge_df(ucla19_st, ucla19_pos)
ucla18_merge = merge_df(ucla18_st, ucla18_pos)
ucla17_merge = merge_df(ucla17_st, ucla17_pos)
ucla16_merge = merge_df(ucla16_st, ucla16_pos)

ucd21_merge = merge_df(ucd21_st, ucd21_pos)
ucd20_merge = merge_df(ucd20_st, ucd20_pos)
ucd19_merge = merge_df(ucd19_st, ucd19_pos)
ucd18_merge = merge_df(ucd18_st, ucd18_pos)
ucd17_merge = merge_df(ucd17_st, ucd17_pos)
ucd16_merge = merge_df(ucd16_st, ucd16_pos)
ucd15_merge = merge_df(ucd15_st, ucd15_pos)
ucd14_merge = merge_df(ucd14_st, ucd14_pos)
ucd13_merge = merge_df(ucd13_st, ucd13_pos)


# In[12]:


#Include Year variable to help divide the trainingset and testset 
ucb21_merge['Year'] = 2021
ucb20_merge['Year'] = 2020
ucb19_merge['Year'] = 2019
ucb18_merge['Year'] = 2018
ucb17_merge['Year'] = 2017
ucb16_merge['Year'] = 2016
ucb14_merge['Year'] = 2014

ucsd21_merge['Year'] = 2021
ucsd20_merge['Year'] = 2020
ucsd19_merge['Year'] = 2019
ucsd18_merge['Year'] = 2018
ucsd17_merge['Year'] = 2017

uci21_merge['Year'] = 2021
uci20_merge['Year'] = 2020
uci19_merge['Year'] = 2019
uci18_merge['Year'] = 2018
uci17_merge['Year'] = 2017
uci16_merge['Year'] = 2016
uci15_merge['Year'] = 2015
uci14_merge['Year'] = 2014
uci13_merge['Year'] = 2013

ucla21_merge['Year'] = 2021
ucla20_merge['Year'] = 2020
ucla19_merge['Year'] = 2019
ucla18_merge['Year'] = 2018
ucla17_merge['Year'] = 2017
ucla16_merge['Year'] = 2016

ucd21_merge['Year'] = 2021
ucd20_merge['Year'] = 2020
ucd19_merge['Year'] = 2019
ucd18_merge['Year'] = 2018
ucd17_merge['Year'] = 2017
ucd16_merge['Year'] = 2016
ucd15_merge['Year'] = 2015
ucd14_merge['Year'] = 2014
ucd13_merge['Year'] = 2013


# In[13]:


#Change column orders and drop variables that are not needed
ucb21 = ucb21_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucb20 = ucb20_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucb19 = ucb19_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucb18 = ucb18_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucb17 = ucb17_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucb16 = ucb16_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucb14 = ucb14_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]

ucsd21 = ucsd21_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucsd20 = ucsd20_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucsd19 = ucsd19_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucsd18 = ucsd18_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucsd17 = ucsd17_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]

uci21 = uci21_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
uci20 = uci20_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
uci19 = uci19_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
uci18 = uci18_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
uci17 = uci17_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
uci16 = uci16_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
uci15 = uci15_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
uci14 = uci14_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
uci13 = uci13_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]


ucla21 = ucla21_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucla20 = ucla20_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucla19 = ucla19_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucla18 = ucla18_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucla17 = ucla17_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucla16 = ucla16_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]

ucd21 = ucd21_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucd20 = ucd20_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucd19 = ucd19_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucd18 = ucd18_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucd17 = ucd17_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucd16 = ucd16_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucd15 = ucd15_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucd14 = ucd14_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]
ucd13 = ucd13_merge[['Pos.', 'Ht.', 'GP', 'MIN', 'FG%', '3PT%', 
                     'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'Year']]


# In[14]:


#Combine all the data to one big df
basketball = pd.concat([ucb21, ucb20, ucb19, ucb18, ucb17, ucb16, ucb14,
                        ucsd21, ucsd20, ucsd19, ucsd18, ucsd17,
                        uci21, uci20, uci19, uci18, uci17, uci16, uci15, uci14, uci13,
                        ucla21, ucla20, ucla19, ucla18, ucla17, ucla16,
                        ucd21, ucd20, ucd19, ucd18, ucd17, ucd16, ucd15, ucd14, ucd13])
basketball.info()


# In[15]:


basketball.replace("-", float("NaN"), inplace=True)
basketball.dropna(subset = ["Ht."], inplace=True)
basketball.dropna(inplace = True)
basketball.shape


# In[16]:


#Converting Ht. to inches 
#New column for Height
new = basketball["Ht."].str.split("-", n = 1, expand = True)

basketball["ft"]= new[0]
basketball["inch"]= new[1]

basketball["ft"] = basketball["ft"].astype(float, errors = 'raise')
basketball["inch"] = basketball["inch"].astype(float, errors = 'raise')
basketball["Height"] = basketball["ft"] * 12 + basketball["inch"]

basketball.drop(columns =["Ht.", "ft", "inch"], inplace = True)


# In[17]:


#Replace values in Pos. column to have constant values
basketball['Pos.'] = basketball['Pos.'].replace({'Forward':'F', 'Guard':'G', 'Center':'C',
                                                 'Forward/Center':'F/C', 'Forward/Guard':'G/F', 
                                                 'C/F':'F/C'})


# In[18]:


basketball


# In[19]:


#Split df into trainingset and testset
#trainingset: 72.6% 
#testset: 27.4%
#the most recently observed data are in testset
basketballtrain = basketball[basketball['Year'] <= 2019]
basketballtest = basketball[basketball['Year'] > 2019]
basketballtrain.drop(['Year'], axis=1, inplace=True)
basketballtest.drop(['Year'], axis=1, inplace=True)

len(basketballtrain), len(basketballtest)


# In[20]:


basketballtest


# In[21]:


#Below is the code for making dummies variable
basketballtrain_dumm = pd.get_dummies(basketballtrain, columns = ['Pos.'], drop_first = True)
basketballtrain_dumm

basketballtest_dumm = pd.get_dummies(basketballtest, columns = ['Pos.'], drop_first = True)
basketballtest_dumm.columns


# In[22]:


y_train = basketballtrain['PTS']
X_train = basketballtrain.drop(['PTS'], axis=1)

y_test = basketballtest['PTS']
X_test = basketballtest.drop(['PTS'], axis=1)


# # 1. Linear Regression

# In[23]:


def OSR2(model, X_test, y_test, y_train):
    y_pred = model.predict(X_test)
    SSE = np.sum((y_test - y_pred)**2)
    SST = np.sum((y_test - np.mean(y_train))**2)
    return (1 - SSE/SST)

def VIF(data, columns):
    values = sm.add_constant(data[columns]).values
    num_col = len(columns) + 1
    vif = [variance_inflation_factor(values, i) for i in range(num_col)]
    return pd.Series(vif[1:], index=columns)


# In[24]:


cols = ['Height', 'GP', 'MIN', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB', 'REB',
       'AST', 'STL', 'BLK','Pos._F', 'Pos._F/C', 'Pos._G',
       'Pos._G/F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

X_ = sm.add_constant(X_)
X_t = sm.add_constant(X_t)

model_1 = sm.OLS(y_, X_).fit()
print(model_1.summary())
print('OSR2:', OSR2(model_1, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# # 1.1 Variable Selection

# In[25]:


#REB - removed / had the largest VIF 
cols = ['Height', 'GP', 'MIN', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL', 'BLK','Pos._F', 'Pos._F/C', 'Pos._G',
       'Pos._G/F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

X_ = sm.add_constant(X_)
X_t = sm.add_constant(X_t)

model_2 = sm.OLS(y_, X_).fit()
print(model_2.summary())
print('OSR2:', OSR2(model_2, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# In[26]:


#Pos._G - removed / had the next largest VIF 
cols = ['Height', 'GP', 'MIN', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL', 'BLK','Pos._F', 'Pos._F/C',
       'Pos._G/F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

X_ = sm.add_constant(X_)
X_t = sm.add_constant(X_t)

model_3 = sm.OLS(y_, X_).fit()
print(model_3.summary())
print('OSR2:', OSR2(model_3, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# In[27]:


#MIN - removed / had the next largest VIF higher than 5
cols = ['Height', 'GP', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL', 'BLK','Pos._F', 'Pos._F/C',
       'Pos._G/F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

X_ = sm.add_constant(X_)
X_t = sm.add_constant(X_t)

model_4 = sm.OLS(y_, X_).fit()
print(model_4.summary())
print('OSR2:', OSR2(model_4, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# In[28]:


#Pos._F/C - removed / insignificant variable / high p-value
cols = ['Height', 'GP', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL', 'BLK','Pos._F',
       'Pos._G/F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

X_ = sm.add_constant(X_)
X_t = sm.add_constant(X_t)

model_5 = sm.OLS(y_, X_).fit()
print(model_5.summary())
print('OSR2:', OSR2(model_5, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# In[29]:


#const - removed / insignificant variable / high p-value
cols = ['Height', 'GP', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL', 'BLK','Pos._F',
       'Pos._G/F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

# X_ = sm.add_constant(X_)
# X_t = sm.add_constant(X_t)

model_6 = sm.OLS(y_, X_).fit()
print(model_6.summary())
print('OSR2:', OSR2(model_6, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# In[30]:


#BLK - removed / insignificant variable / high p-value
cols = ['Height', 'GP', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL','Pos._F',
       'Pos._G/F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

model_7 = sm.OLS(y_, X_).fit()
print(model_7.summary())
print('OSR2:', OSR2(model_7, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# In[31]:


#Pos._G/F - removed / insignificant variable / high p-value
cols = ['Height', 'GP', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL','Pos._F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

model_8 = sm.OLS(y_, X_).fit()
print(model_8.summary())
print('OSR2:', OSR2(model_8, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# In[32]:


#FG% - removed / insignificant variable / high p-value
cols = ['Height', 'GP', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL','Pos._F']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

model_9 = sm.OLS(y_, X_).fit()
print(model_9.summary())
print('OSR2:', OSR2(model_9, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# In[33]:


#Pos._F - removed / insignificant variable / high p-value
cols = ['Height', 'GP', '3PT%', 'FT%', 'OREB', 'DREB',
       'AST', 'STL']
X_ = basketballtrain_dumm[cols]
y_ = basketballtrain_dumm['PTS']

X_t = basketballtest_dumm[cols]
y_t = basketballtest_dumm['PTS']

model_10 = sm.OLS(y_, X_).fit()
print(model_10.summary())
print('OSR2:', OSR2(model_10, X_t, y_t, y_))
VIF(basketballtrain_dumm, cols)


# # 2. Decision Tree Regressor

# In[34]:


from sklearn.preprocessing import OneHotEncoder
drop_enc = OneHotEncoder(drop='first').fit(basketballtrain[['Pos.']])
print(drop_enc.categories_)

# Perform the transformation for both the training and the test set.
X_train_categorical = drop_enc.transform(basketballtrain[['Pos.']]).toarray()
X_train_numerical = X_train[['GP', 'MIN', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'Height']].values

# combine the numerical variables and the one-hot encoded categorical variables
basketballtrain_enc = np.concatenate((X_train_numerical,X_train_categorical), axis = 1)

X_test_categorical = drop_enc.transform(basketballtest[['Pos.']]).toarray()
X_test_numerical = X_test[['GP', 'MIN', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'Height']].values
basketballtest_enc = np.concatenate((X_test_numerical,X_test_categorical), axis = 1)
columns = ['GP', 'MIN', 'FG%', '3PT%', 'FT%', 'OREB', 'DREB', 'REB', 'AST',
       'STL', 'BLK', 'Height', 'F', 'F/C', 'G', 'G/F']


# In[35]:


from sklearn.model_selection import GridSearchCV

grid_values = {'ccp_alpha': np.linspace(0, 2, 500),
               'min_samples_leaf': [5],
               'min_samples_split': [20],
               'max_depth': [30],
               'random_state': [88]} 
            
dtr = DecisionTreeRegressor()
cv = KFold(n_splits=10,random_state=333,shuffle=True) 
dtr_cv = GridSearchCV(dtr, param_grid = grid_values, scoring = 'r2', cv=cv, verbose=1)
dtr_cv.fit(basketballtrain_enc, y_train)


# In[36]:


acc = dtr_cv.cv_results_['mean_test_score']
ccp = dtr_cv.cv_results_['param_ccp_alpha'].data


# In[37]:


#Plot to see which ccp_alpha has a higher validation R2
plt.figure(figsize=(8, 6))
plt.xlabel('ccp alpha', fontsize=16)
plt.ylabel('mean validation R2', fontsize=16)
plt.scatter(ccp, acc, s=2)
plt.plot(ccp, acc, linewidth=3)
plt.grid(True, which='both')
plt.show()
print('Grid best parameter ccp_alpha (max. R2): ', dtr_cv.best_params_['ccp_alpha'])
print('Grid best score (R2): ', dtr_cv.best_score_) 


# In[38]:


#Below is the code for building a decision tree regressor model with ccp_alpha = 0.2004008
dtr2 = DecisionTreeRegressor(min_samples_split=10, 
                            ccp_alpha=0.2004008,
                            random_state = 88)
dtr2 = dtr2.fit(basketballtrain_enc, y_train)
dtr2


# In[39]:


print('Decision Tree Regressor OSR2:', OSR2(dtr2, basketballtest_enc, y_test, y_train))


# In[40]:


print('Node count =', dtr2.tree_.node_count)
plt.figure(figsize=(10,10))
plot_tree(dtr2, 
          feature_names=columns, 
          class_names=['0','1'], 
          filled=True,
          impurity=True,
          rounded=True,
          fontsize=10) 
plt.show()


# In[41]:


pd.DataFrame({'Feature' : columns, 
              'Importance score': 100 * dtr2.feature_importances_}).round(1).sort_values('Importance score', ascending=False)


# # 3. Random Forest Regressor

# In[42]:


from sklearn.model_selection import GridSearchCV

grid_values = {'n_estimators': np.arange(1, 100, 10),
               'max_features': np.linspace(1, 18, 18),
               'min_samples_leaf': [5],
               'min_samples_split': [20],
               'random_state': [88]} 
            
rfr = RandomForestRegressor()
cv = KFold(n_splits=10,random_state=333,shuffle=True) 
rfr_cv = GridSearchCV(rfr, param_grid = grid_values, scoring = 'r2', cv=cv)
rfr_cv.fit(basketballtrain_enc, y_train)


# In[43]:


print('Best n_estimators:', rfr_cv.best_params_['n_estimators'])
print('Best max_features:', rfr_cv.best_params_['max_features'])


# In[44]:


#Below is the code for building a random forest regressor model 
#with n_estimators=81, max_features=1, min_samples_split=20
rfr2 = RandomForestRegressor(n_estimators=81, 
                             max_features= 1,
                             min_samples_split=20,
                            random_state = 88)
rfr2 = rfr2.fit(basketballtrain_enc, y_train)
rfr2


print('Random Forest Regressor OSR2:', OSR2(rfr2, basketballtest_enc, y_test, y_train))


# In[45]:


from sklearn import tree
plt.figure(figsize=(20,10))
_ = tree.plot_tree(rfr2.estimators_[0], feature_names=columns, filled=True,impurity=True,
          rounded=True,fontsize=10)


# In[46]:


pd.DataFrame({'Feature' : columns, 
              'Importance score': 100 * rfr2.feature_importances_}).round(1).sort_values('Importance score', ascending=False)


# # 4. Ridge Regression

# In[66]:


X_train


# In[65]:


y_train


# In[69]:


X_test = X_test.drop(['Pos.'],axis=1)
X_test


# In[70]:


y_test


# In[71]:


alpha_grid = {'alpha': np.logspace(-1, 5, num=50, base=10)}
rr = Ridge(random_state=88)
rr_cv = GridSearchCV(rr, alpha_grid, scoring='neg_mean_squared_error', cv=10)
rr_cv.fit(X_train, y_train)


# In[72]:


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


# In[73]:


print(rr_cv.best_params_)


# In[76]:


real_ridge = Ridge(alpha=6.8664884500430015, fit_intercept=True)
real_ridge.fit(X_train,y_train)
ridge_osr2=OSR2(real_ridge, X_test, y_test, y_train)

rpredict_train=real_ridge.predict(X_train)
rpredict_test=real_ridge.predict(X_test)

ridge_train_mse=mean_squared_error(y_train, rpredict_train)
ridge_test_mse=mean_squared_error(y_test, rpredict_test)

ridge_train_mae=mean_absolute_error(y_train, rpredict_train)
ridge_test_mae=mean_absolute_error(y_test, rpredict_test)

print(ridge_osr2)
print(ridge_train_mse, ridge_test_mse)
print(ridge_train_mae, ridge_test_mae)


# # 5. Lasso Regression

# In[77]:


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


# In[78]:


print(lasso_cv.best_params_)


# In[80]:


real_lasso = Lasso(alpha=0.026826957952797218, fit_intercept=True)
real_lasso.fit(X_train,y_train)
lasso_osr2=OSR2(real_lasso, X_test, y_test, y_train)

lpredict_train=real_lasso.predict(X_train)
lpredict_test=real_lasso.predict(X_test)

lasso_train_mse=mean_squared_error(y_train, lpredict_train)
lasso_test_mse=mean_squared_error(y_test, lpredict_test)

lasso_train_mae=mean_absolute_error(y_train, lpredict_train)
lasso_test_mae=mean_absolute_error(y_test, lpredict_test)

print(lasso_osr2)
print(lasso_train_mse, lasso_test_mse)
print(lasso_train_mae, lasso_test_mae)


# # 6. Final Comparison Table

# In[83]:


#Creating comparison Table 
comparison_data = {'Linear Regression': ['{:.3f}'.format(OSR2(model_10, X_t, y_t, y_)),
                                         '{:.4f}'.format(mean_squared_error(y_t, model_10.predict(X_t))),
                                         '{:.3f}'.format(mean_absolute_error(y_t, model_10.predict(X_t)))],
                    'Decision Tree Regressor': ['{:.3f}'.format(OSR2(dtr2, basketballtest_enc, y_test, y_train)),
                                                '{:.4f}'.format(mean_squared_error(y_test, dtr2.predict(basketballtest_enc))),
                                         '{:.3f}'.format(mean_absolute_error(y_test, dtr2.predict(basketballtest_enc)))],
                   'Lasso Regression' : ['{:.3f}'.format(lasso_osr2),
                                         '{:.4f}'.format((lasso_test_mse)),
                                         '{:.3f}'.format(lasso_test_mae)],
                   'Ridge Regression' : ['{:.3f}'.format(ridge_osr2),
                                         '{:.4f}'.format((ridge_test_mse)),
                                         '{:.3f}'.format(ridge_test_mae)],
                   'Random Forest Regressor': ['{:.3f}'.format(OSR2(rfr2, basketballtest_enc, y_test, y_train)),
                                               '{:.4f}'.format(mean_squared_error(y_test, rfr2.predict(basketballtest_enc))),
                                         '{:.3f}'.format(mean_absolute_error(y_test, rfr2.predict(basketballtest_enc)))],
}

comparison_table = pd.DataFrame(data=comparison_data, index=['OS R-squared', 'Out-of-sample MSE', 'Out-of-sample MAE'])
comparison_table


# # 7. Prediction on scores using Random Forest Regressor

# In[84]:


y_pred0 = rfr2.predict(basketballtrain_enc)
y_pred1 = rfr2.predict(basketballtest_enc)
y_pred0 = y_pred0.astype(int)
y_pred1 = y_pred1.astype(int)
y_pred = np.concatenate((y_pred0, y_pred1))
y_pred


# In[85]:


# Make a dataframe and convert from float to int
predicted_dataset = pd.DataFrame()
predicted_dataset['PTS'] = y_test
predicted_dataset['PRED PTS'] = y_pred1
predicted_dataset

