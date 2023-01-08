#!/usr/bin/env python
# coding: utf-8

# # Productivity Performance of Garment-Production Employees

# **Background**

# The Garment Industry is one of the key examples of the industrial globalization of this modern era. It is a highly labour-intensive industry with lots of manual processes. Satisfying the huge global demand for garment products is mostly dependent on the production and delivery performance of the employees in the garment manufacturing companies. So, it is highly desirable among the decision makers in the garments industry to track, analyse and predict the productivity performance of the working teams in their factories

# **Objective**
# 

# To make predictions about garment employees productivity

# **Dataset**

# The dataset was obtained from https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees

# | Column | Description |
# | :----------- | :----------- |
# | Date | Date in MM-DD-YYYY |
# | Quarter | A portion of the month. A month was divided into four quarters |
# | Department | Character, the department associated |
# | Team_no. | Number, the associated team number |
# | No. of_Workers | Number of workers in each team |
# | No. of style change | Number of changes in the style of a particular product |
# | Targeted productivity | Targeted productivity set by the Authority for each team for each day |
# | SMV | Standard Minute Value, it is the allocated time for a task |
# | WIP | Work in progress. Includes the number of unfinished items for products |
# | Overtime | Represents the amount of overtime by each team in minutes |
# | Incentive | Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action |
# | Idle Time | The amount of time when the production was interrupted due to several reasons |
# | Idle Men | The number of workers who were idle due to production interruption |
# | Actual Productivity | The actual % of productivity that was delivered by the workers. It ranges from 0-1.|
# 

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler


# In[2]:


#Read csv file to dataframe
garmentsData = pd.read_csv("/Users/kgaug/OneDrive/My Projects/garments_worker_productivity.csv")


# In[3]:


#Dataset Inspection
garmentsData.head()


# In[4]:


#Dataset Overview shows total columns (15), rows (1197) and datatypes
garmentsData.info()


# In[5]:


#Check for unique Values
garmentsData.nunique()


# In[6]:


#Check for null values
garmentsData.isnull().sum()


# ### Data Cleaning
# 

# 
# - Only the WIP variable has missing values
# - These will be replaced with 0, indicating that these products which are finished and therefore do not have a wip value
# - Spelling checks are corrected ('sweing' to 'sewing')
# - White spaces are also checked and removed as part of data cleaning

# In[7]:


#Replace missing values with 0
garmentsData = garmentsData.fillna(0)
garmentsData.head()


# In[8]:


#Replace 'sweing' with 'sewing'
garmentsData.replace({'department' : {'sweing': 'sewing'}}, inplace = True)
garmentsData.head()


# In[9]:


#Check Departments 
garmentsData['department'].value_counts()


# In[10]:


#Check Departments Category labels
garmentsData['department'].unique()


# In[11]:


#Remove trailing white spaces
garmentsData['department'] = garmentsData['department'].str.strip()
garmentsData['department'].unique()


# ### Data Exploration

# In[12]:


#Summary Statistics of continuous variables
garmentsData.describe().transpose()


# In[13]:


#Examine data distribution for Work in Progress and Overtime
fig, axes = plt.subplots(1, 2, figsize=(15,8))

ax1 = sns.histplot(data=garmentsData['wip'], ax = axes[0], color = 'purple');
ax1.set(xlabel = 'Work in Progress (products)', title='Work In Progress');

ax1 = sns.histplot(data=garmentsData['over_time'], ax = axes[1]);
ax1.set(xlabel = 'Overtime (minutes)', title='Overtime');


# In[14]:


#Examine Categorical variables
garmentsData['department'].value_counts().plot.pie(explode= [0.05, 0.05], autopct = '%1.1f%%', shadow=True, figsize=(7,5))
plt.title('Department Percentage');


# In[15]:


#Examine Categorical variables
fig, axes = plt.subplots(1, 2, figsize=(15,8))

ax1 = sns.boxenplot(x = garmentsData['department'], y = garmentsData['actual_productivity'], ax = axes[0]);
ax1.set(xlabel = 'Departments', title='Productivity by Department');

ax1 = sns.boxenplot(x = garmentsData['day'], y = garmentsData['actual_productivity'], ax = axes[1]);
ax1.set(xlabel = 'Day', title='Productivity by Day');


# In[16]:


# Plot Correlations matrix
plt.figure(figsize=(20,8))
corrMatrix = garmentsData.corr()

# Plot the correlation coefficient for all column pairs
sns.heatmap(corrMatrix, annot=True,linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()


# ### Exploration Summary

# - From the summary statisctics, *wip* and *overtime* revealed the largest std deviation, 
#  indicating the wide spread of observations from the mean.
# - To visualise the distribution of *wip* and *overtime*, the Histogram was generated which showed positive skewness and thus
# implying that the data is unevenly distributed.
# - The pie chart reveals that the *sewing* department outnumbers the *finishing* department by 15.4%
# - The 1st boxenplot shows that on a productivity range of 0 to 1, the *finishing* team are largely clustered above the mean of apprx. 0.8.
# Whereas the *sewing* department occupy a larger density below the mean of apprx 0.7. It appears that the former department is 
# more productive despite their lower volume in comparison to the latter. 
# - The 2nd boxenplot shows that *Saturday* is most productive as the density is more intense with bigger coverage above the mean
# of apprx 0.8.
# 

# **Correlations**
# - 2 Variable pairs reveal a strong positive correlation where the correlation coefficient is between 0.7 and 0.9
# 1. *Number of workers* and *standard minute value* , (coefficient = 0.9)
# 2. *Number of workers* and *overtime* , (coefficient = 0.7)
# - A moderate correlation exists between *idle men* and *idle time* , (coefficient = 0.5)
# - A moderate correlation exists between *overtime* and *standard minute value* , (coefficient = 0.6)
# - Other variables are either have a low positive correllation or are negatively correlated.

# In[17]:


#Drop non-required columns
garmentsData.drop(['date','quarter','day'], axis=1, inplace=True)


# In[18]:


#Encode department column to numbers for model fitting
garmentsData['department'] = np.where(garmentsData['department'] == 'finishing',1,0)
garmentsData.head()


# In[19]:


#Create target and Features
#np.random.seed(42)
target = 'actual_productivity'

X = garmentsData.drop(columns= target)
y = garmentsData[target]

X.head()


# In[20]:


#visualise target variable distribution
plt.hist(garmentsData['actual_productivity'])
plt.xlabel('Actual Productivity')
plt.ylabel('Frequency')
plt.show()


# In[21]:


#Perform 80/20 data split
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state = 42)


# In[22]:


X_train.shape, y_train.shape


# In[23]:


X_val.shape, y_val.shape


# In[24]:


#Transform data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# ### Random Forest Regression Model
# **This model was selected for the following reasons:**
# - Handles non-linear relationships efficiently
# - The sci-kit learn ensemble combines multiple decision trees to produce an aggregated result, which is more accurate than the normal linear regression

# In[25]:


#Model definition
model = RandomForestRegressor(n_estimators=100,random_state=42)


# In[26]:


#Build Training Model
model.fit(X_train,y_train)


# In[27]:


model.score(X_val,y_val)


# In[28]:


model.score(X_train,y_train)


# In[29]:


#Apply trained model to make predictions on the validation data
y_pred = model.predict(X_val)


# In[30]:


#Show model performance
#print('Coefficients:', model.coef_)
#print('Intercept:', model.intercept_)
print('Mean Absolute Error (MAE): %.2f'% mean_absolute_error(y_val,y_pred))
print('Mean Squared Error (MSE): %.2f'% mean_squared_error(y_val,y_pred))
print('Root Mean Squared Error (RMSE):',np.sqrt(mean_squared_error(y_val,y_pred)))
print('Explained Variance Score: %.2f'% explained_variance_score(y_val,y_pred))
print('Coefficient of determination (R^2): %.2f' % r2_score(y_val,y_pred))


# In[31]:


#Observed vs Predicted values
plt.figure(figsize=(20,8))
ax = range(len(X_val))

plt.plot(ax, y_val, label='Observed', color = 'purple', linestyle='-')
plt.plot(ax, y_pred, label='Predicted', color = 'green', linestyle='--')


# In[32]:


#Visualise Feature Importance
from sklearn.pipeline import Pipeline
feature_list = list(X.columns)
featureImportance = pd.Series(model.feature_importances_,index=feature_list,name='Importance').sort_values(ascending=True)
ax = featureImportance.plot(kind='barh')


# In[33]:


#Visualise Prediction error
from yellowbrick.regressor import PredictionError
visualizer = PredictionError(model)
visualizer.fit(X_train,y_train)
visualizer.score(X_val,y_val)
visualizer.poof()


# ### Model Evaluation
# - The MAE (0.07) and MSE (0.01) indicate a low margin of error
# - The RMSE (0.1) is an indication of a good-fitting model
# - The Explained Variance Score (57%) means there's a high discrepancy between the actual data and the model data. Ideally, we want this number to be as close to 100% as possible
# - The Coefficient of Determination (r^2) explains that 56% of the model output can be explained by the independant variables, the remaining 44% is explained by factors outside of our dataset,
# like the explained variance score, we would like the r^2 to be as close to 100% as possible
# - To obtain a clearer understanding of the model, I visualised the Observed vs Predicted values
# - The feature importance bar graph explain how the independant variables contributed to the model, with *targeted productivity* being the most
# important and *number of style change* being the least important. When improving the model, the latter may have to be excluded for better accuracy
# -The Prediction Error scatterplot clearly shows the margin of error between the best fit line and identity

# **Cross-Validation**

# In[34]:


#Set the seed to obtain the same results each time cross_validation is performed
seed = 7


# In[35]:


#Perform Cross Validation between training and testing data 
#Training data crossvalidation
from sklearn.model_selection import cross_val_score
score_train = cross_val_score(model,X_train,y_train, scoring='neg_mean_squared_error',cv=10)
score_train


# In[36]:


#Print absolute mean score
from numpy import absolute
print(absolute(np.mean(score_train)))
#absScore = absolute(np.mean(score_train))
#print(f'Accuracy: {absScore*100:.2f}%')


# In[37]:


#Testing data cross validation
score_val = cross_val_score(model,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
score_val


# In[38]:


print(absolute(np.mean(score_val)))
#print(f'Accuracy: {np.mean(score_val)*100:.2f}%')


# ### Cross-Validation Summary
# - Cross-validation was performed on the training data and validation data where several metrics were explored including accuracy score, r^2, MSE and MAE
# - MSE produced better results (0.01), thus indicating how well the model performs on new data 

# ### Conclusion
# - The model cannot be perfectly explained by the independant variables as shown by the coefficient of determination.
# - However, the MSE, MAE indicate an acceptable margin of error, this is corroborated by the error prediction scatterplot.
# - The employer can therefore use the Random Forest Regression model to predict employee productivity with moderate accuracy.
