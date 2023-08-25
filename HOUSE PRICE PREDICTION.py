#!/usr/bin/env python
# coding: utf-8

# # HOUSE PRICE PREDICTION

# Let's begin by importing the required libraries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[2]:


import pandas as pd
data=pd.read_csv("E:\\DATA\\MagicBricks.csv")


# In[3]:


print("Shape of Dataset:", data.shape)
data.head()


# In[4]:


data = data.rename(columns={'Price/100000': 'Price1'})


# In[5]:


data


# ### Understanding the Dataset

# In[6]:


# Data Information
data.info()


# In[7]:


# Descriptive statistics
data.describe()


# ### Remove Outliers in Target Variable

# In[8]:


# Show the distribution of House price
sns.distplot(data['Price1'],color="Blue")


# In[ ]:


#The distribution is right-skewed. Let's remove outliers using the IQR as the criteria.


# In[9]:


# Find IQR
Q1 = data['Price1'].quantile(0.25)
Q3 = data['Price1'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[10]:


# Remove outliers with a criteria: 1.5 x IOR
data = data[~((data['Price1'] < (Q1 - 1.5 * IQR)) |(data['Price1'] > (Q3 + 1.5 * IQR)))]
data.shape


# In[11]:


# Show the distribution of price: outliers removed
sns.distplot(data['Price1'], color="blue")


# In[12]:


numeric_vars = ['Price','Price1','Area','BHK','Bathroom','Parking','Per_Sqft']
cat_fields = ['Furnishing', 'Locality', 'Status', 'Transaction', 'Type']


# ### Exploration of Continuous Variables

# In[13]:


# Create a list of continuous variables
cont = ["Price1", "Area", "BHK", "Parking", "Bathroom", "Per_Sqft"]

# Create a dataframe of continuous variables
data_cont = data[cont]
data_cont


# In[14]:


# Visualize correlation between continuous variables

# Compute the correlation matrix
corr = data_cont.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 6))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap="winter", vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)


# In[15]:


df=data[["Price1", "Area", "BHK", "Parking", "Bathroom", "Per_Sqft"]]
df


# In[16]:


# Checking for missing values
df.isnull().sum()


# ### Defining dependent and independent variables

# In[17]:


df1=df.fillna("0")


# In[18]:


# Checking for missing values
df1.isnull().sum()


# In[19]:


df1


# In[24]:


x=df[['Area', 'BHK', 'Bathroom']]
y=df[['Price1']]


# In[25]:


print(x)
print(y)


# ### Checking for linear relationship between dependent and independent variables.

# In[26]:


fig1, (ax1,ax2)=plt.subplots(2,2,figsize=(15,12),sharex=True)
ax1[0].scatter(df['Price1'],df['Area'])
ax1[0].set_xlabel('Price1'); ax1[0].set_ylabel('Area')
ax1[1].scatter(df['Price1'],df['BHK'])
ax1[1].set_xlabel('Price1'); ax1[1].set_ylabel('BHK')
ax2[0].scatter(df['Price1'],df['Bathroom'])
ax2[0].set_xlabel('Price1'); ax2[0].set_ylabel('Bathroom')
ax2[1].scatter(df['Price1'],df['Parking'])
ax2[1].set_xlabel('Price1'); ax2[1].set_ylabel('Parking')
ax2[1].scatter(df['Price1'],df['Per_Sqft'])
ax2[1].set_xlabel('Price1'); ax2[1].set_ylabel('Per_Sqft')


# All the independent variables shows a linear relationship with the dependent variable.

# In[27]:


x = sm.add_constant(x)
ft = sm.OLS(y, x).fit()
ft.summary()


# ### Checking For Multi-Collinearity

# In[28]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(x):
    vif=pd.DataFrame()
    vif['variables'] = x.columns
    vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    
    return(vif)


# In[29]:


calc_vif(x)


# All VIF values are lower than 5. So, there is no possibility of multicollinearity.

# In[30]:


#Sparse Matrix Correlation
df.corr()


# ### Model Evaluation

# In[31]:


# split into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=42)
from sklearn.linear_model import LinearRegression

# Create the model
model = LinearRegression()

# Fit the model using inputs and targets
model.fit(x_train, y_train)

# Make prediction
y_test_pred = model.predict(x_test)

# Performance metrics
lr_r2= r2_score(y_test, y_test_pred)
lr_mae = mean_absolute_error(y_test, y_test_pred)

# Show the metrics
print("Linear Regression R2: ", lr_r2)
print("Linear Regression MAE: ", lr_mae)


# In[32]:


y_test_pred


# The R Square value is 53% approximately which shows multicollinearity is not present in the model.

# ### Checking for Autocorrelation

# In[33]:


plt.figure(figsize=(8,5))
sns.residplot(y_test,y_test_pred)
plt.ylabel("Residuals")
plt.xlabel("Predicted Values")
plt.title("Residual Plot for Autocorrelation")
plt.show()


# In[34]:


from statsmodels.stats.stattools import durbin_watson

test_stats_val=durbin_watson(resids=y_test_pred-y_test)
print("The Durbin Watson Test Statistics value is:",test_stats_val.round(2))

The Durbin-Watson statistic will always have a value ranging between 0 and 4. A value of 2.0 indicates there is no autocorrelation detected in the sample. Values from 0 to less than 2 point to positive autocorrelation and values from 2 to 4 means negative autocorrelation. 
We can observe that the test statistics value for our model is 2, indicating no autocorrelation in our model
# ### Checking for Heteroscedasticity

# It states:
# - H0=No Heteroscedasticity
# - H1=Heteroscedasticity
#     
#     i.e. if p value is lower than 0.05, we reject the null hypothesis and conclude the presence of heteroscedasticity in our model.

# In[35]:


from statsmodels.stats.diagnostic import het_goldfeldquandt

het_goldfeldquandt(y, x)

This shows that heteroscedasticity is not present in the model. 
# ### Plotting the Results

# In[36]:


plt.scatter(y_test,y_test_pred)
plt.title('Actual vs Predicted')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')


# In[37]:


sns.displot(y_test_pred-y_test,kind='kde')

