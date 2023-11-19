#!/usr/bin/env python
# coding: utf-8

# In[46]:


#Importing Libraries
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# In[47]:


#Importing Data
df = pd.read_csv("E:\\Project\\Tata\\Online Retail.csv")


# In[48]:


df


# In[49]:


##Data Cleaning
df.dropna(subset = ['InvoiceNo'], inplace = True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]
df.head()


# In[50]:


df['Description'] = df['Description'].str.strip()


# In[51]:


df['Country'].value_counts()


# In[52]:


df.shape


# In[53]:


df.info


# In[54]:


df.head()


# In[59]:


#Seperating transactions for Germany
mybasket = (df[df['Country']=="Germany"]
           .groupby(['InvoiceNo','Description'])['Quantity']
           .sum().unstack().reset_index().fillna(0)
           .set_index('InvoiceNo'))


# In[60]:


mybasket.head()


# In[61]:


df['Country'].value_counts()


# In[62]:


#Coverting all positive values to 1 and everything else to 0
def my_encode_units(x):
    if x <= 0:
        return 0
    if x >= 0:
        return 1

my_basket_sets = mybasket.applymap(my_encode_units)
my_basket_sets.drop('POSTAGE', inplace = True, axis = 1)


# # Training Model

# In[66]:


#Generating frequent itemsets
my_frequent_itemsets = apriori(my_basket_sets, min_support = 0.07, use_colnames= True)


# In[67]:


#Generating Rules
my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=1)


# In[68]:


#Viewing Top 100 rules
my_rules.head(100)


# # Making Recommendations

# In[69]:


my_basket_sets['ROUND SNACK BOXES SET OF4 WOODLAND'].sum()


# In[70]:


my_basket_sets['SPACEBOY LUNCH BOX'].sum()


# In[77]:


#Filtering rules based on condition
my_rules[(my_rules['lift'] >= 2)&
        (my_rules['confidence'] >= 0.3)]


# In[ ]:




