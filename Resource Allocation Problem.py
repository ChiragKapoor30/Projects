#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pulp as plp
import numpy as np


# In[2]:


location_df  = pd.DataFrame({'location': ['location1', 'location2', 'location3'],
                             'max_resource':[500, 600, 250]
                             })
work_df  = pd.DataFrame({'work': ['work1', 'work2'],
                             'min_resource':[550, 300]
                             })
resource_cost = np.array([[150,200], [220,310], [210,440]])


# In[3]:


model = plp.LpProblem("Resource_allocation_prob", plp.LpMinimize)


# In[4]:


no_of_location = location_df.shape[0]
no_of_work = work_df.shape[0]
x_vars_list = []
for l in range(1,no_of_location+1):
    for w in range(1,no_of_work+1):
        temp = str(l)+str(w)
        x_vars_list.append(temp)
x_vars = plp.LpVariable.matrix("R", x_vars_list, cat = "Integer", lowBound = 0)
final_allocation = np.array(x_vars).reshape(3,2)
print(final_allocation)
res_equation = plp.lpSum(final_allocation*resource_cost)
model +=  res_equation


# In[5]:


for l1 in range(no_of_location):
    model += plp.lpSum(final_allocation[l1][w1] for w1 in range(no_of_work)) <= location_df['max_resource'].tolist()[l1]
for w2 in range(no_of_work):
    model += plp.lpSum(final_allocation[l2][w2] for l2 in range(no_of_location)) >= work_df['min_resource'].tolist()[w2]


# In[6]:


print(model)


# In[7]:


model.solve()
status = plp.LpStatus[model.status]
print(status)


# In[8]:


print("Optimal overall resouce cost: ",str(plp.value(model.objective)))
for each in model.variables():
    print("Optimal cost of ", each, ": "+str(each.value()))

