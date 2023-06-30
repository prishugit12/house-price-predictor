#!/usr/bin/env python
# coding: utf-8

# # Dragon real estate price predictor
# 

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("housing_Data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


# # Train test splitting 

# In[8]:


import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[9]:


train_set, test_set=split_train_test(housing, 0.2)


# In[10]:


print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split
train_set, test_set=train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set['CHAS'].value_counts()


# In[15]:


strat_train_set['CHAS'].value_counts()


# In[16]:


95/7


# In[17]:


376/28


# In[18]:


housing=strat_train_set.copy()


# # Looking for correlations

# In[19]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[20]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize =(12,8))


# In[21]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8 )


# # Trying out attribute combinations

# In[22]:


housing ["TAXRM"] = housing['TAX']/housing['RM']


# In[23]:


housing.head()


# In[24]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[25]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8 )


# In[26]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# # Missing attributes

# In[27]:


# To care of missing attributes , we have three options 
#  1. get rid of the missing data points
#  2. get rid of the whole attribute
#  3. set the value to soome value(0, mean or median)


# In[28]:


a=housing.dropna(subset=["RM"]) #option-1


# In[29]:


a.shape


# In[30]:


housing.drop("RM", axis=1).shape


# In[31]:


median = housing["RM"].median()
housing["RM"].fillna(median)


# In[32]:


housing.shape


# In[33]:


housing.describe()


# In[34]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[35]:


imputer.statistics_


# In[36]:


X = imputer.transform(housing)


# In[37]:


housing_tr = pd.DataFrame(X,columns=housing.columns)


# In[38]:


housing_tr.describe()


# # Creating a pipeline

# In[39]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])


# In[40]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[41]:


housing_num_tr.shape


# # Selecting a desired model for Dragon real estate 

# In[42]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model = LinearRegression()
#model = DecisionTreeRegressor()  
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[43]:


some_data=housing.iloc[:5]


# In[44]:


some_labels=housing_labels.iloc[:5]


# In[45]:


prepared_data=my_pipeline.transform(some_data)


# In[46]:


model.predict(prepared_data)


# In[47]:


list(some_labels)


# # Evaluating the model

# In[48]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[49]:


rmse


# # Using better evluation technique- Cross validation

# In[50]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error",cv=10)
rmse_scores = np.sqrt(-scores)


# In[51]:


rmse_scores


# In[52]:


def print_scores(score):
    print("Scores:", score)
    print("Mean:", score.mean())
    print("Standard deviations:", score.std())


# In[53]:


print_scores(rmse_scores)


# # Saving the model

# In[54]:


from joblib import dump, load
dump(model, 'DragonData.joblib')


# # Testing the model on test data

# In[55]:


X_test=strat_test_set.drop("MEDV", axis=1)
Y_test=strat_test_set["MEDV"].copy()


# In[56]:


X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test, final_predictions)
final_rmse=np.sqrt(final_mse)


# In[57]:


final_rmse


# In[58]:


print(final_predictions,list(Y_test))


# In[ ]:




