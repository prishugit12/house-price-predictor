#!/usr/bin/env python
# coding: utf-8

# # Saving the model

# In[42]:


from joblib import dump, load
dump(model, 'DragonData.joblib')


# # Testing the model on test data

# In[43]:


X_test=strat_test_set.drop("MEDV", axis=1)
Y_test=strat_test_set["MEDV"].copy()


# In[44]:


import numpy as np
from sklearn.metrics import mean_squared_error
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test, final_predictions)
final_rmse=np.sqrt(final_mse)


# In[45]:


final_rmse


# In[46]:


print(final_predictions,list(Y_test))

