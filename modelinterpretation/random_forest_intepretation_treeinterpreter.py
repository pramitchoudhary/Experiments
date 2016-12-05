
# coding: utf-8

# In[1]:

# Reference: https://github.com/andosa/treeinterpreter
# Blog: http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/

from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# In[2]:

from sklearn.datasets import load_boston
boston = load_boston()
rf = RandomForestRegressor()


# In[13]:

boston.data[:300,].shape


# In[16]:

rf = RandomForestRegressor()
fit1 = rf.fit(boston.data[:300], boston.target[:300])


# In[17]:

fit1


# In[37]:

instances = boston.data[[300, 309]]
print "Instance 0 prediction:", rf.predict(instances[0].reshape(1,13))
print "Instance 1 prediction:", rf.predict(instances[1].reshape(1,13))


# In[38]:

prediction, bias, contributions = ti.predict(rf, instances)


# In[40]:

for i in range(len(instances)):
    print "Instance", i
    print "Bias (trainset mean)", bias[i]
    print "Feature contributions:"
    for c, feature in sorted(zip(contributions[i], 
                                 boston.feature_names), 
                             key=lambda x: -abs(x[0])):
        print feature, round(c, 2)
    print "-"*20


# In[42]:

print prediction
print bias + np.sum(contributions, axis=1)


# In[43]:

#  the basic feature importance feature provided by sklearn
fit1.feature_importances_


# In[44]:

# treeinterpreter uses the apply function to retrieve the leave indicies with the help of which, 
# the tree path is retrieved

rf.apply


# In[47]:

rf.apply(instances)


# In[ ]:



