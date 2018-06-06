
# coding: utf-8

# In[1]:

import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
from __future__ import print_function


# In[2]:

from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']


# In[3]:

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)


# In[46]:

len(train_vectors.data)


# In[4]:

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)


# In[5]:

pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')


# In[6]:

from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)


# In[7]:

print(c.predict_proba([newsgroups_test.data[0]]))


# In[63]:

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# In[64]:

idx = 83
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=10)
print("value to be predicted")
print(newsgroups_test.data[idx])
print(newsgroups_test.target[idx])
print(newsgroups_test.target_names)


# In[17]:

print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])


# In[18]:

# Explanation of the method
get_ipython().magic(u'pinfo explainer.explain_instance')


# In[20]:

# ravel is used for array flattening
x = np.array([[1, 2, 3], [4, 5, 6]])
np.ravel(x)


# In[22]:

import sklearn.metrics.pairwise


# In[30]:

X = [[0, 1], [1, 1]]
sklearn.metrics.pairwise_distances(X, X, metric='cosine')


# In[65]:

exp.as_list()
#exp.show_in_notebook(text=False)


# In[35]:

print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])


# In[66]:

get_ipython().magic(u'matplotlib inline')
fig = exp.as_pyplot_figure()


# # Numeric Format

# In[49]:

import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
from __future__ import print_function
np.random.seed(1)


# In[50]:

iris = sklearn.datasets.load_iris()


# In[51]:

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)


# In[52]:

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)


# In[53]:

sklearn.metrics.accuracy_score(labels_test, rf.predict(test))


# In[101]:

import pandas as pd
pd.DataFrame(test).head()


# In[115]:

explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=iris.feature_names, 
                                                   class_names=iris.target_names, discretize_continuous=True)
i = np.random.randint(0, test.shape[0])
exp_num = explainer.explain_instance(test[i], rf.predict_proba, num_features=4, top_labels=1)


# In[103]:

exp_num.predict_proba


# In[104]:

exp_num.class_names


# In[116]:

exp_num.as_list()

