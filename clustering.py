#Reference
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause
#Modified as per need:Pramit Choudhary

from __future__ import print_function
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
import pylab as pl

import logging
from optparse import OptionParser
import sys
import os
from time import time
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(abs_path, 'pramit_data')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

###############################################################################
# Load some categories from the training set(borrowed from 20_news_group)
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]

# Read the text file from the directory
text_files = os.listdir(path)
file_path = []
for dir_entry in text_files:
    file_path.append( os.path.join(path,dir_entry))

dataset = [str.decode(open(f).read(), "UTF-8", "ignore") for f in file_path]

print("%d documents" % len(dataset))
print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()

if opts.use_hashing:
    print(opts.use_hashing)
    if opts.use_idf:
        #print(opts.use_idf)
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = Pipeline((
            ('hasher', hasher),
            ('tf_idf', TfidfTransformer())
        ))
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 stop_words='english', use_idf=opts.use_idf)
X = vectorizer.fit_transform(dataset)
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    lsa = TruncatedSVD(opts.n_components)
    X = lsa.fit_transform(X)
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    X = Normalizer(copy=False).fit_transform(X)

    print("done in %fs" % (time() - t0))
    print()


###############################################################################
# Do the actual clustering
n_clusters = 2
km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)                
print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
k_means_labels = km.labels_
k_means_cluster_centers = km.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)
print("done in %0.3fs" % (time() - t0))
print("unique Labels")
print(k_means_labels_unique)
print()
print("Kmeans-Centroids")
print(k_means_cluster_centers)
print()
print("Labelling result after using K-means")
#Note: Kmeans clustering needs to be run for iteratively for the results to converge. This is more
# of a sample on how it can be done. I need more data to produce better results.
# There is hardly any similarity between the documents. Lets talk about it more.
print(k_means_labels)

# Create a dict to map the labels to the docs 
K_means_label_matching = {}
count =0;
for file_names in text_files: 
   K_means_label_matching[file_names] = k_means_labels[count]
   count+=1
print("Document Mapping to kmeans results")
print(K_means_label_matching)
print()
