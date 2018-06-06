import os
from boto.s3.connection import S3Connection
import pickle
import spacy
import numpy as np
import warnings
import Comment
import dill

def add_to_cache(filepath, key):
    """
    Saves a local file to S3.
    filepath is the full local path to the file.
    key is the S3 key to use in ds_cache bucket
    """
    if is_s3_cache_available():
        print("saving to s3")
        s3_key(key, new=True).set_contents_from_filename(filepath)
        
def write_obj_to_cache(obj, filepath, key, use_s3=True):
    """
    Writes a python object to a file, and also stores that file in S3.
    filepath is the full local path to the file.
    key is the S3 key to use in ds_cache bucket
    """
    pickle.dump(obj, open(filepath, "wb"))
    add_to_cache(filepath, key)

def is_s3_cache_available():
    """
    Return True if a connection can be made to S3 in the current environment
    """
    try:
        S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        return True
    except:
        print("WARNING: Unable to connect to s3")
        return False
def s3_key(key, new=False):
    """
    key is the S3 key in the ds_cache bucket.  This function returns a reference
    to the boto.s3.Key object corresponding to the key parameter.
    If new=True, create a new key.  Otherwise return an existing key.
    If the key doesn't exist, return None
    """
    s3 = S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    cache_bucket = s3.get_bucket('ds_cache')
    if new:
        return cache_bucket.new_key(key)
    return cache_bucket.get_key(key)

def load_cache(filepath, key):
    """
    Loads file into local cache and returns the path.  Returns None if the file
    is not available.
    filepath is the full local path to the file.
    key is the S3 key to use in ds_cache bucket
    """
    #if os.path.exists(filepath):
    #    print("file exists in cache")
    #    return filepath
    if is_s3_cache_available():
        if s3_key(key) is not None:
            print("transferring from s3")
            s3_key(key).get_contents_to_filename(filepath)
            return filepath
    return None

def read_obj_from_cache(filepath, key):
    """
    Reads object from local cache.  Returns None if the file
    is not available.
    filepath is the full local path to the file.
    key is the S3 key to use in ds_cache bucket
    """
    in_cache = load_cache(filepath, key)
    if in_cache:
        return pickle.load(open(in_cache, "rb"))
    return None


AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

class Model(object):
	def __init__(self):
		self.key = None
		self.path = None
		self.container = None
		self.ID = None
		self.model = None
		self.type = None
		self.package_version = None
		self.preprocessing_dag = None
		self.comments = None
		self.is_loaded = False

	def load(self, ID):
		if ID == 'sentiment':
			self.key = s3_key('{}-model'.format(ID))
			self.path = '{}-model.pkl'.format(ID)
			storage_object = read_obj_from_cache(self.path, self.key)
			self.__package_old__(storage_object)
			del storage_object
			
		else:
			warnings.warn("Model not found")
			return None

	def __package_old__(self, storage_object):
		self.ID = storage_object.ID
		self.comments = storage_object.comments
		self.model = storage_object.model
		self.type = storage_object.type
		self.package_version = storage_object.package_version
		self.preprocessing_dag = storage_object.preprocessing_dag
		self.key = storage_object.key
		self.path = storage_object.path
		self.is_loaded = True		

	def package_new(self, ID, model, type, version, dag):
		self.ID = ID
		self.comments = Comment.Thread()
		self.model = model
		self.type = type
		self.package_version = version
		self.preprocessing_dag = dag
		self.key = s3_key('{}-model'.format(ID), new=True)
		self.path = '{}-model.pkl'.format(ID)
		self.is_loaded = True

	def predict(self, x):
		if self.is_loaded:
			processor = self.preprocessing_dag()
			if hasattr(x, '__iter__'):
				results = np.array(map(lambda t: self.model.predict_proba(processor(t))[0],x))
			else:
				results = np.array(self.model.predict_proba(processor(x))[0])
			return results
		else:
			raise ValueError("Model not loaded")


	def save(self):
		if self.ID is None:
			raise ValueError("You need to load or package a model")
		else:
			write_obj_to_cache(self, self.path, self.key)