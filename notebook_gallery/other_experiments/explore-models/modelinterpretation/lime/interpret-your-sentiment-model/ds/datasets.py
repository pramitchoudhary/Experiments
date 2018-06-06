import nltk
import os
from boto.s3.connection import S3Connection
import pickle
import warnings
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

def nltk_corpus(corpus_name):
    corpus = getattr(nltk.corpus, corpus_name)
    try:
        corpus.ensure_loaded()
    except:
        nltk.download(corpus_name)
    return corpus

def _get_creds_from_id(corpus_id):
	pass

def load(corpus_id, from_nltk = True):
	if from_nltk:
		corpus = nltk_corpus(corpus_id)
	else:
		key, path = _get_creds_from_id(corpus_id)
		corpus = read_obj_from_cache(path, key)
	return corpus