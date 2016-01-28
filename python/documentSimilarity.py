_author__ = "Pramit Choudhary"

import os
from itertools import izip
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
abs_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(abs_path, 'data')

def sort_coo(m):
    tuples = izip(m.row, m.col, m.data)
    return sorted(tuples, key=lambda x: (x[0], x[2]),reverse=True)

def find_similarity():
    # Read the text file from the directory
    text_files = os.listdir(path)
    file_path = []
    for dir_entry in text_files:
        file_path.append( os.path.join(path,dir_entry))

    documents = [str.decode(open(f).read(), "UTF-8", "ignore") for f in file_path]
    tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                            stop_words='english').fit_transform(documents)
    # no need to normalize, since Vectorizer will return normalized tf-idf
    pairwise_similarity = tfidf * tfidf.T
    pairwise_matrix = coo_matrix(pairwise_similarity)
    sorted_similarity = defaultdict(list)
    
    ''' Sort the pairiwse matrix in to a dict for better readability '''
    file_handle = open("SimilarityLog.txt", "w")
    for i in sort_coo(pairwise_matrix): 
        if i[2] > 0.1:
            sorted_similarity[i[0]].append((i[1], i[2]))
    # Lazy write to file
    file_handle.write(str(sorted_similarity))
    file_handle.close()

# main
def main():
    try:
        find_similarity()
    except IOError as e:
        print ("I/O error({0}): {1}".format(e.errno, e.strerror))

if __name__ == "__main__":
    main()
