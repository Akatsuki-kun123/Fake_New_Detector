# import required module
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("Data/sample.csv", index_col=0)

# merge documents into a single corpus
doc = data["text"].iloc[0:1]
doc2 = data["text"].iloc[1:2]

# create object
tfidf = TfidfVectorizer()

# get tf-df values
result = tfidf.fit_transform(doc.values.astype('U'))
result_array = result.toarray()

result = tfidf.fit_transform(doc2.values.astype('U'))
result_array2 = result.toarray()

"""
# get indexing
print('\nWord indexes:')
print(tfidf.vocabulary_)

# display tf-idf values
print('\ntf-idf value:')
print(result)
"""

# in matrix form
print("\ntf-idf values in matrix form:")
print(result_array.shape[0], result_array.shape[1])
print(result_array2.shape[0], result_array2.shape[1])

"""
m = result.toarray()
rez = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

print("\n")
print(m)
"""