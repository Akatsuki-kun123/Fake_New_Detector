import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("Data/sample.csv", index_col=0)
data = data.iloc[0:100]

doc, test_doc, y_train, y_test = train_test_split(data["text"], data["class"], test_size=0.25)

# create TfidfVectorizer object
tfidf = TfidfVectorizer()

# get tf-df values of text
doc_vec = tfidf.fit_transform(doc.values.astype("U"))
doc_matrix = doc_vec.toarray()
doc_transposed_matrix = [
    [doc_matrix[j][i] for j in range(len(doc_matrix))]
    for i in range(len(doc_matrix[0]))
]

test_vec = tfidf.transform(test_doc.values.astype("U"))
test_matrix = test_vec.toarray()
test_transposed_matrix = [
    [test_matrix[j][i] for j in range(len(test_matrix))]
    for i in range(len(test_matrix[0]))
]

testInput = input("Input your test data: ")
testInput = ["testInput"]
input_vec = tfidf.transform(testInput)
input_matrix = input_vec.toarray()
input_transposed_matrix = [
    [input_matrix[j][i] for j in range(len(input_matrix))]
    for i in range(len(input_matrix[0]))
]

X = np.array(doc_transposed_matrix)
X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

y = np.array(y_train)

Z = np.array(test_transposed_matrix)
Z = np.concatenate((np.ones((1, Z.shape[1])), Z), axis=0)

t = np.array(y_test)

L = np.array(input_transposed_matrix)
L = np.concatenate((np.ones((1, L.shape[1])), L), axis=0)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]

    count = 0
    check_w_after = 20

    while count < max_count:
        # mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1

            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)

    return w

eta = 0.5
d = X.shape[0]
w_init = np.random.randn(d, 1)
w = logistic_sigmoid_regression(X, y, w_init, eta)

result = sigmoid(np.dot(w[-1].T, Z))
print(sigmoid(np.dot(w[-1].T, L)))

index = 0
accurate = 0
for i in result[0]:
    j = t[index]
    index += 1
    if(i >= 0.5 and j == 1):
        accurate += 1
        continue
    if(i < 0.5 and j == 0):
        accurate += 1

print(accurate / t.shape[0] * 100, end="%")
