import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("Data/sample.csv", index_col=0)

x = data["text"]
y = data["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train.values.astype('U'))
xv_test = vectorization.transform(x_test.values.astype('U'))

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr=LR.predict(xv_test)

LR.score(xv_test, y_test)

print(classification_report(y_test, pred_lr))