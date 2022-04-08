import pandas as pd
import numpy as np
import os
from glob import glob

train = pd.read_csv("../_data/dacon/news/train.csv")
test = pd.read_csv("../_data/dacon/news/test.csv")
submission = pd.read_csv("../_data/dacon/news/sample_submission.csv")

#print(train.head())
#print(test.head())
#print(submission.head())

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

vectorizer.fit(np.array(train["text"]))

train_vec = vectorizer.transform(train["text"])
train_y = train["target"]

test_vec = vectorizer.transform(test["text"])

from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model.fit(train_vec, train_y)

pred = model.predict(test_vec)
submission["target"] = pred

print(submission)

submission.to_csv("final_submission.csv", index = False)