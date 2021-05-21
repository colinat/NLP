#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:51:36 2021

@author: colin
"""

#%% import 20newsgroups data

from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train')
# newsgroups_test = fetch_20newsgroups(subset='test')

#%% extract sample dataset

#data = newsgroups_train.data
categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

#%% STEP 1: CLEANING UP TEXT

import re
from cleantext import clean

data = twenty_train.data

def cleanup_text(data):

    email_headers = re.compile(r'^from:.*(?:\r?\n(?!\r?\n).*)*', re.IGNORECASE)
    clean_data = list(map(lambda x: email_headers.sub('', x).strip(), data[:]))  # remove email headers
    clean_data = list(map(lambda x: re.sub(r"\r?\n\r?\n.*(?:\r?\n(?!\r?\n).*)*$",'', x).strip(), clean_data[:])) # remove ending signatures
    #clean_data = list(map(lambda x: re.sub(r"[a-zA-Z0-9_.\$\-]*@(\w*\.)*\w*",'',x).strip(), clean_data[:])) # remove emails

    ## using clean-text library
# =============================================================================
#     # usage:
#     clean("some input",
#         fix_unicode=True,               # fix various unicode errors
#         to_ascii=True,                  # transliterate to closest ASCII representation
#         lower=True,                     # lowercase text
#         no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
#         no_urls=False,                  # replace all URLs with a special token
#         no_emails=False,                # replace all email addresses with a special token
#         no_phone_numbers=False,         # replace all phone numbers with a special token
#         no_numbers=False,               # replace all numbers with a special token
#         no_digits=False,                # replace all digits with a special token
#         no_currency_symbols=False,      # replace all currency symbols with a special token
#         no_punct=False,                 # remove punctuations
#         replace_with_punct="",          # instead of removing punctuations you may replace them
#         replace_with_url="<URL>",
#         replace_with_email="<EMAIL>",
#         replace_with_phone_number="<PHONE>",
#         replace_with_number="<NUMBER>",
#         replace_with_digit="0",
#         replace_with_currency_symbol="<CUR>",
#         lang="en"                       # set to 'de' for German special handling
#     )
# =============================================================================

    clean_data = list(map(lambda x: clean(x, no_urls=True, no_emails=True, no_digits=True, no_currency_symbols=True, no_punct=True).strip(), clean_data[:]))
    return clean_data

    #clean_data = list(map(lambda x: re.sub(r"\s+",' ',x).strip(), clean_data[:])) # remove newlines, extra whitespaces
    #clean_data = list(map(lambda x: re.sub(r'[<>\(\)\{\}\\\*]','',x).strip(), clean_data[:])) # remove non-conventional punctuations
    #clean_data = list(map(lambda x: re.sub(r'\s+(?=[,.:;])','',x).strip(), clean_data[:])) # remove any whitespaces before commas, period, etc.


#%% STEP 2: TOKENIZING TEXT
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(cleanup_text(data))
X_train_counts.shape

# get index of specific word
# count_vect.vocabulary_.get(u'algorithm')

#%% STEP 3: GENERATE TFIDF MATRIX
from sklearn.feature_extraction.text import TfidfTransformer

## Convert absolute occurences to term frequencies (i.e. normalize)
# X_train_tf = TfidfTransformer(use_idf=False).fit_transform(X_train_counts)
# X_train_tf.shape

## Convert absolute occurences to tf-idf representation (cutting down on common words)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#%% STEP 4: TRAIN CLASSIFIER (NAIVES BAYES)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf,twenty_train.target)

#%% STEP 5: TEST ON SAMEPLE SENTENCE
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print("\'{}\' => {}".format(doc, twenty_train.target_names[category]))
    
#%% STEP 6: BUILDING PIPELINE
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(data, twenty_train.target)
predicted = text_clf.predict(data)
for doc, category in zip(docs_new, predicted):
    print("\'{}\' => {}".format(doc, twenty_train.target_names[category]))

#%% STEP 7: EVALUATION ON TESTSET
import numpy as np

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
#docs_test = cleanup_text(twenty_test.data)  # cleaning up lowers model accuracy
docs_test = twenty_test.data

predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

#%% STEP 8: USING SVM CLASSIFIER INSTEAD
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(docs_test)
print(np.mean(predicted == twenty_test.target))

#%% STEP 9: EVALUATION VIA CONFUSION MATRIX
from sklearn import metrics

print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))

print(metrics.confusion_matrix(twenty_test.target, predicted))

#%% STEP 10: HYPERTUNING PARAMETERS
from sklearn.model_selection import GridSearchCV
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

print(gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

#%% STEP 11: EXPORTING POSSIBLE PARAMETERS COMBINATION
import pandas as pd

df_params = pd.DataFrame.from_dict(gs_clf.cv_results_)
    
