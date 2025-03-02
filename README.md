# SMS Spam Detection Pipeline Documentation

## Table of Contents
1. [Data Loading](#data-loading)
2. [Data Cleaning](#data-cleaning)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Data Preprocessing](#data-preprocessing)
5. [Text Vectorization](#text-vectorization)
6. [Model Building and Evaluation](#model-building-and-evaluation)
7. [Saving Models](#saving-models)
8. [Conclusion](#conclusion)

---

## Data Loading

**Objective:** Load the dataset from a CSV file.

**Key Actions:**
- Import required libraries (`numpy` and `pandas`).
- Read the CSV file (`final.csv`) using encoding `latin1`.
- Display sample records and the DataFrame shape.

```python
import numpy as np
import pandas as pd

df = pd.read_csv('final.csv', encoding='latin1')
df.sample(5)
df.shape
```

---

## Data Cleaning

**Objective:** Prepare the raw data for analysis.

**Key Actions:**
- Inspect data using `df.info()`.
- Remove unwanted columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`).
- Rename columns (`v1` to `target`, `v2` to `text`).
- Check for missing values and duplicates.
- Remove duplicate records.

```python
df.info()

df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
df.sample(5)

df.isnull().sum()
df.duplicated().sum()

df = df.drop_duplicates(keep='first')
df.duplicated().sum()
df.shape
```

---

## Exploratory Data Analysis (EDA)

**Objective:** Gain insights into the dataset.

**Key Actions:**
- Analyze target variable distribution.
- Compute text statistics (number of characters, words, sentences).
- Visualize data using pie charts, histograms, pair plots, and heatmaps.

```python
df.head()
df['target'].value_counts()

import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()

import nltk
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
df[['num_characters', 'num_words', 'num_sentences']].describe()
```

---

## Data Preprocessing

**Text Transformation:** Normalize and clean text.

- Convert text to lowercase.
- Tokenize the text.
- Remove non-alphanumeric tokens.
- Remove stopwords and punctuation.
- Apply stemming using PorterStemmer.

```python
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:] 
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:] 
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)
df.head()
```

**WordClouds and Frequency Analysis:**
- Generate word clouds for spam and ham.
- Plot bar charts for the top 30 words.

```python
from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(ham_wc)
```

---

## Text Vectorization

**Objective:** Convert text to numerical features using TF-IDF.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
X.shape
y = df['target'].values
```

---

## Model Building and Evaluation

**Objective:** Train and evaluate multiple classifiers.

**Key Actions:**
- Split data into training and testing sets.
- Train Naive Bayes, SVC, Decision Tree, and Random Forest models.
- Evaluate using accuracy, confusion matrix, and precision.
- Aggregate performance metrics and visualize results.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

svc = SVC(kernel='sigmoid', gamma=1.0)
dtc = DecisionTreeClassifier(max_depth=5)
rfc = RandomForestClassifier(n_estimators=50, random_state=2)

clfs = {'SVC': svc, 'NB': mnb, 'DT': dtc, 'RF': rfc}

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    print("For", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)
```

---

## Saving Models

**Objective:** Persist the vectorizer and trained model for deployment.

```python
import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
```

---

## Conclusion

This document details an end-to-end pipeline for SMS spam detection. It covers data loading, cleaning, exploratory data analysis, data preprocessing, feature extraction, model building, evaluation, and model persistence. The pipeline is modular and can be extended for further improvements.
