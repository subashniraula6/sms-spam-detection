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

# Deploying the Trained Machine Learning Model Using Streamlit
We design simple web application built with Streamlit that classifies input messages as "Spam" or "Not Spam" using a pre-trained machine learning model and a TF-IDF vectorizer.

## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Preprocessing Function](#preprocessing-function)
4. [Model Loading](#model-loading)
5. [Streamlit Application](#streamlit-application)
6. [Prediction Workflow](#prediction-workflow)
7. [Running the Application](#running-the-application)
8. [Conclusion](#conclusion)

---

## Overview

The app allows users to input a message (email or SMS), preprocesses the text using a custom transformation function, vectorizes the text with TF-IDF, and then predicts whether the message is spam using a pre-trained model.

---

## Dependencies

- **Streamlit:** For building the web interface.
- **pickle:** To load the serialized TF-IDF vectorizer and ML model.
- **NLTK:** For text processing (tokenization, stop words removal, and stemming).
- **string:** For handling string operations.
- **Other Libraries:** Ensure all necessary packages are installed via pip.

---

## Preprocessing Function

The `transform_text` function processes the input text by:
- Converting text to lower case.
- Tokenizing the text.
- Removing non-alphanumeric tokens.
- Removing stop words and punctuation.
- Stemming tokens using the PorterStemmer.
- Returning the cleaned text.

```python
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
```

---

## Model Loading

The pre-trained TF-IDF vectorizer and spam classifier model are loaded using pickle. Ensure that the files `vectorizer.pkl` and `model.pkl` are present in the same directory as the application.

```python
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
```

---

## Streamlit Application

The app uses Streamlit to create a simple web interface:
- `st.title()` sets the title of the app.
- `st.text_area()` provides an input area for the message.
- A button labeled 'Predict' triggers the classification process.

```python
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")
```

---

## Prediction Workflow

When the 'Predict' button is clicked, the following steps occur:
1. Preprocessing: The input message is cleaned using the `transform_text` function.
2. Vectorization: The cleaned text is transformed into numerical features with the loaded TF-IDF vectorizer.
3. Prediction: The pre-trained model predicts whether the message is spam.
4. Display: The result ("Spam" or "Not Spam") is shown to the user.

```python
if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    elif result == 0:
        st.header("Not Spam")
    else:
        st.header("")
```

---

## Running the Application

To run the application:
1. Ensure that `vectorizer.pkl` and `model.pkl` are in the same directory.
2. Install the required libraries if not already installed:
    pip install streamlit nltk
3. Run the app using:
    streamlit run your_app_file.py

---

## Conclusion

This document details an end-to-end pipeline for SMS spam detection. It covers data loading, cleaning, exploratory data analysis, data preprocessing, feature extraction, model building, evaluation, and model persistence. The pipeline is modular and can be extended for further improvements.

And for the Application part, the documentation provides an overview of a Streamlit-based Email/SMS Spam Classifier. The app features a modular design with separate functions for text preprocessing, model loading, and prediction workflow. It offers a user-friendly interface for quickly classifying messages as spam or not spam.
