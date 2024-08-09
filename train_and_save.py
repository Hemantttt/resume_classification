import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

# Load and preprocess data
df = pd.read_csv("Resume.csv")

# Text cleaning function
def clean(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    clean_text = url_pattern.sub('', text)
    clean_text = email_pattern.sub('', clean_text)
    clean_text = re.sub('[^\w\s]', '', clean_text)
    stop_words = set(stopwords.words('english'))
    clean_text  = ' '.join(word for word in clean_text.split() if word.lower() not in stop_words)
    return clean_text

df['Resume'] = df['Resume'].apply(lambda x: clean(x))

# Encode categories
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Vectorize text data
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['Resume'])
y = df['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Logistic Regression model on test data: {accuracy}")

# Save the TF-IDF vectorizer and model
with open('tfidf.pkl', 'wb') as file:
    pickle.dump(tfidf, file)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("TF-IDF vectorizer and model have been saved to 'tfidf.pkl' and 'model.pkl'.")
