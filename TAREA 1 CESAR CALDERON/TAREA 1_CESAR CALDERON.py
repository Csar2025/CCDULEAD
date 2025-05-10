import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load dataset (using a sample dataset; replace with your own if needed)
# For this example, we'll create a small sample dataset
data = {
    'review': [
        'This product is amazing and works perfectly!',
        'Terrible experience, very disappointed',
        'I love this item, great quality',
        'Poor service and bad product',
        'Fantastic purchase, highly recommend',
        'Not worth the money, broke quickly'
    ],
    'sentiment': [1, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative
}
df = pd.DataFrame(data)

# Preprocess reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Split data into training and testing sets
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Function to predict sentiment for new reviews
def predict_sentiment(review):
    cleaned_review = preprocess_text(review)
    review_tfidf = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_tfidf)[0]
    return 'Positive' if prediction == 1 else 'Negative'

# Example usage
new_review = 'This is a great product!'
print(f'\nNew review: "{new_review}"')
print(f'Predicted sentiment: {predict_sentiment(new_review)}')