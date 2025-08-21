import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

good_feedbacks = [
    "Absolutely loved it!", "Great quality and fast delivery.", "Highly recommend this.",
    "Product exceeded my expectations.", "Will definitely buy again.", "Five stars!",
    "Very satisfied with the purchase.", "Amazing experience!", "Excellent service.",
    "Good value for money.",
    "Absolutely loved it!", "Great quality and fast delivery.", "Highly recommend this.",
    "Product exceeded my expectations.", "Will definitely buy again.", "Five stars!",
    "Very satisfied with the purchase.", "Amazing experience!", "Excellent service.",
    "Good value for money.",
  "Absolutely loved it!", "Great quality and fast delivery.", "Highly recommend this.",
    "Product exceeded my expectations.", "Will definitely buy again.", "Five stars!",
    "Very satisfied with the purchase.", "Amazing experience!", "Excellent service.",
    "Good value for money.",
  "Absolutely loved it!", "Great quality and fast delivery.", "Highly recommend this.",
    "Product exceeded my expectations.", "Will definitely buy again.", "Five stars!",
    "Very satisfied with the purchase.", "Amazing experience!", "Excellent service.",
    "Good value for money.",
  "Absolutely loved it!", "Great quality and fast delivery.", "Highly recommend this.",
    "Product exceeded my expectations.", "Will definitely buy again.", "Five stars!",
    "Very satisfied with the purchase.", "Amazing experience!", "Excellent service.",
    "Good value for money."
] 

bad_feedbacks = [
    "Terrible product.", "It broke within days.", "Very poor quality.",
    "Waste of money.", "I want a refund.", "Extremely disappointed.",
    "Not as described.", "It arrived damaged.", "Would not recommend.",
    "Customer service was awful.",
    "Terrible product.", "It broke within days.", "Very poor quality.",
    "Waste of money.", "I want a refund.", "Extremely disappointed.",
    "Not as described.", "It arrived damaged.", "Would not recommend.",
    "Customer service was awful.",
    "Terrible product.", "It broke within days.", "Very poor quality.",
    "Waste of money.", "I want a refund.", "Extremely disappointed.",
    "Not as described.", "It arrived damaged.", "Would not recommend.",
    "Customer service was awful.",
    "Terrible product.", "It broke within days.", "Very poor quality.",
    "Waste of money.", "I want a refund.", "Extremely disappointed.",
    "Not as described.", "It arrived damaged.", "Would not recommend.",
    "Customer service was awful.",
    "Terrible product.", "It broke within days.", "Very poor quality.",
    "Waste of money.", "I want a refund.", "Extremely disappointed.",
    "Not as described.", "It arrived damaged.", "Would not recommend.",
    "Customer service was awful."
  ]  

texts = good_feedbacks[:50] + bad_feedbacks[:50]
labels = ['good'] * 50 + ['bad'] * 50

df = pd.DataFrame({'Feedback': texts, 'Label': labels})

vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')
X = vectorizer.fit_transform(df['Feedback'])

X_train, X_test, y_train, y_test = train_test_split(X, df['Label'], test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)

example_feedbacks = ["This movie was one of the wrost of all time.", "Money was utilised very usefully."]
vectorized = text_preprocess_vectorize(example_feedbacks, vectorizer)
predictions = model.predict(vectorized)

for text, pred in zip(example_feedbacks, predictions):
    print(f"Feedback: \"{text}\" → Predicted Sentiment: {pred}")
