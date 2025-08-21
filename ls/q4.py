import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

positive_reviews = [
    "Absolutely loved the movie!", "Great plot and stunning visuals.",
    "The acting was fantastic!", "What a wonderful experience!",
    "Truly a masterpiece.", "I enjoyed every second of it.",
    "Such an emotional story.", "Highly recommended!",
    "The direction was brilliant.", "An inspiring and uplifting film.",
    "Perfect balance of humor and drama.", "Loved the soundtrack.",
    "It was engaging from start to end.", "Top-notch performance!",
    "Brilliantly written and well-acted.", "A joy to watch.",
    "Would watch it again!", "Full of heart and soul.",
    "A beautifully made film.", "Impressive storytelling.",
    "An absolute gem.", "Feel-good movie of the year!",
    "Everything about it was amazing.", "Visually spectacular.",
    "A thrilling ride!", "The cast did an excellent job.",
    "Beautiful cinematography.", "I couldn’t stop smiling.",
    "Heartwarming and touching.", "Flawless execution.",
    "Clever and entertaining.", "Amazing performances.",
    "Great direction and screenplay.", "A very enjoyable movie.",
    "Highly entertaining.", "Worth every minute.",
    "One of the best films this year.", "So well crafted.",
    "Strong performances all around.", "A true work of art.",
    "It exceeded my expectations.", "Stellar work!",
    "A must-watch for everyone.", "This film blew me away.",
    "Incredible from start to finish.", "Will stay with me forever.",
    "Magical and memorable.", "Uplifting and powerful.",
    "Easily one of my favorites.", "I was moved to tears."
]

negative_reviews = [
    "Terrible movie.", "I hated it.", "Worst film I’ve ever seen.",
    "A complete waste of time.", "Painfully boring.", "So predictable.",
    "Poor acting and bad script.", "I walked out halfway.",
    "Don’t bother watching this.", "Utterly disappointing.",
    "Very slow and dull.", "Story made no sense.", "Cringe-worthy dialogue.",
    "The plot was a mess.", "Poorly executed.", "A total disaster.",
    "I regret watching it.", "It was just bad.", "Nothing good about it.",
    "So poorly made.", "Bad from start to finish.", "Predictable and flat.",
    "Couldn’t relate to any character.", "It lacked emotion.",
    "Unwatchable.", "I want my time back.", "It was torture.",
    "Everything felt forced.", "Such a letdown.", "The cast was wasted.",
    "No redeeming qualities.", "Direction was awful.",
    "Horrible pacing.", "Overacted and underwritten.",
    "I was bored to death.", "The jokes fell flat.",
    "Too long and pointless.", "Worst screenplay ever.",
    "Completely forgettable.", "Such a cliché.",
    "Absolutely terrible.", "Left the theater early.",
    "The hype was not worth it.", "Made no impact.",
    "Felt like a school play.", "Really amateurish.",
    "It didn’t make sense.", "The ending was stupid.",
    "A lifeless, soulless film.", "It failed on every level."
]

reviews = positive_reviews + negative_reviews
labels = ['positive'] * 50 + ['negative'] * 50

df = pd.DataFrame({'Review': reviews, 'Sentiment': labels})

from  sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=500)
X = vectorizer.fit_transform(df['Review'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, df['Sentiment'], test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", round(acc * 100, 2), "%")

def predict_review_sentiment(model, vectorizer, review):
    vect_review = vectorizer.transform([review])
    return model.predict(vect_review)[0]

example = "The movie was heartwarming and absolutely beautiful!"
print("Sentiment", predict_review_sentiment(model, vectorizer, example))
