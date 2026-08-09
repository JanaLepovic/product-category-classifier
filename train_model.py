import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Učitavanje podataka
df = pd.read_csv("products.csv")

# Uklanjanje nepotrebnih razmaka iz naziva kolona
df.columns = df.columns.str.strip()

# Uklanjanje redova bez naziva proizvoda ili kategorije
df = df.dropna(subset=["Product Title", "Category Label"])

# Standardizacija teksta
df["Product Title"] = df["Product Title"].str.lower().str.strip()
df["Category Label"] = df["Category Label"].str.strip()

# Spajanje različitih naziva istih kategorija
df["Category Label"] = df["Category Label"].replace({
    "fridge": "Fridges",
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones"
})

# Ulazni podaci i ciljne kategorije
X = df["Product Title"]
y = df["Category Label"]

# Pretvaranje teksta u numerički oblik
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Treniranje finalnog modela
model = LinearSVC()
model.fit(X_tfidf, y)

# Čuvanje modela i vektorizatora
with open("product_category_model.pkl", "wb") as file:
    pickle.dump({
        "model": model,
        "vectorizer": vectorizer
    }, file)

print("Model je uspešno treniran i sačuvan.")