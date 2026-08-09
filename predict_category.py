import pickle


# Učitavanje sačuvanog modela i TF-IDF vektorizatora
with open("product_category_model.pkl", "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
vectorizer = saved_data["vectorizer"]


# Interaktivno testiranje novih proizvoda
while True:

    product_title = input(
        "\nUnesite naziv proizvoda ili unesite 'exit' za izlaz: "
    )

    if product_title.lower() == "exit":
        print("Program je završen.")
        break

    # Pretvaranje naziva proizvoda u isti numerički oblik
    product_vector = vectorizer.transform([product_title.lower()])

    # Predikcija kategorije
    prediction = model.predict(product_vector)[0]

    print("Predviđena kategorija:", prediction)