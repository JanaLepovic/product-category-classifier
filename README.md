# Product Category Classifier

Machine learning projekat za automatsku predikciju kategorije proizvoda na osnovu njegovog naziva.

## Dataset

Za treniranje je korišćen `products.csv` skup sa više od 35.000 proizvoda.

Tokom pripreme podataka:
- uklonjene su nedostajuće vrednosti iz kolona `Product Title` i `Category Label`
- uklonjeni su nepotrebni razmaci iz naziva kolona
- standardizovane su kategorije:
  - `fridge` → `Fridges`
  - `CPU` → `CPUs`
  - `Mobile Phone` → `Mobile Phones`

Nakon čišćenja ostalo je 10 kategorija proizvoda.

## Feature engineering

Analizirane su dodatne karakteristike naziva proizvoda:
- dužina naslova
- broj reči
- prisustvo brojeva

Finalni model koristi tekst naslova proizvoda predstavljen pomoću TF-IDF vektorizacije.

## Modeli

Upoređena su dva algoritma:

- Logistic Regression
- Linear SVC

Rezultati na test skupu:

- Logistic Regression accuracy: oko 96.0%
- Linear SVC accuracy: oko 97.2%

Linear SVC je izabran kao finalni model jer je ostvario bolje ukupne rezultate.

## Ručno testiranje

Model je dodatno testiran na šest primera proizvoda iz zadatka.

Tačno je klasifikovao 4 od 6 proizvoda.

Greške su se pojavile kod dva proizvoda iz kategorije `Fridge Freezers`, koje je model klasifikovao kao `Dishwashers`.

Ovo pokazuje da model može imati poteškoće sa kratkim nazivima i oznakama modela koje ne sadrže dovoljno jasnih informacija o vrsti proizvoda.

## Struktura projekta

- `products.csv` – dataset
- `Machine_learning_final_task.ipynb` – analiza, čišćenje, feature engineering, treniranje i evaluacija
- `train_model.py` – treniranje i čuvanje finalnog modela
- `predict_category.py` – interaktivna predikcija kategorije
- `product_category_model.pkl` – sačuvani Linear SVC model i TF-IDF vektorizator
- `.gitignore` – fajlovi koji se ne prate kroz Git

## Pokretanje projekta

Za treniranje modela:

```bash
python train_model.py