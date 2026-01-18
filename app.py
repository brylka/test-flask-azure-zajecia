from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Tworzymy aplikację Flask
app = Flask(__name__)

# Trenujemy model przy starcie aplikacji
print("Trenuję model...")
iris = load_iris()
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(iris.data, iris.target)
print("Model gotowy!")

# Nazwy gatunków
SPECIES = ['setosa', 'versicolor', 'virginica']


@app.route('/health', methods=['GET'])
def health():
    """Endpoint do sprawdzenia czy serwis działa"""
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint do predykcji gatunku irysa

    Oczekuje JSON:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    # Pobierz dane z requestu
    data = request.get_json()

    # Przygotuj features dla modelu
    features = np.array([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]])

    # Predykcja
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()

    # Zwróć wynik
    return jsonify({
        "species": SPECIES[prediction],
        "probability": round(float(probability), 3)
    })


@app.route('/', methods=['GET'])
def home():
    """Strona główna z instrukcją"""
    return jsonify({
        "message": "Iris Classifier API",
        "endpoints": {
            "GET /health": "Sprawdź status",
            "POST /predict": "Wyślij dane irysa, otrzymaj predykcję"
        },
        "example_input": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    })


if __name__ == '__main__':
    app.run()
