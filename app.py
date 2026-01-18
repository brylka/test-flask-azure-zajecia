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

@app.route('/form', methods=['GET'])
def form():
    """Prosty formularz HTML do predykcji używający endpointa /predict"""
    return '''
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="utf-8">
        <title>Klasyfikator irysów</title>
    </head>
    <body>
        <h1>Klasyfikator irysów</h1>
        <input id="sl" type="number" value="5.1" step="0.1" placehilder="sepal_length"><br>
        <input id="sw" type="number" value="3.5" step="0.1" placehilder="sepal_width"><br>
        <input id="pl" type="number" value="1.4" step="0.1" placehilder="petal_length"><br>
        <input id="pw" type="number" value="0.2" step="0.1" placehilder="petal_width"><br>
        <button onclick="predict()">Predykcja</button>
        <h2 id="result"></h2>
        <script>
            async function predict() {
                const data = {
                    sepal_length: parseFloat(document.getElementById('sl').value),
                    sepal_width: parseFloat(document.getElementById('sw').value),
                    petal_length: parseFloat(document.getElementById('pl').value),
                    petal_width: parseFloat(document.getElementById('pw').value)
                }
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                })
                const result = await res.json()
                document.getElementById('result').innerText = 'Gatunetk: ' + result.species + ' (Prawdopodobieństwo: ' + (result.probability*100) + '%)'
            }
        </script>
    </body>
    </html>
    '''



if __name__ == '__main__':
    app.run()
