from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Witaj Merito! 2026.01.18"


if __name__ == "__main__":
    app.run()