from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("stock_model.pkl")
poly = joblib.load("poly.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    year = int(request.form['year'])

    data = poly.transform([[year]])

    prediction = model.predict(data)

    return render_template(
        'result.html',
        year=year,
        prediction=round(prediction[0], 2)
    )

if __name__ == '__main__':
    app.run(debug=True)