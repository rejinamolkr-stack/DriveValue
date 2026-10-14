
from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("DriveValue_CarPrice_Model.pkl")

FEATURES = [
    "enginesize", "horsepower", "curbweight", "boreratio",
    "fuelsystem", "drivewheel", "wheelbase", "carlength",
    "carwidth", "manufacturer", "citympg", "highwaympg"
]

FUEL_SYSTEMS = [
    "1bbl", "2bbl", "4bbl", "idi", "mfi", "mpfi", "spdi", "spfi"
]

DRIVE_WHEELS = ["4wd", "fwd", "rwd"]

MANUFACTURERS = [
    "alfa-romero", "audi", "bmw", "buick", "chevrolet",
    "dodge", "honda", "isuzu", "jaguar", "maxda", "mazda",
    "mercury", "mitsubishi", "nissan", "peugeot", "plymouth",
    "porcshce", "porsche", "renault", "saab", "subaru",
    "toyota", "toyouta", "vokswagen", "volkswagen", "vw", "volvo"
]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DriveValue | Car Price Prediction</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    padding: 24px 12px;
    background: #F5EEE5;
    color: #463226;
    font-family: Arial, sans-serif;
}

.container {
    max-width: 760px;
    margin: 12px auto;
    background: #FFFCF8;
    border: 1px solid #E5D6C6;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 25px rgba(83, 54, 34, 0.10);
}

.top {
    padding: 22px 24px 18px;
    background: #70452D;
    color: #FFFFFF;
}

.brand {
    font-size: 12px;
    letter-spacing: 2px;
    color: #E8D2BC;
    font-weight: bold;
}

h1 {
    font-size: 25px;
    margin: 10px 0 6px;
}

.subtitle {
    font-size: 13px;
    color: #F0E2D4;
    line-height: 1.5;
    margin: 0;
}

.content {
    padding: 22px 24px;
}

.section-title {
    font-size: 12px;
    color: #8A6247;
    letter-spacing: 1.3px;
    font-weight: bold;
    margin: 0 0 16px;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 15px 12px;
}

label {
    display: block;
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 7px;
    color: #5B4030;
}

input, select {
    width: 100%;
    min-width: 0;
    padding: 10px 9px;
    border: 1px solid #DFD0C0;
    border-radius: 7px;
    background: #FAF6F1;
    color: #463226;
    font-size: 13px;
    outline: none;
}

input:focus, select:focus {
    border-color: #9B7252;
    box-shadow: 0 0 0 2px #EDE0D3;
}

.button-row {
    display: flex;
    gap: 10px;
    margin-top: 22px;
}

button, .reset {
    flex: 1;
    text-align: center;
    padding: 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: bold;
    cursor: pointer;
    text-decoration: none;
}

button {
    border: none;
    background: #70452D;
    color: #FFFFFF;
}

button:hover {
    background: #57351F;
}

.reset {
    border: 1px solid #DCCABA;
    color: #70452D;
    background: #FFFFFF;
}

.result {
    margin-top: 20px;
    padding: 17px;
    border: 1px solid #D9C4AD;
    border-radius: 10px;
    text-align: center;
    background: #F2E5D6;
}

.result-label {
    color: #79563C;
    font-size: 11px;
    letter-spacing: 1.5px;
    font-weight: bold;
}

.price {
    font-size: 28px;
    font-weight: bold;
    color: #593820;
    margin-top: 7px;
}

.error {
    margin-top: 16px;
    padding: 12px;
    border-radius: 8px;
    background: #F9E6E1;
    color: #8B3427;
    font-size: 13px;
}

.footer {
    text-align: center;
    padding: 12px;
    font-size: 11px;
    color: #927B68;
    border-top: 1px solid #EEE2D6;
}

@media (max-width: 540px) {
    body {
        padding: 10px;
    }

    .top, .content {
        padding: 18px;
    }

    .form-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 13px 10px;
    }

    h1 {
        font-size: 22px;
    }
}
</style>
</head>

<body>
<div class="container">

    <header class="top">
        <div class="brand">DRIVEVALUE</div>
        <h1>Car Price Predictor</h1>
        <p class="subtitle">
            Estimate a car's price using its specifications and machine learning.
        </p>
    </header>

    <main class="content">
        <p class="section-title">ENTER VEHICLE DETAILS</p>

        <form action="/predict" method="POST">
            <div class="form-grid">

                <div>
                    <label for="enginesize">Engine Size</label>
                    <input id="enginesize" type="number" name="enginesize"
                           min="1" step="any" placeholder="e.g. 130"
                           value="{{ values.get('enginesize', '') }}" required>
                </div>

                <div>
                    <label for="horsepower">Horsepower</label>
                    <input id="horsepower" type="number" name="horsepower"
                           min="1" step="any" placeholder="e.g. 110"
                           value="{{ values.get('horsepower', '') }}" required>
                </div>

                <div>
                    <label for="curbweight">Curb Weight</label>
                    <input id="curbweight" type="number" name="curbweight"
                           min="1" step="any" placeholder="e.g. 2500"
                           value="{{ values.get('curbweight', '') }}" required>
                </div>

                <div>
                    <label for="boreratio">Bore Ratio</label>
                    <input id="boreratio" type="number" name="boreratio"
                           min="0" step="any" placeholder="e.g. 3.5"
                           value="{{ values.get('boreratio', '') }}" required>
                </div>

                <div>
                    <label for="fuelsystem">Fuel System</label>
                    <select id="fuelsystem" name="fuelsystem" required>
                        {% for item in fuel_systems %}
                        <option value="{{ item }}"
                        {% if values.get('fuelsystem', 'mpfi') == item %}selected{% endif %}>
                            {{ item }}
                        </option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label for="drivewheel">Drive Wheel</label>
                    <select id="drivewheel" name="drivewheel" required>
                        {% for item in drive_wheels %}
                        <option value="{{ item }}"
                        {% if values.get('drivewheel', 'fwd') == item %}selected{% endif %}>
                            {{ item|upper }}
                        </option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label for="wheelbase">Wheelbase</label>
                    <input id="wheelbase" type="number" name="wheelbase"
                           min="1" step="any" placeholder="e.g. 95"
                           value="{{ values.get('wheelbase', '') }}" required>
                </div>

                <div>
                    <label for="carlength">Car Length</label>
                    <input id="carlength" type="number" name="carlength"
                           min="1" step="any" placeholder="e.g. 175"
                           value="{{ values.get('carlength', '') }}" required>
                </div>

                <div>
                    <label for="carwidth">Car Width</label>
                    <input id="carwidth" type="number" name="carwidth"
                           min="1" step="any" placeholder="e.g. 66"
                           value="{{ values.get('carwidth', '') }}" required>
                </div>

                <div>
                    <label for="manufacturer">Manufacturer</label>
                    <select id="manufacturer" name="manufacturer" required>
                        {% for item in manufacturers %}
                        <option value="{{ item }}"
                        {% if values.get('manufacturer', 'toyota') == item %}selected{% endif %}>
                            {{ item|replace('-', ' ')|title }}
                        </option>
                        {% endfor %}
                    </select>
                </div>

                <div>
                    <label for="citympg">City MPG</label>
                    <input id="citympg" type="number" name="citympg"
                           min="1" step="any" placeholder="e.g. 27"
                           value="{{ values.get('citympg', '') }}" required>
                </div>

                <div>
                    <label for="highwaympg">Highway MPG</label>
                    <input id="highwaympg" type="number" name="highwaympg"
                           min="1" step="any" placeholder="e.g. 32"
                           value="{{ values.get('highwaympg', '') }}" required>
                </div>

            </div>

            <div class="button-row">
                <button type="submit">Predict Car Price &rarr;</button>
                <a class="reset" href="/">Reset</a>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result">
            <div class="result-label">ESTIMATED CAR PRICE</div>
            <div class="price">{{ "{:,.2f}".format(prediction) }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
    </main>

    <footer class="footer">
        DriveValue &bull; Car Price Prediction Using Machine Learning
    </footer>

</div>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(
        HTML,
        values={},
        prediction=None,
        error=None,
        fuel_systems=FUEL_SYSTEMS,
        drive_wheels=DRIVE_WHEELS,
        manufacturers=MANUFACTURERS
    )


@app.route("/predict", methods=["POST"])
def predict():
    values = request.form.to_dict()

    try:
        numeric_features = [
            "enginesize", "horsepower", "curbweight", "boreratio",
            "wheelbase", "carlength", "carwidth", "citympg", "highwaympg"
        ]

        for feature in numeric_features:
            if float(values[feature]) <= 0:
                raise ValueError("Please enter values greater than zero.")

        input_data = {
            feature: (
                float(values[feature])
                if feature in numeric_features
                else values[feature]
            )
            for feature in FEATURES
        }

        input_df = pd.DataFrame([input_data], columns=FEATURES)
        prediction = float(model.predict(input_df)[0])

        return render_template_string(
            HTML,
            values=values,
            prediction=prediction,
            error=None,
            fuel_systems=FUEL_SYSTEMS,
            drive_wheels=DRIVE_WHEELS,
            manufacturers=MANUFACTURERS
        )

    except Exception as e:
        return render_template_string(
            HTML,
            values=values,
            prediction=None,
            error="Prediction failed: " + str(e),
            fuel_systems=FUEL_SYSTEMS,
            drive_wheels=DRIVE_WHEELS,
            manufacturers=MANUFACTURERS
        )


if __name__ == "__main__":
    app.run(debug=True)