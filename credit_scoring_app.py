from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. Create dummy dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=6, n_informative=4, random_state=42)
df = pd.DataFrame(X, columns=['income', 'debt', 'credit_score', 'loan_amount', 'late_payments', 'savings'])
df['default'] = y

# 2. Preprocessing
X = df.drop('default', axis=1)
y = df['default']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Create Flask app
app = Flask(__name__)

# 5. HTML Template
html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Credit Scoring Predictor</title>
    <style>
        body { font-family: Arial; background: #f4f4f4; padding: 20px; }
        .container { background: white; padding: 20px; max-width: 500px; margin: auto; border-radius: 8px; box-shadow: 0 0 10px #ccc; }
        input[type=number] { width: 100%; padding: 10px; margin: 8px 0; }
        input[type=submit] { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; margin-top: 10px; cursor: pointer; width: 100%; }
        h2 { text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Credit Scoring Predictor</h2>
        <form action="/predict" method="post">
            <label>Income:</label><input type="number" name="income" step="any" required><br>
            <label>Debt:</label><input type="number" name="debt" step="any" required><br>
            <label>Credit Score:</label><input type="number" name="credit_score" step="any" required><br>
            <label>Loan Amount:</label><input type="number" name="loan_amount" step="any" required><br>
            <label>Late Payments:</label><input type="number" name="late_payments" step="any" required><br>
            <label>Savings:</label><input type="number" name="savings" step="any" required><br>
            <input type="submit" value="Check Creditworthiness">
        </form>
        {% if result %}
            <h3 style="text-align:center;">Prediction: {{ result }}</h3>
        {% endif %}
    </div>
</body>
</html>
'''

# 6. Routes
@app.route('/', methods=['GET'])
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        input_features = [float(request.form.get(f)) for f in ['income', 'debt', 'credit_score', 'loan_amount', 'late_payments', 'savings']]
        
        # Preprocess
        input_scaled = scaler.transform([input_features])

        # Predict
        prediction = model.predict(input_scaled)[0]
        result = "High Risk (Default)" if prediction == 1 else "Low Risk (Good Credit)"

        return render_template_string(html_template, result=result)

    except Exception as e:
        return f"Error: {str(e)}"

# 7. Run app
if __name__ == '__main__':
    app.run(debug=True)
