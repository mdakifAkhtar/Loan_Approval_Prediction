from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret-key"  # required for session

model = joblib.load("loan_approve.pkl")


@app.route("/", methods=["GET"])
def index():
    # Pull one-time result out of session (if any), then clear it
    prediction_text = session.pop("prediction_text", None)
    prediction_class = session.pop("prediction_class", None)
    errors = session.pop("errors", None)

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        prediction_class=prediction_class,
        errors=errors,
        form_data=None,  # always empty on GET -> fields stay blank
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {
        "age": request.form.get("age", ""),
        "income": request.form.get("income", ""),
        "credit_score": request.form.get("credit_score", ""),
        "loan_amount": request.form.get("loan_amount", ""),
        "years_employed": request.form.get("years_employed", ""),
    }

    errors = []

    def parse_float(key, label):
        try:
            return float(form_data[key])
        except (ValueError, TypeError):
            errors.append(f"{label} must be a valid number.")
            return None

    age = parse_float("age", "Age")
    income = parse_float("income", "Annual Income")
    credit_score = parse_float("credit_score", "Credit Score")
    loan_amount = parse_float("loan_amount", "Loan Amount")
    years_employed = parse_float("years_employed", "Years Employed")

    if not errors:
        if age < 18:
            errors.append("Age must be 18 or older.")
        if income < 0:
            errors.append("Annual Income must be 0 or greater.")
        if credit_score < 300 or credit_score > 900:
            errors.append("Credit Score must be between 300 and 900.")
        if loan_amount <= 0:
            errors.append("Loan Amount must be greater than 0.")
        if years_employed < 0:
            errors.append("Years Employed must be 0 or greater.")

    if errors:
        session["errors"] = errors
        return redirect(url_for("index"))

    features = np.array([[income, credit_score, loan_amount, years_employed]])
    prediction = model.predict(features)
    result = int(prediction[0])

    if result == 1:
        session["prediction_text"] = "Loan Approved"
        session["prediction_class"] = "approved"
    else:
        session["prediction_text"] = "Loan Not Approved"
        session["prediction_class"] = "rejected"

    return redirect(url_for("index"))


import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)