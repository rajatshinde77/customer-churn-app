from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask import Flask, render_template, request, send_file, redirect
from mongo_db import predictions_collection
from flask import jsonify

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet

from bson.objectid import ObjectId
from datetime import datetime

import os
import pickle
import numpy as np

CHURN = "Customer Will Churn"
NOT_CHURN = "Customer Will Not Churn"

# -------------------------
# Initialize Flask App
# -------------------------
app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY", "default_secret")


# Prevent browser caching (fix back button login issue)
@app.after_request
def prevent_cache(response):
    response.headers["Cache-Control"] = "no-store"
    return response


# -------------------------
# Login Manager Setup
# -------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, id):
        self.id = id

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME") or "admin"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or "admin123"

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)


# -------------------------
# Load ML Model
# -------------------------
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Store last prediction for PDF
last_prediction = {}

# -------------------------
# Landing Page
# -------------------------
@app.route("/", methods=["GET"])
def landing():
    return render_template("landing.html")


# -------------------------
# Login Page
# -------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:

            user = User(username)
            login_user(user)

            return redirect("/dashboard")

        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")


# -------------------------
# Logout
# -------------------------
@app.route("/logout", methods=["GET"])
@login_required
def logout():
    logout_user()
    return redirect("/login")


# -------------------------
# Home Page
# -------------------------
@app.route("/home", methods=["GET"])
@login_required
def home_page():
    return render_template("index.html")


# -------------------------
# Prediction History
# -------------------------
@app.route("/history", methods=["GET"])
@login_required
def history():
    if predictions_collection is not None:
        data = list(predictions_collection.find())
    else:
        data = []
    return render_template("history.html", data=data)


# -------------------------
# Dashboard
# -------------------------
@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard():

    if predictions_collection is not None:
        total_predictions = predictions_collection.count_documents({})
        churn_count = predictions_collection.count_documents({"prediction": CHURN})
        not_churn_count = predictions_collection.count_documents({"prediction": NOT_CHURN})
    else:
        total_predictions = 0
        churn_count = 0
        not_churn_count = 0

    return render_template(
        "dashboard.html",
        total=total_predictions,
        churn=churn_count,
        not_churn=not_churn_count
    )

# -------------------------
# analytics
# -------------------------
@app.route("/analytics", methods=["GET"])
@login_required
def analytics():

    if predictions_collection is not None:
        data = list(predictions_collection.find())
    else:
        data = []

    total = len(data)

    if total == 0:
        avg_monthly = 0
    else:
        avg_monthly = sum(float(d.get("monthly_charges", 0)) for d in data) / total

    churn_count = sum(1 for d in data if d.get("prediction") == CHURN)
    not_churn_count = total - churn_count

    # ✅ FIX ADDED
    if total != 0:
        churn_rate = round((churn_count / total) * 100, 2)
    else:
        churn_rate = 0

    return render_template(
        "analytics.html",
        avg_monthly=round(avg_monthly,2),
        churn=churn_count,
        not_churn=not_churn_count,
        churn_rate=churn_rate   # ✅ IMPORTANT
    )
# -------------------------
# Model Comparison Page
# -------------------------
@app.route("/model_comparison", methods=["GET"])
@login_required
def model_comparison():

    models = [
        {"name": "Logistic Regression", "accuracy": 81.62},
        {"name": "Random Forest", "accuracy": 80.27}
    ]

    # Sort models (best first)
    models = sorted(models, key=lambda x: x["accuracy"], reverse=True)

    best_model = models[0]["name"]

    return render_template(
        "model_comparison.html",
        models=models,
        best_model=best_model
    )


# -------------------------
# Prediction Route
# -------------------------
def generate_reasons(pred, tenure, monthly, contract, tech, security, dependents, senior):

    reasons = []

    # -------- CHURN CASE --------
    if pred == 1:
        conditions = [
            (tenure < 6, "Short tenure"),
            (monthly > 80, "High charges"),
            (contract == 0, "Month-to-month contract"),
            (tech == 0, "No technical support"),
            (security == 0, "No online security"),
            (dependents == 0, "No dependents"),
            (senior == 1, "Senior citizen")
        ]

    # -------- NON-CHURN CASE --------
    else:
        conditions = [
            (tenure > 24, "Long tenure"),
            (contract == 2, "Long-term contract"),
            (tech == 1, "Has technical support"),
            (security == 1, "Secure services"),
            (monthly < 70, "Stable charges")
        ]

    # -------- APPLY CONDITIONS --------
    for condition, message in conditions:
        if condition:
            reasons.append(message)

    return reasons

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json() if request.is_json else request.form

    try:

        # -------------------------
        # Input Features
        # -------------------------
        features = [
            float(data["gender"]),
            float(data["SeniorCitizen"]),
            float(data["Partner"]),
            float(data["Dependents"]),
            float(data["tenure"]),

            float(data["contract"]),
            float(data["internet_service"]),
            float(data["payment_method"]),
            float(data["tech_support"]),
            float(data["streaming_tv"]),
            float(data["online_security"]),

            0,0,0,0,0,0,

            float(data["monthly_charges"]),
            float(data["total_charges"])
        ]

        final_features = np.array([features])

        # -------------------------
        # Prediction
        # -------------------------
        prediction = model.predict(final_features)

        # -------------------------
        # Extract values
        # -------------------------
        tenure = float(data["tenure"])
        monthly_charges = float(data["monthly_charges"])
        contract = int(data["contract"])
        tech_support = int(data["tech_support"])
        online_security = int(data["online_security"])
        dependents = int(data["Dependents"])
        senior = int(data["SeniorCitizen"])

        # -------------------------
        # Generate Reasons (FIXED)
        # -------------------------
        reasons = generate_reasons(
            prediction[0],
            tenure,
            monthly_charges,
            contract,
            tech_support,
            online_security,
            dependents,
            senior
        )

        # -------------------------
        # Probability & Risk
        # -------------------------
        probability = model.predict_proba(final_features)[0][1]
        risk = round(probability * 100, 2)

        if risk >= 70:
            risk_level = "HIGH"
        elif risk >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # -------------------------
        # Final Result
        # -------------------------
        if prediction[0] == 1:
            result = CHURN
        else:
            result = NOT_CHURN

        # -------------------------
        # Save to DB
        # -------------------------
        if predictions_collection is not None:
            predictions_collection.insert_one({
                "tenure": data["tenure"],
                "monthly_charges": data["monthly_charges"],
                "total_charges": data["total_charges"],
                "prediction": result
        })

        # -------------------------
        # Store for PDF
        # -------------------------
        global last_prediction
        last_prediction = {
                "prediction": result,
                "probability": risk,
                "risk_level": risk_level,
                "tenure": data["tenure"],
                "monthly": data["monthly_charges"],
                "total": data["total_charges"],
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        # -------------------------
        # Return Output
        # -------------------------
        if request.is_json:
            return jsonify({
                "prediction": result,
                "risk": risk,
                "risk_level": risk_level,
                "reasons": reasons
            })
        
        return render_template(
            "result.html",
            prediction_text=result,
            risk=risk,
            risk_level=risk_level,
            reasons=reasons
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"

# -------------------------
# Download PDF Report
# -------------------------
@app.route("/download_report", methods=["GET"])
@login_required
def download_report():

    styles = getSampleStyleSheet()
    file_path = "prediction_report.pdf"

    story = []

    # LOGO
    logo_path = "static/logo_report.png"
    if os.path.exists(logo_path):
        logo = Image(logo_path, width=2*inch, height=1*inch)
        story.append(logo)

    story.append(Spacer(1, 10))

    # TITLE
    story.append(Paragraph("Customer Churn Prediction Report", styles['Title']))
    story.append(Spacer(1,20))

    # REPORT INFO
    story.append(Paragraph(f"<b>Report ID:</b> LAST-PREDICTION", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {last_prediction['date']}", styles['Normal']))

    story.append(Spacer(1,10))

    # DATA
    story.append(Paragraph(f"<b>Prediction Result:</b> {last_prediction['prediction']}", styles['Normal']))
    story.append(Paragraph(f"<b>Churn Probability:</b> {last_prediction['probability']}%", styles['Normal']))
    story.append(Paragraph(f"<b>Risk Level:</b> {last_prediction['risk_level']}", styles['Normal']))
    story.append(Paragraph(f"<b>Tenure:</b> {last_prediction['tenure']}", styles['Normal']))
    story.append(Paragraph(f"<b>Monthly Charges:</b> {last_prediction['monthly']}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Charges:</b> {last_prediction['total']}", styles['Normal']))

    story.append(Spacer(1, 20))

    story.append(Paragraph("Generated by Customer Churn Prediction System", styles['Italic']))

    pdf = SimpleDocTemplate(file_path)
    pdf.build(story)

    return send_file(file_path, as_attachment=True)

@app.route('/download/<id>', methods=["GET"])
@login_required
def download_from_history(id):

    if predictions_collection is None:
        return "Database not available"

    record = predictions_collection.find_one({"_id": ObjectId(id)})

    if not record:
        return "Record not found"

    file_path = "report_{}.pdf".format(id)

    styles = getSampleStyleSheet()
    story = []

    # LOGO
    logo_path = "static/logo_report.png"
    if os.path.exists(logo_path):
        logo = Image(logo_path, width=2*inch, height=1*inch)
        story.append(logo)

    story.append(Spacer(1, 10))

    # TITLE
    story.append(Paragraph("Customer Churn Prediction Report", styles['Title']))
    story.append(Spacer(1, 10))

    # REPORT INFO
    now = datetime.now()
    story.append(Paragraph(f"<b>Report ID:</b> {id}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {now.strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Paragraph(f"<b>Time:</b> {now.strftime('%H:%M:%S')}", styles['Normal']))

    story.append(Spacer(1, 15))

    # DATA
    story.append(Paragraph(f"<b>Tenure:</b> {record['tenure']}", styles['Normal']))
    story.append(Paragraph(f"<b>Monthly Charges:</b> {record['monthly_charges']}", styles['Normal']))
    story.append(Paragraph(f"<b>Total Charges:</b> {record['total_charges']}", styles['Normal']))
    story.append(Paragraph(f"<b>Prediction:</b> {record['prediction']}", styles['Normal']))

    story.append(Spacer(1, 20))

    story.append(Paragraph("Generated by Customer Churn Prediction System", styles['Italic']))

    pdf = SimpleDocTemplate(file_path)
    pdf.build(story)

    return send_file(file_path, as_attachment=True)

# -------------------------
# Run Flask App
# -------------------------
if __name__ == "__main__":
    app.run()