from pymongo import MongoClient

try:
    # ✅ FINAL CORRECT CONNECTION STRING
    MONGO_URI = "mongodb+srv://rajat:<Rajat@141099>@cluster0.0i4fppz.mongodb.net/?appName=Cluster0"

    client = MongoClient(MONGO_URI)

    db = client["churn_db"]
    predictions_collection = db["predictions"]

    print("MongoDB Connected Successfully ✅")

except Exception as e:
    print("MongoDB Connection Error ❌:", e)
    predictions_collection = None