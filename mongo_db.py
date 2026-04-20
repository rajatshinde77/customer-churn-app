from pymongo import MongoClient
from urllib.parse import quote_plus

try:
    username = "rajat"
    password = quote_plus("Rajat@141099")   # ✅ auto-fix password

    MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.o14fppz.mongodb.net/?retryWrites=true&w=majority"

    client = MongoClient(MONGO_URI)

    db = client["churn_db"]
    predictions_collection = db["predictions"]

    print("MongoDB Connected Successfully ✅")

except Exception as e:
    print("MongoDB Connection Error ❌:", e)
    predictions_collection = None