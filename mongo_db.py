from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

db = client["churn_prediction_db"]

predictions_collection = db["predictions"]