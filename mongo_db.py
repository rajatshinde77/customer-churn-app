from pymongo import MongoClient

MONGO_URI = "mongodb+srv://rajat:Rajat%40141099@cluster0.o14fppz.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI)

db = client["churn_db"]
predictions_collection = db["predictions"]