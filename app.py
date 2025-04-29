# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pymongo import MongoClient
import datetime

# Load the trained model
model = joblib.load('models/learning_style_model.pkl')

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["learning_style_db"]
collection = db["user_predictions"]

# Create FastAPI instance
app = FastAPI()

# Define input data format
class UserActivity(BaseModel):
    avg_reading_time: float
    avg_video_time: float
    quiz_accuracy: float
    clicks: int

# Define the prediction endpoint
@app.post("/predict")
def predict_learning_style(activity: UserActivity):
    # Prepare input data
    data = [[activity.avg_reading_time, activity.avg_video_time, activity.quiz_accuracy, activity.clicks]]
    
    # Get prediction
    prediction = model.predict(data)
    predicted_style = prediction[0]
    
    # Save data and prediction in MongoDB
    prediction_data = {
        "avg_reading_time": activity.avg_reading_time,
        "avg_video_time": activity.avg_video_time,
        "quiz_accuracy": activity.quiz_accuracy,
        "clicks": activity.clicks,
        "predicted_style": predicted_style,
        "timestamp": datetime.datetime.now()
    }
    
    collection.insert_one(prediction_data)
    
    # Return prediction
    return {"predicted_style": predicted_style}
