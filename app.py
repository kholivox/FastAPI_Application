from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI()

# Request schema
class Message(BaseModel):
    message: str

# Prediction endpoint
@app.post("/predict")
def predict(data: Message):
    text = data.message.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        vector = vectorizer.transform([text.lower()])
        intent = model.predict(vector)[0]

        return {"intent": intent}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))