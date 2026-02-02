from fastapi import FastAPI
import pickle
import re

app = FastAPI()

# Load model & vectorizer
model = pickle.load(open("final_disease_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@app.get("/")
def root():
    return {"status": "Disease Prediction API running"}

@app.post("/predict")
def predict_disease(data: dict):
    text = clean_text(data["symptoms"])
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    return {
        "disease": prediction
    }
