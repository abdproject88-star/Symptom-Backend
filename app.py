from fastapi import FastAPI
from pydantic import BaseModel
import pickle, re, numpy as np
from roman_urdu_map import normalize_roman_urdu
from symptom_anchor import SYMPTOM_ANCHORS

app = FastAPI()

# Load model & vectorizer
model = pickle.load(open("final_disease_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
VOCAB = set(vectorizer.get_feature_names_out())

class InputData(BaseModel):
    symptoms: str

def clean_text(text: str) -> str:
    text = normalize_roman_urdu(text.lower())
    for w in ["mein","ka","ki","hai","ha"]:
        text = text.replace(w,"")
    text = text.replace("dard","pain")
    text = re.sub(r"[^a-z\s]"," ", text)
    return re.sub(r"\s+"," ", text).strip()

def vocab_overlap(text: str):
    words = set(text.split())
    overlap = words & VOCAB
    return len(overlap), overlap

def anchor_boost(cleaned_text: str, probs):
    for anchor, diseases in SYMPTOM_ANCHORS.items():
        if all(w in cleaned_text for w in anchor.split()):
            for d in diseases:
                if d in model.classes_:
                    idx = list(model.classes_).index(d)
                    probs[idx] *= 1.8
    return probs

@app.get("/")
def root():
    return {"status": "Disease Prediction API running"}

@app.post("/predict")
def predict(data: InputData):
    cleaned = clean_text(data.symptoms)
    overlap_count, overlap_words = vocab_overlap(cleaned)

    # ✅ INVALID input check
    if overlap_count == 0:
        return {"status":"invalid", "message":"No medical intent detected"}

    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    probs = anchor_boost(cleaned, probs)
    probs = probs / probs.sum()

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    confidence_percent = round(confidence * 100, 2)
    assurity = "High" if confidence_percent >= 70 else "Medium" if confidence_percent >= 40 else "Low"

    return {
        "status":"success",
        "disease": model.classes_[idx],
        "confidence_percent": confidence_percent,
        "assurity_level": assurity,
        "recognized_terms": list(overlap_words),
        "note": "Anchored probabilistic prediction"
    }
