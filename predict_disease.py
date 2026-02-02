
import pickle, re, numpy as np, csv
from roman_urdu_map import normalize_roman_urdu
from symptom_anchor import SYMPTOM_ANCHORS

model = pickle.load(open("final_disease_model.pkl","rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl","rb"))
VOCAB = set(vectorizer.get_feature_names_out())

def clean_text(text):
    text = normalize_roman_urdu(text.lower())
    for w in ["mein","ka","ki","hai","ha"]:
        text = text.replace(w,"")
    text = text.replace("dard","pain")
    text = re.sub(r"[^a-z\s]"," ",text)
    return re.sub(r"\s+"," ",text).strip()

def vocab_overlap(text):
    words = set(text.split())
    overlap = words & VOCAB
    return len(overlap), overlap

def anchor_boost(cleaned_text, probs):
    for anchor, diseases in SYMPTOM_ANCHORS.items():
        if all(w in cleaned_text for w in anchor.split()):
            for d in diseases:
                if d in model.classes_:
                    idx = list(model.classes_).index(d)
                    probs[idx] *= 1.8
    return probs

def predict_disease(user_input):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    overlap_count, overlap_words = vocab_overlap(cleaned)

    if overlap_count == 0 and vec.nnz == 0:
        return {"status":"invalid","message":"No medical intent detected"}

    probs = model.predict_proba(vec)[0]
    probs = anchor_boost(cleaned, probs)
    probs = probs / probs.sum()

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    confidence_percent = round(confidence*100,2)
    assurity = "High" if confidence_percent>=70 else "Medium" if confidence_percent>=40 else "Low"

    return {
        "disease": model.classes_[idx],
        "confidence_percent": confidence_percent,
        "assurity_level": assurity,
        "recognized_terms": list(overlap_words),
        "note": "Anchored probabilistic prediction"
    }

if __name__ == "__main__":
    print(predict_disease(input("Enter symptoms: ")))
