from flask import Flask, request, render_template
import joblib
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


# Load models and vectorizer
nb_model = joblib.load("naive_bayes_model.pkl")
lr_model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

app = Flask(__name__)

# Text preprocessing function
def preprocess(text):
    text_clean = re.sub(r"http\S+|www\S+|https\S+", '', text.lower())
    text_clean = re.sub(r'\@\w+|\#','', text_clean)
    text_clean = re.sub(r'[^a-z\s]', '', text_clean)
    tokens = [word for word in text_clean.split() if word not in stop_words]
    return " ".join(tokens)

def predict_model(text, model):
    vec = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        pred_idx = np.argmax(probs)
        return le.classes_[pred_idx], probs[pred_idx]
    else:
        pred_idx = model.predict(vec)[0]
        return le.classes_[pred_idx], None

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        comment = request.form["comment"]
        comment_clean = preprocess(comment)
        
        nb_pred, nb_conf = predict_model(comment_clean, nb_model)
        lr_pred, lr_conf = predict_model(comment_clean, lr_model)
        
        result = {
            "Naive Bayes": {"Emotion": nb_pred, "Confidence": f"{nb_conf:.2f}" if nb_conf else "N/A"},
            "Logistic Regression": {"Emotion": lr_pred, "Confidence": f"{lr_conf:.2f}" if lr_conf else "N/A"}
        }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
