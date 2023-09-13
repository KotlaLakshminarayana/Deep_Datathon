from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

pipe_lr = joblib.load(open("pickle_data.pkl", "rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0], np.max(pipe_lr.predict_proba([docx]))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        raw_text = request.form.get("raw_text")
        prediction, confidence = predict_emotions(raw_text)
        return render_template("result.html", raw_text=raw_text, prediction=prediction, confidence=confidence)
    return render_template("home.html")

@app.route("/monitor")
def monitor():
    
    page_visits = [
        {"Pagename": "Home", "Time_of_Visit": "2023-09-05 12:00:00"},
        {"Pagename": "Home", "Time_of_Visit": "2023-09-05 12:15:00"},
        {"Pagename": "About", "Time_of_Visit": "2023-09-05 13:30:00"},
    ]
    
    return render_template("monitor.html", page_visits=page_visits)

if __name__ == "__main__":
    app.run(debug=True)
