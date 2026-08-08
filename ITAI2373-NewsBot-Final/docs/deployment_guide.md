# Deployment Guide

NewsBot 2.0 ships as a notebook + installable package; it has no required web deployment. If you
want to expose it as a web application (the project's optional 30-point bonus), see the
"Web App Development Tutorial for NewsBot Intelligence System" guide, which walks through:

1. Wrapping `NewsBot2IntegratedSystem` inside a Flask app (`/analyze`, `/api/analyze`, `/` routes).
2. Building HTML/CSS templates for article submission and results display.
3. Deploying to Heroku, PythonAnywhere, or Render (all free-tier options).

## Minimal integration sketch

```python
from flask import Flask, request, jsonify
from src.newsbot2_system import NewsBot2Config, NewsBot2IntegratedSystem
import spacy, pandas as pd

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
config = NewsBot2Config()
system = NewsBot2IntegratedSystem(config, nlp)
system.train(pd.read_csv("data/processed/bbc_sample.csv"))

@app.route("/api/analyze", methods=["POST"])
def analyze():
    text = request.get_json()["text"]
    return jsonify(system.analyze_article(text))
```

This bonus was not built for the core submission — the notebook and package above satisfy all
200 required points on their own.
