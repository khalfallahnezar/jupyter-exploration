# 🤖 NewsBot Intelligence System — ITAI 2373 Mid-Term

An end-to-end Natural Language Processing system that ingests raw news articles and returns structured
intelligence: **category**, **named entities**, **sentiment**, and interpretable **linguistic patterns**.
The project integrates every technique from Modules 1–8 of ITAI 2373 into a single working pipeline.

**Author:** Nezar Khalfallah 
**Course:** ITAI 2373 — Natural Language Processing
**Deliverable:** Mid-Term Project — NewsBot Intelligence System

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `NewsBot_Intelligence_System.ipynb` | Complete, runnable notebook with all analyses and saved outputs |
| `NewsBot_Reflection_Nezar_Khalfallah.pdf` | 2-page reflective journal |
| `README.md` | This file |

## ▶️ How to Run

1. Open `NewsBot_Intelligence_System.ipynb` in **Google Colab** (free tier is sufficient).
2. Run the **Setup and Installation** cell first — it installs libraries and downloads the spaCy model
   and NLTK data.
3. `Runtime → Run all`. The notebook downloads the dataset automatically, so no manual upload is needed.
   (If the download is ever unavailable, the loader falls back to a built-in sample so the notebook still runs.)

Total runtime is a few minutes on the Colab free tier.

## 📊 Dataset

**BBC News** — 2,225 full-text articles across five categories (*business, entertainment, politics, sport,
tech*). The notebook stratify-samples to 1,200 articles to stay within Colab limits. The dataset satisfies all
project requirements: ≥ 500 articles, ≥ 4 categories, substantial full-text content, clean English labels, and
no missing values.

## 🧩 System Architecture (Modules 1–8)

1. **Application context** — media-monitoring / business-intelligence use case and target users.
2. **Preprocessing** — clean → tokenize → stop-word removal → lemmatization → length filtering.
3. **TF-IDF** — 3,000-feature unigram+bigram vectorizer; per-category term analysis, word clouds, heatmap.
4. **POS analysis** — tag-proportion profiling that reveals writing-style differences across desks.
5. **Syntax & semantics** — spaCy dependency parsing for subjects/objects/noun-phrases + parse visualization.
6. **Sentiment** — VADER compound scoring and cross-category tone comparison.
7. **Classification** — Naive Bayes vs. Logistic Regression vs. SVM on a combined TF-IDF + sentiment + length
   feature matrix (MinMax-scaled), compared by accuracy, macro-F1, and cross-validation.
8. **NER** — spaCy entity extraction (PERSON, ORG, GPE, DATE, MONEY) with frequency and by-category analysis.

All components are wrapped in a single `NewsBotIntelligenceSystem` class and demonstrated live on brand-new
articles that are not part of the training data.

## 📈 Key Results

- **Classification:** Logistic Regression ≈ **95% test accuracy** (macro-F1 ≈ 0.95); the five categories are
  linguistically well-separated.
- **Writing style:** business/politics articles are proper-noun and number heavy; sport is verb heavy.
- **Sentiment:** entertainment skews most positive, politics most negative.
- **Entities:** tens of thousands of mentions extracted; top organizations surface per beat.

## 🚀 Possible Extensions

Transformer (BERT) embeddings for noisier corpora, a Streamlit/Gradio dashboard, entity-level sentiment and
co-occurrence graphs, and temporal trend tracking once publication dates are available.

## 📝 Notes

- **File-naming conventions (per assignment):**
  Report → `MT_Report_NA_Nezar_Khalfallah_ITAI2373`,
- AI tools were used as a coding aid; system design, analysis, and reflection are my own work.
- Dataset: D. Greene and P. Cunningham, *BBC News* dataset (2006).
