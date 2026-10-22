# Text Emotion Classification

A Flask application that classifies text emotions using TF-IDF features and saved Naive Bayes and logistic regression models.

## Features

- Text preprocessing with NLTK
- Naive Bayes and logistic regression predictions
- Decoded emotion labels in the web interface

## Technology

Python and Flask.

## Local setup

Clone the repository:

```bash
git clone https://github.com/nasirabbas827/textemotion_Rafia_Hammad_final.git
cd textemotion_Rafia_Hammad_final
```

Use a virtual environment:

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
python -m pip install flask joblib numpy scikit-learn nltk
```

The install command lists packages used by the inspected source; this repository does not include a pinned dependency file. Saved machine learning models may require the versions used during training.

Complete the project-specific setup below, then run from the repository root:

```bash
python app.py
```

Open `http://127.0.0.1:5000/`.

## Project-specific setup

Keep `naive_bayes_model.pkl`, `logreg_model.pkl`, `tfidf_vectorizer.pkl` and `label_encoder.pkl` alongside `app.py`. Install the NLTK resources requested by the code and use scikit-learn versions compatible with these artifacts.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Flask application |
| `templates/` | HTML templates |
| `textemotion_Rafia_Hammad_final.ipynb` | Notebook for experiments or model preparation |

## Development notes

Setup steps describe local development. Configure your own database, mail and API credentials where applicable. Review debug settings and replace development secrets before deploying. The application was not executed as part of this documentation update.

Maintained by [Nasir Abbas](https://github.com/nasirabbas827).
