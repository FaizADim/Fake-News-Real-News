# Fake News Real News

A Jupyter notebook project for training and evaluating text classifiers that label news articles as **Fake** or **Real**. The project includes the training data and serialized models.

> **Important:** These models are text classifiers, not fact-checkers. Their predictions reflect patterns in the training data and should not be treated as evidence that an article is true or false.

## Repository contents

| Path | Description |
| --- | --- |
| `Model.ipynb` | Loads and preprocesses the data, trains and evaluates the models, and tests them on sample articles. |
| `News/Fake.csv` | Articles labeled Fake. |
| `News/True.csv` | Articles labeled Real. |
| `News/vectorizer.pkl` | Fitted TF-IDF text vectorizer used by the saved classifiers. |
| `News/lr.pkl` | Logistic Regression classifier. |
| `News/dt.pkl` | Decision Tree classifier. |
| `News/gb.pkl` | Gradient Boosting classifier. |
| `News/rf.pkl` | Random Forest classifier. |

The CSV files have `title`, `text`, `subject`, and `date` columns. Training uses the article `text` column and assigns labels based on which CSV the row came from.

## Requirements

- Python 3
- Jupyter Notebook
- pandas
- scikit-learn
- joblib

Install the Python dependencies in your environment:

```bash
python -m pip install pandas scikit-learn joblib notebook
```

## Run the notebook

From the repository root, start Jupyter:

```bash
jupyter notebook
```

Open `Model.ipynb` and run its cells. The notebook reads the two CSV files, preprocesses the article text, creates a stratified train/test split, fits a TF-IDF vectorizer, trains four classifiers, and saves the resulting models and vectorizer in `News/`.

**Before running:** The notebook currently contains absolute file paths from its original machine. Update `FAKE_PATH`, `TRUE_PATH`, and `OUT_DIR` in the training cell to point to this repository's `News/` directory. The first cell also contains a machine-specific directory path; update it or skip that file-listing cell.

Training can take a while, especially because the notebook runs five-fold cross-validation for each classifier.

## Use a saved model

Use the same text-cleaning steps and vectorizer used during training when making predictions. For example, from the repository root:

```python
import joblib
import re
import string

def wordopt(text):
    text = text.lower()
    text = re.sub(r'^[a-z\s,\.]+\([^)]+\)\s*[-–—]\s*', '', text)
    text = re.sub(r'\bby [a-z]+ [a-z]+\b', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\b(?=[a-z]+\d|\d+[a-z])[a-z0-9]+\b', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

vectorizer = joblib.load("News/vectorizer.pkl")
model = joblib.load("News/lr.pkl")  # lr, dt, gb, or rf

article = "Paste the article text here."
features = vectorizer.transform([wordopt(article)])
prediction = model.predict(features)[0]
print("Real" if prediction == 1 else "Fake")
```

Only load pickle files you trust. Pickled scikit-learn objects may also require compatible library versions to load.
