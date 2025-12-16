# Stock News Sentiment Classification (BiLSTM)

## Overview

This project focuses on financial news sentiment classification using traditional NLP baselines and deep learning models. The goal is to evaluate whether neural sequence models can provide meaningful gains over classical approaches on a small, domain-specific dataset, where data scarcity and domain-specific vocabulary often challenge deep models.

## Dataset

Given short financial news sentences, classify the sentiment into:

- **Negative**
- **Neutral**
- **Positive**

This is a **multi-class text classification** problem with class imbalance.

- **Source**: Financial PhraseBank (Kaggle)
- **Samples**: ~5,800 financial news sentences
- **Labels**: negative / neutral / positive

The raw dataset is cleaned and normalized before modeling.

## Project structure
```text
stock-news-sentiment-tensorflow/
├── notebooks/
├── src/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── preprocessing/
│   ├── models/
│   │   ├── best/
│   │   └── tuning/
│   └── results/
├── README.md
├── requirements.txt
└── .gitignore
```

## Methodology

The project follows a structured machine learning workflow:

1. **Exploratory Data Analysis (EDA)**  
   Analysis of label distribution, text length, and class imbalance, followed by a TF-IDF + Logistic Regression baseline.

2. **Deep Learning Baseline**  
   A BiLSTM model with trainable embeddings is trained using class-weighted loss to handle imbalanced sentiment classes.

3. **Hyperparameter Tuning**  
   Key model and training hyperparameters are tuned to maximize validation Macro F1-score.

4. **Final Inference**  
   The best-performing model is reconstructed and evaluated once on a held-out test set to obtain the final metrics.

## Evaluation Protocol

To ensure fair comparison, all models were evaluated using a fixed train/validation/test split with stratification to preserve class distribution across splits.

The test set was held out and never used during model selection or hyperparameter tuning.

## Final Result & Model Comparison

The performance of different models was evaluated using **Macro F1-score** to account for class imbalance in the dataset.

| Model                        | Macro F1 (Test) |
|:-----------------------------|:----------------|
| TF-IDF + Logistic Regression | 0.624           |
| BiLSTM (Baseline)            | 0.642           |
| BiLSTM (Tuned, Final)        | 0.637           |

The tuned BiLSTM model achieved a modest yet consistent improvement over the TF-IDF baseline (Macro F1-score: 0.624 → 0.637), indicating limited but measurable gains from sequence modeling on this dataset.

The final, authoritative evaluation metrics are stored in:
- `artifacts/results/final_metrics.json`

Overall, the experiment demonstrates that when dataset size and domain context are limited, classical methods can remain as strong baselines for financial text classification.

## Interpretation

Hyperparameter tuning improved validation performance, but test performance plateaued.
This suggests that the model reached the generalization limit of the dataset rather than suffering from underfitting.

Such behaviour is common in small NLP datasets, where classical baselines often capture much of the available signal.

## Requirements
- Python >= 3.10

## Installation

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## How to Run

Execute notebooks in the following order:

1. `01_eda.ipynb`
2. `02_bilstm_baseline_model.ipynb`
3. `03_bilstm_hyperparameter_tuning.ipynb`
4. `04_final_inference.ipynb`

## Reproducibility Notes

- Fixed random seeds are used where applicable.
- Train/validation/test splits are reproduced using consistent parameters.
- All intermediate artifacts are persisted for traceability.

## Takeaways

- Strong classical TF-IDF baselines remain highly competitive, with neural models providing only modest gains on small financial datasets.
- Validation improvements from hyperparameter tuning do not always translate to test performance.
- Model selection should be guided by dataset size and complexity, not model sophistication alone.
- Artifact management is critical for reproducible ML workflows.
- Test Macro F1-score stabilized around 0.637 cross modeling approaches.

## Future Improvements

- Incorporate pretrained embeddings (e.g. GloVe).
- Explore transformer-based models.
- Apply data augmentation for class imbalance, which could enhance model robustness.

## Author

Terence Leung
