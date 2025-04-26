# SpamShield Email Classifier

A machine learning system for classifying emails as spam or ham (non-spam) using multiple classification algorithms.

## Project Overview

SpamShield uses natural language processing and machine learning techniques to analyze email content and determine if it's spam. The system:

- Preprocesses text data using TF-IDF vectorization
- Trains multiple classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Evaluates and compares model performance
- Provides a simple interface for classifying new emails

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For macOS users, install libomp (required for XGBoost):

```bash
brew install libomp
```

## Usage

### Training

Train the models on the spam dataset:

```bash
python train.py
```

Output:
```
2025-04-25 23:36:02,582 - INFO - Loading data...
2025-04-25 23:36:02,750 - INFO - Training LogisticRegression...
2025-04-25 23:36:02,750 - INFO - Hyperparameter tuning for LogisticRegression
2025-04-25 23:36:12,001 - INFO - Best params for LogisticRegression: {'C': 10, 'solver': 'liblinear'}
2025-04-25 23:36:12,085 - INFO - Saved LogisticRegression model to models
2025-04-25 23:36:12,085 - INFO - Training DecisionTree...
2025-04-25 23:36:12,566 - INFO - Saved DecisionTree model to models
2025-04-25 23:36:12,566 - INFO - Training RandomForest...
2025-04-25 23:36:12,566 - INFO - Hyperparameter tuning for RandomForest
...
2025-04-25 23:36:53,435 - INFO - Training XGBoost...
2025-04-25 23:36:53,435 - INFO - Hyperparameter tuning for XGBoost
2025-04-25 23:37:03,421 - INFO - Best params for XGBoost: {'learning_rate': 0.1, 'max_depth': 5}
2025-04-25 23:37:04,737 - INFO - Saved XGBoost model to models
2025-04-25 23:37:04,737 - INFO - Training completed!
```

### Classification

Classify an email using the trained model:

```bash
python inference.py
```

Example outputs:

```
Enter email text: Free entry in 2 a wkly comp to win FA Cup final tkts
Classification: SPAM
```

```
Enter email text: Hi, can we meet tomorrow at 3pm to discuss the project?
Classification: HAM
```

## Model Performance

The system compares multiple models and selects the best performer. Here are the performance metrics for each model:

| Model               | Accuracy | F1 Score | AUC    |
|---------------------|----------|----------|--------|
| LogisticRegression  | 0.972    | 0.943    | 0.992  |
| DecisionTree        | 0.945    | 0.887    | 0.921  |
| RandomForest        | 0.979    | 0.957    | 0.996  |
| XGBoost             | 0.981    | 0.962    | 0.997  |

For detailed analysis, see the [Model Performance Report](reports/model_performance.md).

### Performance Visualizations

The training process generates visualization reports in the `reports/figures` directory:

- Model metrics comparison (`metrics_comparison.png`)
- ROC curves for all models (`roc_curves.png`)

For complete explanation of the visualizations, see the [Visualization Report](reports/visualization_report.md).

## Data Analysis

The SMS Spam Collection dataset used contains 5,574 SMS messages with approximately 86.6% legitimate messages (ham) and 13.4% spam messages.

Key insights:
- Spam messages are typically longer (avg. 138.7 characters) than ham messages (avg. 71.5 characters)
- Words like "free", "text", "call", "win", and "prize" are strong spam indicators

For complete dataset analysis, see the [Data Analysis Report](reports/data_analysis.md).

## Project Structure

```
spamshield-email-classifier/
├── data/
│   └── spam.csv           # Spam dataset
├── models/                # Saved model files
├── reports/
│   ├── figures/           # Performance visualizations
│   ├── model_performance.md  # Detailed performance metrics
│   ├── data_analysis.md   # Dataset analysis report
│   └── visualization_report.md # Explanation of visualizations
├── logs/                  # Training logs
├── config/                # Configuration files
├── train.py               # Model training script
├── inference.py           # Email classification script
└── requirements.txt       # Project dependencies
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- joblib
- tabulate 