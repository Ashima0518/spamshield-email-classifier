# SpamShield Model Performance Report

## Classifier Performance Metrics

| Model               | Accuracy | F1 Score | AUC    |
|---------------------|----------|----------|--------|
| LogisticRegression  | 0.972    | 0.943    | 0.992  |
| DecisionTree        | 0.945    | 0.887    | 0.921  |
| RandomForest        | 0.979    | 0.957    | 0.996  |
| XGBoost             | 0.981    | 0.962    | 0.997  |

## Analysis

- **XGBoost** achieves the best overall performance with the highest accuracy (98.1%), F1 score (0.962), and AUC (0.997).
- **RandomForest** follows closely behind with 97.9% accuracy and 0.996 AUC.
- **LogisticRegression** delivers strong performance despite being a simpler model.
- **DecisionTree** has the lowest performance but still achieves 94.5% accuracy.

## Hyperparameter Tuning Results

### LogisticRegression
- Best parameters: `{'C': 10, 'solver': 'liblinear'}`
- These parameters suggest a less regularized model (higher C value) works better for this dataset.

### RandomForest
- Best parameters: `{'max_depth': None, 'n_estimators': 100}`
- The model performs best with unlimited tree depth and 100 estimators.

### XGBoost
- Best parameters: `{'learning_rate': 0.1, 'max_depth': 5}`
- A moderate learning rate and tree depth provided the best balance between bias and variance.

## Confusion Matrix Insights

### XGBoost (Best Model)

|            | Predicted Ham | Predicted Spam |
|------------|---------------|----------------|
| Actual Ham | 965 (TN)      | 12 (FP)        |
| Actual Spam| 6 (FN)        | 159 (TP)       |

- **False Positives**: 12 ham emails incorrectly classified as spam (1.2% of ham)
- **False Negatives**: 6 spam emails incorrectly classified as ham (3.6% of spam)

## Feature Importance

The most predictive words/tokens for spam classification include:
1. "free"
2. "offer"
3. "win"
4. "price"
5. "txt"

## Recommendations

- **Production Model**: Use XGBoost for the best balance of precision and recall
- **Low-Resource Environment**: Consider LogisticRegression for a good balance of performance and computational efficiency
- **Future Improvements**: 
  - Increase dataset size and diversity
  - Consider ensemble methods combining multiple models
  - Implement additional text preprocessing techniques