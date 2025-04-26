# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_curve, roc_auc_score, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

# Configuration
class Config:
    DATA_PATH = Path('data/spam.csv')
    MODEL_SAVE_PATH = Path('models/')
    FIGURE_SAVE_PATH = Path('reports/figures/')
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CLASS_NAMES = ['ham', 'spam']
    HYPERPARAMETERS = {
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10]
        },
        'XGBoost': {
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    }

# Data preprocessing
class DataPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def preprocess(self, df):
        df = self._clean_data(df)
        X = self.vectorizer.fit_transform(df['text']).toarray()
        y = df['label'].values
        return X, y
    
    def _clean_data(self, df):
        df = df.copy()
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df['text'] = df['text'].str.lower()
        df['text'] = df['text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        return df

# Model Trainer
class ModelTrainer:
    def __init__(self):
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(max_depth=5),
            'RandomForest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.best_models = {}
        self.results = {}
        
    def train(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            
            # Hyperparameter tuning
            if name in Config.HYPERPARAMETERS:
                model = self._hyperparameter_tuning(model, name, X_train, y_train)
            
            # Training
            model.fit(X_train, y_train)
            self.best_models[name] = model
            
            # Evaluation
            self.results[name] = self._evaluate_model(model, X_test, y_test)
            self._save_model(model, name)
            
        self._plot_comparison()
        return self.results
    
    def _hyperparameter_tuning(self, model, name, X, y):
        logging.info(f"Hyperparameter tuning for {name}")
        grid = GridSearchCV(
            model,
            Config.HYPERPARAMETERS[name],
            cv=3,
            scoring='f1',
            n_jobs=-1
        )
        grid.fit(X, y)
        logging.info(f"Best params for {name}: {grid.best_params_}")
        return grid.best_estimator_
    
    def _evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(
                y_test, y_pred, target_names=Config.CLASS_NAMES, output_dict=True
            ),
            'fpr': roc_curve(y_test, y_proba)[0],
            'tpr': roc_curve(y_test, y_proba)[1]
        }
    
    def _save_model(self, model, name):
        joblib.dump(model, Config.MODEL_SAVE_PATH / f'{name}_model.pkl')
        logging.info(f"Saved {name} model to {Config.MODEL_SAVE_PATH}")
    
    def _plot_comparison(self):
        # Metrics comparison
        metrics = pd.DataFrame.from_dict({
            name: {'Accuracy': res['accuracy'], 'F1 Score': res['f1'], 'AUC': res['auc']}
            for name, res in self.results.items()
        }, orient='index')
        
        plt.figure(figsize=(10, 6))
        metrics.plot(kind='bar', rot=0)
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(Config.FIGURE_SAVE_PATH / 'metrics_comparison.png')
        plt.close()
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        for name, res in self.results.items():
            plt.plot(res['fpr'], res['tpr'], 
                     label=f'{name} (AUC = {res["auc"]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.savefig(Config.FIGURE_SAVE_PATH / 'roc_curves.png')
        plt.close()

# Main execution
if __name__ == "__main__":
    # Create directories
    Config.MODEL_SAVE_PATH.mkdir(exist_ok=True)
    Config.FIGURE_SAVE_PATH.mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # Load data
    logging.info("Loading data...")
    df = pd.read_csv(Config.DATA_PATH, encoding='latin-1')
    df.columns = ['label', 'text'] if df.shape[1] == 2 else df.columns
    
    # Rename columns if needed based on dataset format
    if 'type' in df.columns and 'text' in df.columns:
        df.rename(columns={'type': 'label'}, inplace=True)
    
    # Keep only the relevant columns
    if df.shape[1] > 2:
        df = df[['label', 'text']]
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df)
    joblib.dump(preprocessor.vectorizer, Config.MODEL_SAVE_PATH / 'vectorizer.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, stratify=y
    )
    
    # Train and evaluate
    trainer = ModelTrainer()
    results = trainer.train(X_train, X_test, y_train, y_test)
    
    # Save final report
    report = pd.DataFrame({
        name: [res['accuracy'], res['f1'], res['auc']]
        for name, res in results.items()
    }, index=['Accuracy', 'F1 Score', 'AUC']).T
    
    report.to_markdown('reports/model_performance.md')
    logging.info("Training completed!")