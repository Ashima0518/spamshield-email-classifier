import joblib
import numpy as np

class SpamClassifier:
    def __init__(self):
        self.vectorizer = joblib.load('models/vectorizer.pkl')
        self.models = {
            'LogisticRegression': joblib.load('models/LogisticRegression_model.pkl'),
            'XGBoost': joblib.load('models/XGBoost_model.pkl')
        }
    
    def predict(self, text, model_name='XGBoost'):
        text = self._preprocess(text)
        vectorized = self.vectorizer.transform([text])
        model = self.models[model_name]
        return model.predict(vectorized)[0]
    
    def _preprocess(self, text):
        return text.lower().strip()

if __name__ == "__main__":
    classifier = SpamClassifier()
    email = input("Enter email text: ")
    prediction = classifier.predict(email)
    print(f"Classification: {'SPAM' if prediction else 'HAM'}")