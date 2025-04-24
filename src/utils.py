# src/utils.py

import joblib
import dill
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class Float32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.astype('float32')



def load_model():
    with open("models/lightgbm_best_model.pkl", "rb") as f:
       # return  joblib.load(f)
        return dill.load(f)

model = load_model()

def predict_price(input_data: dict) -> float:
    input_df = pd.DataFrame(input_data)
    prediction = np.expm1(model.predict(input_df)[0])
    return prediction
