import dill
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Float32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.astype('float32')


MODEL_PATHS = {
    "Lightgbm": "models/lightgbm_best_model.pkl",
    # "random_forest": "models/random_forest_best_model.pkl",
    "SVR": "models/svr_best_model.pkl",
    "Xgboost": "models/xgboost_best_model.pkl",
    # "stacking": "models/stacking_model_best_model.pkl"
    
}


# Load all models once
def load_all_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        with open(path, "rb") as f:
            models[name] = dill.load(f)
    return models


# Load models at the start
MODELS = load_all_models()


def predict_price(input_data: dict, model_name: str = "Lightgbm") -> float:
    try:
        model = MODELS[model_name]
    except KeyError:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODELS.keys())}")
    
    input_df = pd.DataFrame(input_data)
    prediction = np.expm1(model.predict(input_df)[0])
    return prediction
