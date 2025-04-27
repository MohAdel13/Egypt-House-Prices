import dill
import numpy as np
import pandas as pd


def load_model():
    """
        Load the pre-trained model.

        This function uses the `dill` library to deserialize the model from the
        specified file path and returns
        the trained model for further predictions.

        Returns:
            model: The deserialized model.
        """

    with open("models/lightgbm_best_model.pkl", "rb") as f:
        return dill.load(f)


model = load_model()


def predict_price(input_data: dict) -> float:
    """
       Predict the price using the loaded model.

       This function takes in a dictionary of input features, converts it into a
       pandas DataFrame, and then uses the pre-trained model to make a price prediction.
       The prediction is transformed using the inverse of the log transformation
       (`np.expm1`) to return the actual price.

       Args:
           input_data (dict): A dictionary containing feature names as keys and their
                               respective values as input for prediction.

       Returns:
           float: The predicted price, transformed back from the log scale.
       """

    input_df = pd.DataFrame(input_data)
    prediction = np.expm1(model.predict(input_df)[0])
    return prediction
