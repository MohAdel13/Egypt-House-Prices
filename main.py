from data import preprocess
from features import build_features
from models import train

df = preprocess.preprocess_cycle('data/Egypt_Houses_Price.csv')

preprocessor, x_train, x_test, y_train, y_test = build_features.features_operations(df)

metrics = train.model_train(preprocessor, x_train, x_test, y_train, y_test, 'light_gbm')

print(metrics)
