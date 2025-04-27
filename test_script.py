import preprocess
import train

df = preprocess.preprocess_cycle('data/Egypt_Houses_Price.csv')

print(df.info())

preprocessor, x_train, x_test, y_train, y_test = preprocess.features_operations(df)

metrics = train.model_train(preprocessor, x_train, x_test, y_train, y_test, 'light_gbm')

print(metrics)

