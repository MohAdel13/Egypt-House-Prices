from sklearn.metrics import mean_squared_error, r2_score


def evaluate(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    return {"Train_MSE": train_mse, "Train_R2": train_r2, 'Test_MSE': test_mse, 'Test_R2': test_r2}
