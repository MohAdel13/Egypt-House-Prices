from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def model_train(preprocessor, x_train, x_test, y_train, y_test, typ):
    """
        Train and evaluate a regression model with MLflow tracking.

        Args:
            preprocessor: Preprocessing pipeline (e.g., ColumnTransformer).
            x_train (DataFrame): Training feature data.
            x_test (DataFrame): Testing feature data.
            y_train (Series): Training target data.
            y_test (Series): Testing target data.
            typ (str): Model type to train ('light_gbm', 'random_forest', 'svr', 'xgboost').

        Returns:
            dict: Dictionary containing training and testing MSE and R2 scores.
        """

    preprocessor.fit_transform(x_train)

    trained_model = None
    best_parameters = None

    with mlflow.start_run(run_name=typ):
        if typ == 'light_gbm':
            trained_model, best_parameters = light_gbm(preprocessor, x_train, y_train)

        elif typ == 'random_forest':
            trained_model, best_parameters = random_forest(preprocessor, x_train, y_train)

        elif typ == 'svr':
            trained_model, best_parameters = svr(preprocessor, x_train, y_train)

        elif typ == 'xgboost':
            trained_model, best_parameters = xgboost(preprocessor, x_train, y_train)

        metrics = evaluate(trained_model, x_train, y_train, x_test, y_test)

        mlflow.log_params(best_parameters)
        mlflow.log_metric("Train_MSE", metrics['Train_MSE'])
        mlflow.log_metric("Train_R2", metrics['Train_R2'])
        mlflow.log_metric("Test_MSE", metrics['Test_MSE'])
        mlflow.log_metric("Test_R2", metrics['Test_R2'])
        mlflow.sklearn.log_model(trained_model, typ + "_model")

    return metrics


def light_gbm(preprocessor, x_train, y_train):
    """
       Train a LightGBM regressor model with GridSearchCV.

       Args:
           preprocessor: Preprocessing pipeline.
           x_train (DataFrame): Training features.
           y_train (Series): Training target.

       Returns:
           Tuple: (best_estimator, best_params) from GridSearchCV.
       """

    model = LGBMRegressor(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # param_grid = {
    #     'model__n_estimators': [50, 100, 200],
    #     'model__learning_rate': [0.01, 0.1, 0.2],
    #     'model__max_depth': [-1, 10, 20]
    # }

    param_grid = {
        'model__n_estimators': [200],
        'model__learning_rate': [0.2],
        'model__max_depth': [10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


class Float32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype('float32')


def random_forest(preprocessor, x_train, y_train):
    """
        Train a Random Forest regressor model with GridSearchCV.

        Args:
            preprocessor: Preprocessing pipeline.
            x_train (DataFrame): Training features.
            y_train (Series): Training target.

        Returns:
            Tuple: (best_estimator, best_params) from GridSearchCV.
        """

    model = RandomForestRegressor(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # param_grid = {
    #     'model__n_estimators': [100, 200, 300, 400]
    #     ,
    #     'model__max_depth': [30, 40, 50, 60],
    #     'model__min_samples_split': [5, 7, 10, 15],
    #     'model__min_samples_leaf': [1, 2, 4, 6]
    # }

    param_grid = {
        'model__n_estimators': [400],
        'model__max_depth': [30],
        'model__min_samples_split': [5],
        'model__min_samples_leaf': [1]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def svr(preprocessor, x_train, y_train):
    """
        Train a Support Vector Regressor (SVR) model with GridSearchCV.

        Args:
            preprocessor: Preprocessing pipeline.
            x_train (DataFrame): Training features.
            y_train (Series): Training target.

        Returns:
            Tuple: (best_estimator, best_params) from GridSearchCV.
        """

    model = SVR()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    param_grid = {
        'model__kernel': ['linear', 'rbf', 'poly']
        , 'model__C': [0.1, 1, 10]
        , 'model__epsilon': [0.1, 0.2, 0.5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def xgboost(preprocessor, x_train, y_train):
    """
        Train an XGBoost regressor model with GridSearchCV.

        Args:
            preprocessor: Preprocessing pipeline.
            x_train (DataFrame): Training features.
            y_train (Series): Training target.

        Returns:
            Tuple: (best_estimator, best_params) from GridSearchCV.
        """

    model = XGBRegressor(tree_method='hist', random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # param_grid = {
    #     'model__n_estimators': [50, 100, 200],
    #     'model__learning_rate': [0.01, 0.1, 0.2],
    #     'model__max_depth': [3, 5, 7]
    # }

    param_grid = {
        'model__n_estimators': [200],
        'model__learning_rate': [0.1],
        'model__max_depth': [7]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate(model, X_train, y_train, X_test, y_test):
    """
        Evaluate a trained regression model on training and testing data.

        Args:
            model: Trained model.
            X_train (DataFrame): Training feature data.
            y_train (Series): Training target data.
            X_test (DataFrame): Testing feature data.
            y_test (Series): Testing target data.

        Returns:
            dict: Dictionary containing training/testing MSE and R2 scores.
        """

    train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)

    test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    return {"Train_MSE": train_mse, "Train_R2": train_r2, 'Test_MSE': test_mse, 'Test_R2': test_r2}
