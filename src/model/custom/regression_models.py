from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import pickle


# Save Model In artifacts
def save_model(model, path):
    pickle.dump(model, path)


# Regression Class for custom training
class RegressionModels:

    @staticmethod
    def linear_regression_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = LinearRegression(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def ridge_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = Ridge(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def lasso_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = Lasso(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def elastic_net_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = ElasticNet(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def decision_tree_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = DecisionTreeRegressor(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    @staticmethod
    def random_forest_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = RandomForestRegressor(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    def support_vector_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = SVR(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    def ada_boost_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = AdaBoostRegressor(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model

    def gradient_boosting_regressor(X_train, y_train, fit_model=False, **kwargs):
        model = GradientBoostingRegressor(**kwargs)
        if fit_model:
            model.fit(X_train, y_train)
            return model
        else:
            return model
