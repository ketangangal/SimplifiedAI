import numpy as np
import pandas as pd
import os
from src.utils.common.common_helper import read_config
from loguru import logger
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from from_root import from_root

config_args = read_config("./config.yaml")

log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")


class ModelTrain_Regression:
    def __init__(self, X_train, X_test, y_train, y_test, start: bool):
        try:
            logger.info("Constructor created in Auto Regression!")
            self.frame = pd.DataFrame(columns=['Model_Name', 'MAE', 'RMSE', 'R2-Score'])
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            if start:
                logger.info("Auto Regression training started!")
                self.linear_regression_()
                self.Lasso_()
                self.ridge_()
                self.ElasticNet_()
                self.SVR_()
                self.KNeighborsRegressor_()
                self.DecisionTreeRegressor_()
                self.RandomForestRegressor_()
                self.AdaBoostRegressor_()
                self.GradientBoostingRegressor_()
            logger.info("Auto Regression training completed!")
        except Exception as e:
            logger.error(f"{e} occurred in Auto Regression model training!")
            raise Exception(e)

    def linear_regression_(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append({'Model_Name': 'LinearRegression', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
                                       ignore_index=True)
        logger.info(f"Linear Regression - MAE :{MAE} RMSE: {RMSE}")

    def ridge_(self):
        model = Ridge()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append({'Model_Name': 'Ridge', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
                                       ignore_index=True)
        logger.info(f"Ridge Regression - MAE :{MAE} RMSE: {RMSE}")

    def Lasso_(self):
        model = Lasso()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append({'Model_Name': 'Lasso', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
                                       ignore_index=True)
        logger.info(f"Lasso Regression - MAE :{MAE} RMSE: {RMSE}")

    def ElasticNet_(self):
        model = ElasticNet()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append({'Model_Name': 'ElasticNet', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
                                       ignore_index=True)
        logger.info(f"Elastic Net - MAE :{MAE} RMSE: {RMSE}")

    def SVR_(self):
        model = SVR()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append({'Model_Name': 'SVR', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
                                       ignore_index=True)
        logger.info(f"SVR - MAE :{MAE} RMSE: {RMSE}")

    def KNeighborsRegressor_(self):
        model = KNeighborsRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append({'Model_Name': 'KNeighborsRegressor', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
                                       ignore_index=True)
        logger.info(f"KNN Regressor - MAE :{MAE} RMSE: {RMSE}")

    def DecisionTreeRegressor_(self):
        model = DecisionTreeRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append(
            {'Model_Name': 'DecisionTreeRegressor', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
            ignore_index=True)
        logger.info(f"Decision Tree Regressor - MAE :{MAE} RMSE: {RMSE}")

    def RandomForestRegressor_(self):
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append(
            {'Model_Name': 'RandomForestRegressor', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
            ignore_index=True)
        logger.info(f"Random Forest Regressor - MAE :{MAE} RMSE: {RMSE}")

    def AdaBoostRegressor_(self):
        model = AdaBoostRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append({'Model_Name': 'AdaBoostRegressor', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
                                       ignore_index=True)
        logger.info(f"ADA Boost Regressor - MAE :{MAE} RMSE: {RMSE}")

    def GradientBoostingRegressor_(self):
        model = GradientBoostingRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        MAE = mean_absolute_error(self.y_test, y_pred)
        RMSE = np.sqrt(mean_squared_error(self.y_test, y_pred))
        R2 = r2_score(self.y_test, y_pred)
        self.frame = self.frame.append(
            {'Model_Name': 'GradientBoostingRegressor', 'MAE': MAE, 'RMSE': RMSE, 'R2-Score': R2},
            ignore_index=True)
        logger.info(f"Gradient Boost Regressor - MAE :{MAE} RMSE: {RMSE}")

    def results(self):
        return self.frame.sort_values('MAE')
