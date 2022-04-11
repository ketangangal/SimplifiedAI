import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from src.utils.common.common_helper import read_config
from loguru import logger
import os
from from_root import from_root

config_args = read_config("./config.yaml")

log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")


class ModelTrain_Classification:
    def __init__(self, X_train, X_test, y_train, y_test, start: bool):
        try:
            logger.info("Constructor Created!")
            self.frame = pd.DataFrame(columns=['Model_Name', 'Accuracy', 'Classes', 'Precision', 'Recall', 'F1_Score'])
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            if start:
                logger.info("Auto Classification Training Started!")
                self.LogisticRegression_()
                self.SVC_()
                self.KNeighborsClassifier_()
                self.DecisionTreeClassifier_()
                self.RandomForestClassifier_()
                self.GradientBoostingClassifier_()
                self.AdaBoostClassifier_()
                logger.info("Auto Classification Training Completed!")

        except Exception as e:
            logger.error(f"{e} occurred in Auto Classification model training!")
            raise Exception(e)
        
    def LogisticRegression_(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'LogisticRegression', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def SVC_(self):
        model = SVC()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'SVC', 'Accuracy': accuracy, 'Classes': self.y_test.unique(), 'Precision': precision,
             'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def KNeighborsClassifier_(self):
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'KNeighborsClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def DecisionTreeClassifier_(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'DecisionTreeClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def RandomForestClassifier_(self):
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'RandomForestClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def GradientBoostingClassifier_(self):
        model = GradientBoostingClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'GradientBoostingClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def AdaBoostClassifier_(self):
        model = AdaBoostClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1_score_ = f1_score(self.y_test, y_pred, average=None)
        self.frame = self.frame.append(
            {'Model_Name': 'AdaBoostClassifier', 'Accuracy': accuracy, 'Classes': self.y_test.unique(),
             'Precision': precision, 'Recall': recall,
             'F1_Score': f1_score_}, ignore_index=True)

    def results(self):
        return self.frame