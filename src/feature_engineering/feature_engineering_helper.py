import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, MaxAbsScaler
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.utils.common.common_helper import read_config
from loguru import logger
from flask import session
import os
import numpy as np
from from_root import from_root

config_args = read_config("./config.yaml")

log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")


class FeatureEngineering:
    def __init__(self):
        pass

    @staticmethod
    def change_column_name(df, column, new_name):
        """[summary]
        Change Column Name
        Args:
            df ([type]): [description]
            column ([type]): [description]
            new_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        try:
            df = df.rename(columns={column: new_name})
            logger.info("Renaming Column implemented!")
            return df
        except Exception as e:
            logger.error(f"{e} occurred in Remaining Columns!")
            raise Exception(e)

    @staticmethod
    def change_data_type(df, column, type_):
        """[summary]
        Change Column DataType
        Args:
            df ([type]): [description]
            column ([type]): [description]
            type_ ([type]): [description]

        Returns:
            [type]: [description]
        """
        try:
            df[column] = df[column].astype(type_)
            logger.info("Change Datatype implemented !")
            return df
        except Exception as e:
            logger.error(f"{e} occurred in Change Datatype!")
            raise Exception(e)

    @staticmethod
    def train_test_Split(cleanedData, label, train_size, random_state):
        try:
            X_train, X_test, y_train, y_test = train_test_split(cleanedData, label, train_size=train_size,
                                                                random_state=random_state)
            logger.info("Train Test Split implemented!")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"{e} occurred in Train Test Split!")
            raise Exception(e)

    @staticmethod
    def scaler(data, typ):
        try:
            if typ == 'MinMax Scaler':
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                logger.info("MinMax Scaler implemented!")
                return scaled_data, scaler
        except Exception as e:
            logger.error(f"{e} occurred in Min Max Scaler!")
            raise Exception(e)

        try:
            if typ == 'Standard Scaler':
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                logger.info("Standard Scaler implemented!")
                return scaled_data, scaler
        except Exception as e:
            logger.error(f"{e} occurred in Standard Scaler!")
            raise Exception(e)

        try:

            if typ == 'Robust Scaler':
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(data)
                logger.info("Robust Scaler implemented!")
                return scaled_data, scaler
        except Exception as e:
            logger.error(f"{e} occurred in Robust Scaler")
            raise Exception(e)

        try:
            if typ == 'Power Transformer Scaler':
                scaler = PowerTransformer(method='yeo-johnson')
                scaled_data = scaler.fit_transform(data)
                logger.info("Power Transformer Scaler implemented!")

                return scaled_data, scaler
        except Exception as e:
            logger.error(f"{e} occurred in Power Transformer Scaler!")
            raise Exception(e)

        try:
            if typ == 'Max Abs Scaler':
                scaler = MaxAbsScaler()
                scaled_data = scaler.fit_transform(data)
                logger.info("Max Abs Scaler implemented!")
                return scaled_data, scaler
        except Exception as e:
            logger.error(f"{e} occurred in Max Abs Scaler!")
            raise Exception(e)

    @staticmethod
    def encodings(df, cols, kind: str, **kwargs):
        try:
            if kind == 'Label/Ordinal Encoder':
                label = ce.OrdinalEncoder(cols=cols, **kwargs)
                label_df = label.fit_transform(df)
                logger.info("Label/Ordinal Encoder implemented!")
                return label_df, label

        except Exception as e:
            logger.error(f"{e} occurred in Label/Ordinal Encoder!")
            raise Exception(e)

        try:
            if kind == 'One Hot Encoder':
                onehot = ce.OneHotEncoder(cols=cols)
                onehot_df = onehot.fit_transform(df)
                logger.info("One Hot Encoder implemented!")
                
                #  onehot_df=pd.get_dummies(df, cols, drop_first=True)
                # logger.info("One Hot Encoder implemented!")
                # return (onehot_df,None)  
            
                return onehot_df, onehot
        except Exception as e:
            logger.error(f"{e} occurred in One Hot Encoder!")
            raise Exception(e)

        try:
            if kind == 'Binary Encoder':
                onehot = ce.BinaryEncoder(cols=cols, **kwargs)
                onehot_df = onehot.fit_transform(df)
                logger.info("Binary Encoder implemented!")
                return onehot_df, onehot
        except Exception as e:
            logger.error(f"{e} occurred in Binary Encoder!")
            raise Exception(e)

        try:
            if kind == 'Base N Encoder':
                onehot = ce.BaseNEncoder(cols=cols)
                onehot_df = onehot.fit_transform(df)
                logger.info("Base N Encoder implemented !")

                return onehot_df, onehot
        except Exception as e:
            logger.error(f"{e} occurred in Base N Encoder!")
            raise Exception(e)

        try:

            if kind == 'Hash Encoder':
                hash_ = ce.HashingEncoder(cols=cols, **kwargs)
                hash_df = hash_.fit_transform(df)
                logger.info("Hash Encoder implemented!")

                return hash_df, hash_
        except Exception as e:
            logger.error(f"{e} occurred in Hash Encoder!")
            raise Exception(e)

        try:

            if kind == 'Target Encoder':
                target = ce.TargetEncoder(cols=cols)
                target_df = target.fit_transform(df, **kwargs)
                logger.info("Target Encoder implemented!")
                return target_df, target
        except Exception as e:
            logger.error(f"{e} occurred in Target Encoder!")
            raise Exception(e)

    def handleDatetime(self, frame, cols):
        try:
            year = pd.DataFrame()
            day = pd.DataFrame()
            month = pd.DataFrame()
            count = 0
            for col in cols:
                frame[col] = pd.to_datetime(frame[col])

                year[f'{col}_{count}'] = pd.to_datetime(frame[col]).dt.year
                count += 1
                day[f'{col}_{count}'] = pd.to_datetime(frame[col]).dt.day
                count += 1
                month[f'{col}_{count}'] = pd.to_datetime(frame[col]).dt.month
                count += 1

            frame = pd.concat([frame, year, day, month], axis=1)
            logger.info("Handle Date-Time implemented!")
            return frame
        except Exception as e:
            logger.error(f"{e} occurred in Handle Date-Time Encoder!")
            raise Exception(e)

    @staticmethod
    def feature_selection(features, target, typ, dclas=None, **kwargs):
        important_features = pd.DataFrame()
        try:
            if typ == 'SelectKBest':
                # chi2 + anova test
                best_features = SelectKBest(score_func=chi2, **kwargs)
                imp = best_features.fit(features, target)
                important_features['score'] = imp.scores_
                important_features['columns'] = features.columns
                logger.info("SelectKBest implemented!")
                return important_features.sort_values('scores', ascending=False)

            elif typ == 'Find Constant Features':

                # chi2 + anova test
                vari_thr = VarianceThreshold(**kwargs)
                imp = vari_thr.fit(features)
                logger.info("Find Constant Features implemented!")
                return features.columns[imp.get_support()]

            elif typ == 'Extra Trees Classifier':

                best_features = ExtraTreesClassifier()
                best_features.fit(features, target)
                df = pd.DataFrame()
                df['Value'] = best_features.feature_importances_
                df['Feature'] = features.columns
                logger.info("Extra Trees Classifier implemented!")
                return df.sort_values(by='Value', ascending=False)

            elif typ == 'Extra Trees Regressor':

                best_features = ExtraTreesRegressor()
                best_features.fit(features, target)
                important_features['score'] = best_features.feature_importances_
                important_features['columns'] = features.columns
                logger.info("Extra Trees Regressor implemented!")
                return important_features.sort_values('scores', ascending=False)

            elif typ == 'Mutual Info Classification':

                importances = mutual_info_classif(features, target)
                df = pd.DataFrame()
                df['Feature'] = features.columns
                df['Value'] = importances
                return df.sort_values('Value', ascending=False)

            elif typ == 'Mutual Info Regressor':
                importances = mutual_info_regression(features, target)
                df = pd.DataFrame()

                df['Feature'] = features.columns
                df['Value'] = importances
                return df.sort_values('Value', ascending=False)

            elif typ == 'Backward Elimination':
                if session['project_type'] == 2:
                    dclas = DecisionTreeClassifier()
                    sfs = SequentialFeatureSelector(dclas, direction='backward', scoring='balanced_accuracy', **kwargs)
                    sfs.fit(features, target)
                    logger.info("Backward Elimination implemented!")
                    return list(features.columns[sfs.get_support()])
                else:
                    dclas = DecisionTreeRegressor()
                    sfs = SequentialFeatureSelector(dclas, direction='backward', scoring='neg_mean_absolute_error',
                                                    **kwargs)
                    sfs.fit(features, target)
                    logger.info("Backward Elimination implemented!")
                    return list(features.columns[sfs.get_support()])

            elif typ == 'Forward Selection':
                if session['project_type'] == 2:
                    dclas = DecisionTreeClassifier()
                    sfs = SequentialFeatureSelector(dclas, direction='forward', scoring='balanced_accuracy', **kwargs)
                    sfs.fit(features, target)
                    logger.info("Backward Elimination implemented!")
                    return list(features.columns[sfs.get_support()])
                else:
                    dclas = DecisionTreeRegressor()
                    sfs = SequentialFeatureSelector(dclas, direction='forward', scoring='neg_mean_absolute_error',
                                                    **kwargs)
                    sfs.fit(features, target)
                    logger.info("Backward Elimination implemented!")
                    return list(features.columns[sfs.get_support()])

        except Exception as e:
            logger.error(f"{e} occurred in Backward Elimination!")
            raise Exception(e)

    @staticmethod
    def dimenstion_reduction(data, comp):
        try:
            model = PCA(n_components=comp)
            pca = model.fit_transform(data)
            logger.info("PCA implemented !")
            return pca, np.cumsum(model.explained_variance_ratio_), model
        except Exception as e:
            logger.error(f"{e} occurred in PCA!")
            raise Exception(e)
