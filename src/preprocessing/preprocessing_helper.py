import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import normalize
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss
from scipy.stats import skew, kurtosis
from src.utils.common.common_helper import read_config
from loguru import logger
import os
from from_root import from_root

config_args = read_config("./config.yaml")

log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")


class Preprocessing:

    @staticmethod
    def get_data(filepath):
        try:
            data = filepath
            df = pd.read_csv(data)
            logger.info("Data successfully loaded into data frame")
            return df
        except Exception as e:
            logger.info(e)

    @staticmethod
    def col_seperator(df, typ: str):
        try:
            logger.info("Column Separator Type {typ}")
            if typ == 'Numerical_columns':
                Numerical_columns = df.select_dtypes(exclude='object')
                logger.info("Successfully Implemented")
                return Numerical_columns

            elif typ == 'Categorical_columns':
                Categorical_columns = df.select_dtypes(include='object')
                logger.info("Successfully Implemented")
                return Categorical_columns
            else:
                logger.error("Type Not Found")
        except Exception as e:
            logger.info(e)

    @staticmethod
    def delete_col(df, cols: list):

        temp_list = []
        for i in cols:
            if i in df.columns:
                temp_list.append(i)
            else:
                raise Exception('Column Not Found')

        try:
            df = df.drop(temp_list, axis=1)
            logger.info("Column Successfully Dropped!")
            return df
        except Exception as e:
            logger.info(e)
            raise e

    @staticmethod
    def missing_values(df):
        try:
            columns = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False).index
            values = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False).values

            mv_df = pd.DataFrame(columns, columns=['Columns'])

            mv_df['Missing_Values'] = values
            mv_df['Percentage'] = np.round((values / len(df)) * 100, 2)
            logger.info("Missing Values Successfully Implemented")
            return columns, values, mv_df

        except Exception as e:
            logger.info(e)

    @staticmethod
    def find_skewness(x):
        try:
            logger.info(f"Skewness : {skew(x)}")
            return skew(x)
        except Exception as e:
            logger.error(e)

    @staticmethod
    def find_kurtosis(x):
        try:
            logger.info(f"Skewness : {kurtosis(x)}")
            return kurtosis(x)
        except Exception as e:
            logger.error(e)

    @staticmethod
    def fill_numerical(df, typ, cols, value=None):
        for i in cols:
            if i in df.columns:
                continue
            else:
                return 'Column Not Found'
        if typ == 'Mean':
            try:
                logger.info("Missing Values Filled with Mean")
                return df[cols].fillna(df[cols].mean())
            except Exception as e:
                logger.info(e)
        elif typ == 'Median':
            try:
                logger.info("Missing Values Filled with Mean")
                return df[cols].fillna(df[cols].median())
            except Exception as e:
                logger.info(e)
        elif typ == 'Arbitrary Value':
            try:
                logger.info("Missing Values Filled with Arbitrary Value")
                return df[cols].fillna(value)
            except Exception as e:
                logger.info(e)

        elif typ == 'Interpolate':
            try:
                logger.info("Missing Values Filled with Interpolate")
                return df[cols].interpolate(value)
            except Exception as e:
                logger.info(e)
        else:
            logger.error("Invalid Input")
            return 'Type Not present'

    @staticmethod
    def fill_categorical(df, typ=None, col=None, value=None):
        # Replace na with some meaning of na
        try:
            if typ == 'replace':
                temp_list = []
                for i in col:
                    if i in df.cols:
                        temp_list.append(i)
                    else:
                        return 'Column Not Found'

                if col and value is not None:
                    logger.info("Categorical Values Filled with Replace")
                    return df[col].fillna(value)
                else:
                    return 'Please provide values and columns'

            elif typ == 'Mode':
                if col is not None:
                    logger.info("Categorical Values Filled with Mode")
                    return df[col].fillna(df.mode()[col][0])
                else:
                    return 'Please give provide values and columns'

            elif typ == 'New Category':
                if col is not None:
                    logger.info("Categorical Values Filled with New Category")
                    return df[col].fillna(value)
                else:
                    return 'Please give provide values and columns'
            else:
                logger.error("Invalid Input")
                return 'Type not found'
        except Exception as e:
            logger.error(e)

    @staticmethod
    def Unique(df, percent):
        try:
            percent = percent / 25
            holder = []
            for column in df.columns:
                if df[column].nunique() > int(len(df) * percent / 4):
                    print(column, '+', df[column].unique())
                    holder.append(column)
            logger.info(f"Found {holder} Unique elements!")
            return holder
        except Exception as e:
            logger.error(e)

    @staticmethod
    def encodings(df, cols, kind: str):
        try:
            if kind == 'One Hot Encoder':
                onehot = ce.OneHotEncoder(cols=cols)
                onehot_df = onehot.fit_transform(df)
                logger.info("One Hot Encoding Implemented!")
                return onehot_df
            elif kind == 'Dummy Encoder':
                dummy_df = pd.get_dummies(data=cols, drop_first=True)
                logger.info("Dummy Encoding Implemented!")
                return dummy_df
            elif kind == 'Effective Encoder':
                target = ce.TargetEncoder(cols=cols)
                target_df = target.fit_transform(df)
                logger.info("Effective Encoding Implemented!")
                return target_df
            elif kind == 'Binary Encoder':
                binary = ce.BinaryEncoder(cols=cols, return_df=True)
                binary_df = binary.fit_transform(df)
                logger.info("Binary Encoding Implemented!")
                return binary_df
            elif kind == 'Base N Encoder':
                basen = ce.BaseNEncoder(cols=cols)
                basen_df = basen.fit_transform(df)
                logger.info("Base N Encoding Implemented!")
                return basen_df
            else:
                logger.error("Wrong Input!")
        except Exception as e:
            logger.error(e)

    @staticmethod
    def balance_data(df, kind: str, target):
        try:
            if len(df[(df[target] == 0)]) >= df[(df[target] == 1)]:
                df_majority = df[(df[target] == 0)]
                df_minority = df[(df[target] == 1)]
            else:
                df_majority = df[(df[target] == 1)]
                df_minority = df[(df[target] == 0)]

            logger.info("Found Majority and Minority CLasses")
            if kind == 'UnderSampling':
                df_majority_undersampled = resample(df_majority,
                                                    replace=True,
                                                    n_samples=len(df_minority),
                                                    random_state=42)
                logger.info("UnderSampling Implemented")
                return pd.concat([df_majority_undersampled, df_minority])

            elif kind == 'UpSampling':
                df_minority_upsampled = resample(df_minority,
                                                 replace=True,
                                                 n_samples=len(df_majority),
                                                 random_state=42)
                logger.info("UpSampling Implemented")
                return pd.concat([df_minority_upsampled, df_majority])

            elif kind == 'Smote':
                sm = SMOTE(sampling_strategy='minority', random_state=42)
                oversampled_X, oversampled_Y = sm.fit_sample(df.drop(target, axis=1), df[target])
                oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
                logger.info("Smote Implemented")
                return oversampled
            else:
                logger.info("No Method Found")
                return 'Please specify correct mtd'
        except Exception as e:
            logger.error(e)

    @staticmethod
    def drop_duplicate(df, cols: list):
        try:
            df = df.drop_duplicates(subset=cols, inplace=True)
            logger.info("Drop Duplicate Implemented!")
            return df
        except Exception as e:
            logger.info(e)

    @staticmethod
    def handle_low_variance(df, var_range):
        try:
            Categorical_columns = df.select_dtypes(include='object')
            df = df.drop(Categorical_columns, axis=1, inplace=True)
            normalize_df = normalize(df)
            df_scaled = pd.DataFrame(normalize_df)
            variance = df_scaled.var()
            cols = df.columns
            variable = []
            for i in range(0, len(variance)):
                if variance[i] >= var_range:
                    variable.append(cols[i])
            new_df = df[variable]
            logger.info("Low Variance successfully handled")
            return new_df
        except Exception as e:
            logger.info(e)

    @staticmethod
    def handle_high_variance(df, var_range):
        try:
            Categorical_columns = df.select_dtypes(include='object')
            df = df.drop(Categorical_columns, axis=1, inplace=True)
            normalize_df = normalize(df)
            df_scaled = pd.DataFrame(normalize_df)
            variance = df_scaled.var()
            cols = df.columns
            variable = []
            for i in range(0, len(variance)):
                if variance[i] <= var_range:
                    variable.append(cols[i])
            new_df = df[variable]
            logger.info("High Variance successfully handled")
            return new_df
        except Exception as e:
            logger.error(e)
            return e.__str__()

    @staticmethod
    def under_sample(dataframe, target_col, class_dict):
        try:
            X = dataframe.drop(columns=[target_col])
            y = dataframe[target_col]

            dict_ = {}
            for item in class_dict:
                dict_[item[0]] = item[1]

            ns = NearMiss(dict_)
            X_resampled, y_resampled = ns.fit_resample(X, y)

            resampled_dataset = X_resampled.join(y_resampled)
            logger.info("Under Sampling Implemented!")
            return resampled_dataset
        except Exception as e:
            logger.info(e)
            raise e

    @staticmethod
    def over_sample(dataframe, target_col, class_dict):
        try:
            X = dataframe.drop(columns=[target_col])
            y = dataframe[target_col]

            dict_ = {}
            for item in class_dict:
                dict_[item[0]] = item[1]
            ros = RandomOverSampler(dict_)
            X_resampled, y_resampled = ros.fit_resample(X, y)

            resampled_dataset = X_resampled.join(y_resampled)
            logger.info("Over sampling Implemented!")
            return resampled_dataset
        except Exception as e:
            logger.info(e)
            raise e

    @staticmethod
    def smote_technique(dataframe, target_col, class_dict):
        try:
            X = dataframe.drop(columns=[target_col])
            y = dataframe[target_col]

            dict_ = {}
            for item in class_dict:
                dict_[item[0]] = item[1]

            sm = SMOTE(dict_)
            X_resampled, y_resampled = sm.fit_resample(X, y)

            resampled_dataset = X_resampled.join(y_resampled)
            logger.info("Smote Successfully Implemented!")
            return resampled_dataset
        except Exception as e:
            logger.info(e)
            raise e
