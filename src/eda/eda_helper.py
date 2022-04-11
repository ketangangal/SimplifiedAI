import pandas as pd
import numpy as np
from src.utils.common.common_helper import read_config
from loguru import logger
import os
from from_root import from_root

config_args = read_config("./config.yaml")

"""[summary]
Class for EDA Operations
Returns:
    [type]: [description]
"""
log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])

logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")


class EDA:
    data_types = ['bool', "int_", "int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float_", "float16", "float32", "float64"]

    @staticmethod
    def five_point_summary(dataframe):
        """[summary]
        Return 5 Point Summary For Given  Dataset
        Args:
            dataframe ([type]): [description]

        Returns:
            [type]: DataFrame/ Exception
        """
        my_dict = {'Features': [], 'Min': [], 'Q1': [], 'Median': [], 'Q3': [],
                   'Max': []}
        for column in dataframe.select_dtypes(include=np.number).columns:
            try:
                column_data = dataframe[pd.to_numeric(dataframe[column], errors='coerce').notnull()][column]
                q1 = np.percentile(column_data, 25)
                q3 = np.percentile(column_data, 75)

                my_dict['Features'].append(column)
                my_dict['Min'].append(np.min(column_data))
                my_dict['Q1'].append(q1)
                my_dict['Median'].append(np.median(column_data))
                my_dict['Q3'].append(q3)
                my_dict['Max'].append(np.max(column_data))
            except Exception as e:
                logger.error(f"{e} occurred in Five Point Summary!")
        logger.info('Five Point Summary Implemented!')
        return pd.DataFrame(my_dict).sort_values(by=['Features'], ascending=True)

    @staticmethod
    def data_dtype_info(df):
        df_ = pd.DataFrame()
        df_['Column'] = list(df.dtypes.index)
        df_['DataType'] = list(df.dtypes.values)
        df_['Null Count'] = list(df.isnull().sum().values)

        return df_

    @staticmethod
    def correlation_report(dataframe, method='pearson'):
        try:
            logger.info("Correlation Report Implemented!")
            return dataframe.corr(method=method)

        except Exception as e:
            logger.error(f"{e} occurred in Correlation Plot!")

    @staticmethod
    def get_no_records(dataframe, count=100, order='top'):
        try:
            if order == 'top':
                logger.info("Get No of Records Implemented!")
                return dataframe.head(count)
            else:
                logger.info("Get No of Records Implemented!")
                return dataframe.tail(count)

        except Exception as e:
            logger.error(f"{e} occurred in Get No of Records!")

    @staticmethod
    def find_dtypes(df3):
        try:
            for i in df3.columns:
                yield str(df3[i].dtypes)
            logger.info("Find Dtypes Implemented!")
        except Exception as e:
            logger.error(f"{e} occurred in Find Dtypes!")

    @staticmethod
    def find_median(df3):
        try:
            for i in df3.columns:
                if df3[i].dtypes in EDA.data_types:
                    yield str(round(df3[i].median(), 2))
                else:
                    yield str('-')
            logger.info("Find Median Implemented!")
        except Exception as e:
            logger.error(f"{e} occurred in Find Median!")

    @staticmethod
    def find_mode(df3):
        try:
            for i in df3.columns:
                mode = df3[i].mode()
                yield mode[0] if len(mode) > 0 else '-'
            logger.info("Find Mode Implemented!")
        except Exception as e:
            logger.error(f"{e} occurred in Find Mode!")

    @staticmethod
    def find_mean(df3):
        try:
            for i in df3.columns:
                if df3[i].dtypes in EDA.data_types:
                    yield str(round(df3[i].mean(), 2))
                else:
                    yield str('-')
            logger.info("Find Mean Implemented!")
        except Exception as e:
            logger.error(f"{e} occurred in Find Mean!")

    @staticmethod
    def missing_cells_table(df):
        try:
            df = df[[col for col in df.columns if df[col].isnull().any()]]

            missing_value_df = pd.DataFrame({
                'Column': df.columns,
                'Missing values': df.isnull().sum(),
                'Missing values (%)': (df.isnull().sum() / len(df)) * 100,
                'Mean': EDA.find_mean(df),
                'Median': EDA.find_median(df),
                'Mode': EDA.find_mode(df),
                'Datatype': EDA.find_dtypes(df)
            }).sort_values(by='Missing values', ascending=False)
            logger.info("Missing Cells Table Implemented!")
            return missing_value_df
        except Exception as e:
            logger.error(f"{e} occurred in Missing Cells Table!")

    @staticmethod
    def outlier_detection_iqr(dataframe, lower_bound=25, upper_bound=75):
        my_dict = {'Features': [], f'IQR ({lower_bound}-{upper_bound})': [], 'Q3 + 1.5*IQR': [], 'Q1 - 1.5*IQR': [],
                   'Upper outlier count': [],
                   'Lower outlier count': [], 'Total outliers': [], 'Outlier percent': []}
        for column in dataframe.select_dtypes(include=np.number).columns:
            try:
                upper_count = 0
                lower_count = 0
                q1 = np.percentile(dataframe[column].fillna(dataframe[column].mean()), lower_bound)
                q3 = np.percentile(dataframe[column].fillna(dataframe[column].mean()), upper_bound)
                IQR = round(q3 - q1)
                upper_limit = round(q3 + (IQR * 1.5))
                lower_limit = round(q1 - (IQR * 1.5))

                for element in dataframe[column].fillna(dataframe[column].mean()):
                    if element > upper_limit:
                        upper_count += 1
                    elif element < lower_limit:
                        lower_count += 1

                my_dict['Features'].append(column)
                my_dict[f'IQR ({lower_bound}-{upper_bound})'].append(IQR)
                my_dict['Q3 + 1.5*IQR'].append(upper_limit)
                my_dict['Q1 - 1.5*IQR'].append(lower_limit)
                my_dict['Upper outlier count'].append(upper_count)
                my_dict['Lower outlier count'].append(lower_count)
                my_dict['Total outliers'].append(upper_count + lower_count)
                my_dict['Outlier percent'].append(round((upper_count + lower_count) / len(dataframe[column]) * 100, 2))
            except Exception as e:
                logger.error(f"{e} occurred in Outlier Detection IQR!")
        logger.info("Outlier Detection IQR Implemented!")
        return pd.DataFrame(my_dict).sort_values(by=['Total outliers'], ascending=False)

    @staticmethod
    def z_score_outlier_detection(dataframe):
        my_dict = {"Features": [], "Mean": [], "Standard deviation": [], 'Upper outlier count': [],
                   'Lower outlier count': [], 'Total outliers': [], 'Outlier percent': []}

        for column in dataframe.select_dtypes(include=np.number).columns:
            try:
                upper_outlier = 0
                lower_outlier = 0
                col_mean = np.mean(dataframe[column])
                col_std = np.std(dataframe[column])

                for element in dataframe[column]:
                    z = (element - col_mean) / col_std
                    if z > 3:
                        upper_outlier += 1
                    elif z < -3:
                        lower_outlier += 1

                my_dict["Features"].append(column)
                my_dict["Mean"].append(col_mean)
                my_dict["Standard deviation"].append(col_std)
                my_dict["Upper outlier count"].append(upper_outlier)
                my_dict["Lower outlier count"].append(lower_outlier)
                my_dict["Total outliers"].append(upper_outlier + lower_outlier)
                my_dict["Outlier percent"].append(
                    round((upper_outlier + lower_outlier) / len(dataframe[column]) * 100, 2))

            except Exception as e:
                logger.error(f"{e} occurred in Outlier Detection Zscore!")
        df = pd.DataFrame(my_dict).sort_values(by=['Total outliers'], ascending=False).reset_index()
        logger.info("Outlier Detection Zscore Implemented!")
        return df

    @staticmethod
    def outlier_detection(data, kind: str):
        try:
            data = pd.Series(data)
            if kind == 'Box':
                pass
            elif kind == 'z-score':
                outliers = []
                # threshold = 3
                mean = np.mean(data)
                std = np.std(data)
                data = np.array(data)
                for da in data:
                    val = (da - mean) / std
                    if val > 3:
                        outliers.append(da)
                    elif val < -3:
                        outliers.append(da)
                return outliers
            elif kind == 'iqr':
                outliers = []
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                data = np.array(data)
                lower_bound_value = q1 - 1.5 * iqr
                upper_bound_value = q3 + 1.5 * iqr

                for da in data:
                    if da < lower_bound_value or da > upper_bound_value:
                        outliers.append(da)

                return outliers
            logger.info("Outlier Detection Implemented!")
        except Exception as e:
            logger.error(f"{e} occurred in Outlier Detection Zscore!")
