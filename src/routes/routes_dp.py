from flask import Blueprint, request, render_template, session, redirect, url_for
from src.utils.common.data_helper import load_data, update_data
from src.utils.common.plotly_helper import PlotlyHelper
from src.utils.common.project_report_helper import ProjectReports
from src.eda.eda_helper import EDA
from src.constants.constants import ProjectActions
from src.utils.common.common_helper import read_config
import os
from loguru import logger
from from_root import from_root
import pandas as pd
import numpy as np
from src.preprocessing.preprocessing_helper import Preprocessing
from src.constants.constants import NUMERIC_MISSING_HANDLER, OBJECT_MISSING_HANDLER

config_args = read_config("./config.yaml")

log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")

app_dp = Blueprint('dp', __name__)


@app_dp.route('/dp/<action>', methods=['GET'])
def data_preprocessing(action):
    try:
        if 'pid' in session and 'id' in session:
            df = load_data()
            if df is not None:
                if action == "delete-columns":
                    col_lst = list(df.columns)
                    if session['target_column'] is not None:
                        col_lst.remove(session['target_column'])
                    logger.info('Redirect To Delete Columns!')
                    ProjectReports.insert_record_dp('Redirect To Delete Columns!')
                    return render_template('dp/delete_columns.html', columns=list(df.columns), action=action)
                elif action == "duplicate-data":
                    duplicate_data = df[df.duplicated()].head(500)
                    data = duplicate_data.to_html()
                    logger.info('Redirect To Handle Duplicate Data!')
                    ProjectReports.insert_record_dp('Redirect To Handle Duplicate Data!')
                    return render_template('dp/duplicate.html', columns=list(df.columns), action=action, data=data,
                                           duplicate_count=len(duplicate_data))

                elif action == "outlier":
                    logger.info('Redirect To Handler Outlier!')
                    ProjectReports.insert_record_dp('Redirect To Handle Duplicate Data!')
                    columns = Preprocessing.col_seperator(df, 'Numerical_columns')
                    return render_template('dp/outliers.html', columns=columns, action=action)

                elif action == "missing-values":
                    logger.info('Redirect To Missing-Values!')
                    ProjectReports.insert_record_dp('Redirect To Missing-Values!')
                    columns = list(df.columns)
                    return render_template('dp/missing_values.html', columns=columns, action=action)

                elif action == "delete-outlier" or action == "remove-duplicate-data":
                    logger.info('Redirect To Handler Outlier!')
                    ProjectReports.insert_record_dp('Redirect To Handler Outlier!')
                    columns = Preprocessing.col_seperator(df, 'Numerical_columns')
                    return redirect('/dp/outlier')

                elif action == "imbalance-data":

                    logger.info('Redirect To Handle Imbalance Data!')
                    ProjectReports.insert_record_dp('Redirect To Handle Imbalance Data!')

                    if session['project_type'] == 2 and session['target_column'] is None:
                        return redirect('/target-column')

                    if session['project_type'] != 2:
                        return render_template('dp/handle_imbalance.html', error="This section only for classification")

                    target_column = session['target_column']
                    cols_ = [col for col in df.columns if col != target_column]

                    # Check data contain any categorical independent features
                    Categorical_columns = Preprocessing.col_seperator(df.loc[:, cols_], "Categorical_columns")
                    if len(Categorical_columns.columns) > 0:
                        return render_template('dp/handle_imbalance.html', action=action, columns=list(df.columns),
                                               error="Data contain some categorical indepedent features, please perform encoding first")

                    df_counts = pd.DataFrame(df.groupby(target_column).count()).reset_index(level=0)
                    y = list(pd.DataFrame(df.groupby(target_column).count()).reset_index(level=0).columns)[-1]
                    df_counts['Count'] = df_counts[y]
                    graphJSON = PlotlyHelper.barplot(df_counts, x=target_column, y=df_counts['Count'])
                    pie_graphJSON = PlotlyHelper.pieplot(df_counts, names=target_column, values=y, title='')
                    data = {}

                    for (key, val) in zip(df_counts[target_column], df_counts['Count']):
                        data[str(key)] = val

                    columns = list(df.columns)
                    return render_template('dp/handle_imbalance.html',
                                           target_column=target_column, action=action,
                                           pie_graphJSON=pie_graphJSON, graphJSON=graphJSON,
                                           data=data,
                                           perform_action=True)
                else:
                    return render_template('dp/help.html')
            else:
                logger.critical('Data Frame is None')

        else:
            return redirect('/')
    except Exception as e:
        logger.error(e)
        return render_template('500.html', exception=e)


@app_dp.route('/dp/<action>', methods=['POST'])
def data_preprocessing_post(action):
    try:
        if 'pid' in session and 'id' in session:
            df = load_data()
            if df is not None:
                if action == "delete-columns":
                    logger.info('Redirect To Delete Columns!')
                    columns = request.form.getlist('columns')
                    ProjectReports.insert_project_action_report(ProjectActions.DELETE_COLUMN.value, ",".join(columns))
                    df = Preprocessing.delete_col(df, columns)
                    df = update_data(df)
                    return render_template('dp/delete_columns.html', columns=list(df.columns), action=action,
                                           status='success')

                elif action == "duplicate-data":
                    logger.info('Redirect To Handle Duplicate Data!')
                    columns = request.form.getlist('columns')
                    if len(columns) > 0:
                        df = df[df.duplicated(columns)]
                    else:
                        df = df[df.duplicated()]
                    data = df.head(500).to_html()
                    return render_template('dp/duplicate.html', columns=list(df.columns), action=action,
                                           data=data, duplicate_count=len(df), selected_column=','.join(columns))

                elif action == "remove-duplicate-data":
                    logger.info('Redirect To Handle Duplicate Data POST API')
                    columns = request.form['selected_column']

                    if len(columns) > 0:
                        data = df.drop_duplicates(subset=list(columns.split(",")), keep='last')
                    else:
                        data = df.drop_duplicates(keep='last')

                    df = update_data(data)

                    duplicate_data = df[df.duplicated()]
                    data = duplicate_data.head(500).to_html()
                    return render_template('dp/duplicate.html', columns=list(df.columns), action="duplicate-data",
                                           data=data,
                                           duplicate_count=len(duplicate_data), success=True)

                elif action == "outlier":
                    logger.info('Redirected to outlier POST API')

                    method = request.form['method']
                    column = request.form['columns']

                    lower = 25
                    upper = 75
                    if 'lower' in request.form:
                        lower = int(request.form['lower'])

                    if 'upper' in request.form:
                        upper = int(request.form['upper'])

                    graphJSON = ""
                    pie_graphJSON = ""
                    columns = Preprocessing.col_seperator(df, 'Numerical_columns')
                    outliers_list = []
                    logger.info(f'Method {method}')
                    logger.info(f'Columns {column}')
                    if method == "iqr":
                        # lower = request.form['lower']
                        # upper = request.form['upper']
                        result = EDA.outlier_detection_iqr(df.loc[:, [column]], lower, upper)
                        if len(result) > 0:
                            graphJSON = PlotlyHelper.boxplot_single(df, column)
                        data = result.to_html()

                        outliers_list = EDA.outlier_detection(list(df.loc[:, column]), 'iqr')
                        unique_outliers = np.unique(outliers_list)
                    else:
                        result = EDA.z_score_outlier_detection(df.loc[:, [column]])
                        data = result.to_html()

                        outliers_list = EDA.outlier_detection(list(df.loc[:, column]), 'z-score')
                        graphJSON = PlotlyHelper.create_distplot([outliers_list], [column])
                        unique_outliers = np.unique(outliers_list)

                    df_outliers = pd.DataFrame(pd.Series(outliers_list).value_counts(), columns=['value']).reset_index(
                        level=0)
                    if len(df_outliers) > 0:
                        pie_graphJSON = PlotlyHelper.pieplot(df_outliers, names='index', values='value',
                                                             title='Outlier Value Count')

                    logger.info('Sending Data on the front end')
                    return render_template('dp/outliers.html', columns=columns, method=method, selected_column=column,
                                           outliers_list=outliers_list, unique_outliers=unique_outliers,
                                           pie_graphJSON=pie_graphJSON,
                                           lower=lower,
                                           upper=upper,
                                           action=action, data=data,
                                           outliercount=result['Total outliers'][0] if len(
                                               result['Total outliers']) > 0 else 0,
                                           graphJSON=graphJSON)

                elif action == "missing-values":
                    logger.info('Redirect To Missing Values POST API!')
                    if 'method' in request.form:
                        method = request.form['method']
                        selected_column = request.form['selected_column']
                        success = False
                        logger.info(f'Method {method}')
                        logger.info(f'Columns {selected_column}')
                        if method == 'Mean':
                            df[selected_column] = Preprocessing.fill_numerical(df, 'Mean', [selected_column])
                        elif method == 'Median':
                            df[selected_column] = Preprocessing.fill_numerical(df, 'Median', [selected_column])
                        elif method == 'Arbitrary Value':
                            df[selected_column] = Preprocessing.fill_numerical(df, 'Median', [selected_column],
                                                                               request.form['arbitrary'])
                        elif method == 'Interpolate':
                            df[selected_column] = Preprocessing.fill_numerical(df, 'Interpolate', [selected_column],
                                                                               request.form['interpolate'])
                        elif method == 'Mode':
                            df[selected_column] = Preprocessing.fill_categorical(df, 'Mode', selected_column)
                        elif method == 'New Category':
                            df[selected_column] = Preprocessing.fill_categorical(df, 'New Category', selected_column,
                                                                                 request.form['newcategory'])
                        elif method == 'Select Exist':
                            df[selected_column] = Preprocessing.fill_categorical(df, 'New Category', selected_column,
                                                                                 request.form['selectcategory'])

                        df = update_data(df)
                        success = True
                        columns = list(df.columns)
                        # print(selected_column)
                        columns.remove(selected_column)
                        logger.info('Sending Data on Front End')
                        return render_template('dp/missing_values.html', columns=columns, action=action,
                                               selected_column=selected_column, success=success)
                    else:
                        logger.info('Method is not present in request.form')
                        columns = list(df.columns)
                        selected_column = request.form['columns']
                        print(selected_column)
                        data = EDA.missing_cells_table(df.loc[:, [selected_column]])
                        null_value_count = 0
                        unique_category = []
                        outlier_handler_methods = []
                        if len(data) > 0:
                            unique_category = list(df[df[selected_column].notna()][selected_column].unique())
                            null_value_count = data['Missing values'][0]
                            if df[selected_column].dtype == 'object':
                                outlier_handler_methods = OBJECT_MISSING_HANDLER

                            else:
                                outlier_handler_methods = NUMERIC_MISSING_HANDLER

                        data = data.to_html()
                        columns.remove(selected_column)
                        logger.info('Sending Data on Front End')
                        return render_template('dp/missing_values.html', unique_category=unique_category,
                                               columns=columns, selected_column=selected_column, action=action,
                                               data=data, null_value_count=null_value_count,
                                               handler_methods=outlier_handler_methods)



                elif action == "delete-outlier":
                    logger.info('Delete outlier')
                    values = request.form.getlist('columns')
                    selected_column = request.form['selected_column']
                    columns = Preprocessing.col_seperator(df, 'Numerical_columns')
                    list_ = []
                    if df[selected_column].dtype == 'float':
                        list_ = [float(da) for da in list(values)]
                    else:
                        list_ = [int(da) for da in list(values)]

                    df = df[~df[selected_column].isin(list_)]
                    df = update_data(df)
                    logger.info('Sending Data on Front End')
                    return render_template('dp/outliers.html', columns=columns, action="outlier", status="success")

                elif action == "imbalance-data":
                    logger.info('Redirected to Imbalanced Data')
                    try:
                        if 'perform_action' in request.form:
                            target_column = request.form['target_column']
                            method = request.form['method']
                            logger.info(f'{target_column} {method} {range}')

                            class_dict = []

                            for label in list(df[target_column].unique()):
                                class_dict.append([label, int(request.form[str(label)])])

                            if method == 'OS':
                                new_df = Preprocessing.over_sample(df, target_column, class_dict)
                            elif method == 'US':
                                new_df = Preprocessing.under_sample(df, target_column, class_dict)
                            else:
                                new_df = Preprocessing.smote_technique(df, target_column, class_dict)

                            df = update_data(new_df)

                            target_column = session['target_column']
                            df_counts = pd.DataFrame(df.groupby(target_column).count()).reset_index(level=0)
                            y = list(pd.DataFrame(df.groupby(target_column).count()).reset_index(level=0).columns)[-1]
                            df_counts['Count'] = df_counts[y]
                            graphJSON = PlotlyHelper.barplot(df_counts, x=target_column, y=df_counts['Count'])
                            pie_graphJSON = PlotlyHelper.pieplot(df_counts, names=target_column, values=y, title='')
                            data = {}

                            for (key, val) in zip(df_counts[target_column], df_counts['Count']):
                                data[str(key)] = val

                            columns = list(df.columns)
                            return render_template('dp/handle_imbalance.html',
                                                   target_column=target_column, action=action,
                                                   pie_graphJSON=pie_graphJSON, graphJSON=graphJSON,
                                                   data=data, success=True,
                                                   perform_action=True)

                        else:
                            logger.info('perform_action was not found on request form')
                            target_column = request.form['target_column']
                            df_counts = pd.DataFrame(df.groupby(target_column).count()).reset_index(level=0)
                            y = list(pd.DataFrame(df.groupby(target_column).count()).reset_index(level=0).columns)[-1]
                            graphJSON = PlotlyHelper.barplot(df_counts, x=target_column, y=y)
                            pie_graphJSON = PlotlyHelper.pieplot(df_counts, names=target_column, values=y, title='')

                            logger.info('Sending Data on Handle Imbalance page')
                            return render_template('dp/handle_imbalance.html', columns=list(df.columns),
                                                   target_column=target_column, action="imbalance-data",
                                                   pie_graphJSON=pie_graphJSON, graphJSON=graphJSON,
                                                   perform_action=True)

                    except Exception as e:
                        logger.error(e)
                        return render_template('dp/handle_imbalance.html', action=action, columns=list(df.columns),
                                               error=str(e))

                else:
                    return redirect('dp/help.html')
            else:
                logger.critical('DataFrame has no Data')
                return redirect('/')

        else:
            return redirect(url_for('/'))
    except Exception as e:
        logger.error(e)
        return render_template('500.html', exception=e)
