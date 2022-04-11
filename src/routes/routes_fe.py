from flask import Blueprint, request, render_template, session, redirect, url_for
from src.utils.common.data_helper import load_data, update_data
from src.utils.common.project_report_helper import ProjectReports
from src.utils.databases.mysql_helper import MySqlHelper
from src.utils.common.common_helper import save_project_encdoing, save_project_pca, save_project_scaler
from src.utils.common.common_helper import read_config
import os
from loguru import logger
from from_root import from_root
import pandas as pd
import numpy as np
from src.preprocessing.preprocessing_helper import Preprocessing
from src.constants.constants import ENCODING_TYPES, FEATURE_SELECTION_METHODS_CLASSIFICATION, \
    FEATURE_SELECTION_METHODS_CURSOR, FEATURE_SELECTION_METHODS_RGRESSOR, ProjectActions, \
    SUPPORTED_DATA_TYPES, SUPPORTED_SCALING_TYPES
from src.feature_engineering.feature_engineering_helper import FeatureEngineering

mysql = MySqlHelper.get_connection_obj()

config_args = read_config("./config.yaml")

log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")

app_fe = Blueprint('fe', __name__)


@app_fe.route('/fe/<action>', methods=['GET'])
def feature_engineering(action):
    try:
        if 'pid' in session:
            df = load_data()
            if df is not None:
                data = df.head().to_html()
                if action == 'help':
                    return render_template('fe/help.html')
                elif action == 'handle-datatype':
                    columns = list(df.columns)
                    try:
                        logger.info('Redirect To Handle DatType')
                        ProjectReports.insert_record_fe('Redirect To Handle DatType')

                        if session['target_column']:
                            columns = [col for col in columns if col != session['target_column']]

                        return render_template('fe/handle_datatype.html', action=action,
                                               columns=df.loc[:, columns].dtypes.apply(lambda x: x.name).to_dict(),
                                               supported_dtypes=SUPPORTED_DATA_TYPES)
                    except Exception as e:
                        return render_template('fe/handle_datatype.html', supported_dtypes=SUPPORTED_DATA_TYPES,
                                               allowed_operation="not",
                                               columns=df.loc[:, columns].dtypes.apply(lambda x: x.name).to_dict(),
                                               status="error",
                                               msg=f'{e}')

                elif action == 'encoding':

                    logger.info('Redirect To Encoding')
                    ProjectReports.insert_record_fe('Redirect To Encoding')
                    # 1 regression
                    # 2 classification
                    # 3 clustring
                    """Check If Prohect type is Regression or Classificaion and target Columns is not Selected"""
                    # try:
                    if session['project_type'] != 3 and session['target_column'] is None:
                        return redirect('/target-column')
                    # except Exception as e:
                    #     logger.error(f'{e}, Target Column is not selected for Encoding!')
                    """ Check Encoding Already Performed or not"""
                    
                    
                    query_ = f"Select * from tblProject_Actions_Reports  where ProjectId={session['pid']} and ProjectActionId=4"
                    rows = mysql.fetch_all(query_)
                    
                    encoded_columns_list=[]
                    if len(rows) > 0:
                        return render_template('fe/encoding.html', encoding_types=ENCODING_TYPES,
                                               allowed_operation="not",
                                               columns=[], status="error",
                                               msg="You Already Performed Encoding. Don't do this again")
                    return render_template('fe/encoding.html', encoding_types=ENCODING_TYPES, status="success",
                                           columns=list(df.columns[df.dtypes == 'object']), action=action)

                elif action == 'change-column-name':

                    logger.info('Redirect To Change Column Name')
                    ProjectReports.insert_record_fe('Redirect To Change Column Name')
                    return render_template('fe/change_column_name.html', columns=list(df.columns), action=action)

                elif action == 'scaling':

                    logger.info('Redirect To Scaling')
                    ProjectReports.insert_record_fe('Redirect To Scaling')

                    """Check If Prohect type is Regression or Classificaion and target Columns is not Selected"""
                    try:
                        if session['project_type'] != 3 and session['target_column'] is None:
                            return redirect('/target-column')

                        """ Check Scaling Already Performed or not"""
                        query_ = f"Select * from tblProject_Actions_Reports  where ProjectId={session['pid']} and ProjectActionId=5"
                        rows = mysql.fetch_all(query_)

                        if len(rows) > 0:
                            return render_template('fe/scaling.html', scaler_types=SUPPORTED_SCALING_TYPES,
                                                   allowed_operation="not",
                                                   columns=[], status="error",
                                                   msg="You Already Performed Scaling. Don't do this again")
                        columns = df.columns
                        if session['target_column']:
                            columns = [col for col in columns if col != session['target_column']]

                        df = df.loc[:, columns]

                        if len(df.columns[df.dtypes == 'category']) > 0 or len(df.columns[df.dtypes == 'object']) > 0:
                            return render_template('fe/scaling.html', scaler_types=SUPPORTED_SCALING_TYPES,
                                                   allowed_operation="not",
                                                   columns=[], status="error",
                                                   msg="Scaling can't be performed at this point, data contain categorical data. Please perform encoding first")

                        return render_template('fe/scaling.html', scaler_types=SUPPORTED_SCALING_TYPES,
                                               columns=list(df.columns))
                    except Exception as e:
                        logger.error(e)
                        raise Exception(e)

                elif action == 'feature_selection':

                    logger.info('Redirect To Feature Secltion')
                    ProjectReports.insert_record_fe('Redirect To Feature Secltion')
                    methods = []
                
                    if session['project_type'] == 1:
                        methods = FEATURE_SELECTION_METHODS_RGRESSOR
                    elif session['project_type'] == 2:
                        methods = FEATURE_SELECTION_METHODS_CLASSIFICATION
                    elif session['project_type'] == 3:
                        methods = FEATURE_SELECTION_METHODS_CURSOR

                    elif session['target_column'] is None:
                        return redirect('/target-column')

                    target_column = session['target_column']
                    cols_ = [col for col in df.columns if col != target_column]

                    df = df.loc[:, cols_]

                    if len(df.columns[df.dtypes == 'category']) > 0 or len(df.columns[df.dtypes == 'object']) > 0:
                        return render_template('fe/feature_selection.html',
                                               methods=methods,
                                               status="error",
                                               columns_len=df.shape[1] - 1,
                                               msg="Feature Selection can't be performed at this point, data contain categorical data. Please perform encoding first")

                    return render_template('fe/feature_selection.html',
                                           methods=methods,
                                           project_type=session['project_type'],
                                           columns_len=df.shape[1] - 1)

                elif action == 'dimension_reduction':

                    """ Check PCA Already Performed or not"""
                    query_ = f"Select * from tblProject_Actions_Reports  where ProjectId={session['pid']} and ProjectActionId=6"
                    rows = mysql.fetch_all(query_)
                    if len(rows) > 0:
                        return render_template('fe/dimension_reduction.html',
                                               columns=[], status="error",
                                               not_allowed=True,
                                               msg="You Already Performed Dimensionalty Reduction. Don't do this again",data=df.to_html())

                    """ Check Feature Scaling  Performed or not"""
                    query_ = f"Select * from tblProject_Actions_Reports  where ProjectId={session['pid']} and ProjectActionId=5"
                    rows = mysql.fetch_all(query_)
                    if len(rows)==0:
                        return render_template('fe/dimension_reduction.html',
                                               columns=[], status="error",
                                               not_allowed=True,
                                               msg="Please Perform Feature Scaling First")


                    columns = list(df.columns)
                    if session['target_column']:
                        columns = [col for col in columns if col != session['target_column']]

                    df = df.loc[:, columns]

                    logger.info('Redirect To Dimention Reduction')
                    ProjectReports.insert_record_fe('Redirect To Dimention Reduction')
                    # Check this remove target column
                    data = df.head(200).to_html()
                    return render_template('fe/dimension_reduction.html', action=action, data=data)

                elif action == 'train_test_split':
                    return render_template('fe/train_test_split.html', data=data)
                else:
                    return 'Non-Implemented Action'
            else:
                return redirect('/')
        else:
            return redirect('/')
    except Exception as e:
        logger.error(e)
        return render_template('500.html', exception=e)


@app_fe.route('/fe/<action>', methods=['POST'])
def feature_engineering_post(action):
    try:
        if 'pid' in session:
            df = load_data()
            if df is not None:
                data = df.head().to_html()
                if action == 'handle-datatype':
                    try:
                        selected_column = request.form['column']
                        datatype = request.form['datatype']
                        df = FeatureEngineering.change_data_type(df, selected_column, datatype)
                        df = update_data(df)

                        logger.info('Changed Column DataType')
                        ProjectReports.insert_record_fe('Changed Column DataType', selected_column, datatype)
                        ProjectReports.insert_project_action_report(ProjectActions.CHANGE_DATA_TYPE.value,
                                                                    selected_column, datatype)

                        return render_template('fe/handle_datatype.html', status="success", action=action,
                                               columns=df.dtypes.apply(lambda x: x.name).to_dict(),
                                               supported_dtypes=SUPPORTED_DATA_TYPES)

                    except Exception as e:
                        return render_template('fe/handle_datatype.html', status="error", action=action,
                                               columns=df.dtypes.apply(lambda x: x.name).to_dict(),
                                               supported_dtypes=SUPPORTED_DATA_TYPES)
                elif action == 'change-column-name':
                    try:
                        selected_column = request.form['selected_column']
                        column_name = request.form['column_name']
                        df = FeatureEngineering.change_column_name(df, selected_column, column_name.strip())
                        df = update_data(df)

                        logger.info('Changed Column Name')
                        ProjectReports.insert_record_fe('Changed Column DataType', selected_column, column_name)
                        ProjectReports.insert_project_action_report(ProjectActions.COLUMN_NAME_CHANGE.value,
                                                                    selected_column, column_name.strip())

                        return render_template('fe/change_column_name.html', status="success", columns=list(df.columns),
                                               action=action)
                    except Exception as e:

                        logger.info('Changed Column Name')
                        ProjectReports.insert_record_fe('Changed Column DataType', selected_column, column_name, 0)

                        return render_template('fe/change_column_name.html', status="error", columns=list(df.columns),
                                               action=action)
                elif action == 'encoding':
                    try:

                        encoding_type = request.form['encoding_type']
                        columns = request.form.getlist('columns')


                        cat_data = df.loc[:, columns]
                        encoder_ = None
                        non_encoded_columns=[col for col in df.columns if col not in columns]

                        if encoding_type == "Base N Encoder":
                            (df_, encoder_) = FeatureEngineering.encodings(cat_data, cat_data.columns, encoding_type,
                                                                           base=int(request.form['base']))
                        elif encoding_type == "Target Encoder":
                            (df_, encoder_) = FeatureEngineering.encodings(cat_data, cat_data.columns, encoding_type,
                                                                           n_components=request.form['target'])
                        elif encoding_type == "Hash Encoder":
                            """This is remaining to handle"""
                            (df_, encoder_) = FeatureEngineering.encodings(cat_data, cat_data.columns, encoding_type,
                                                                           n_components=int(request.form['hash']))
                        else:
                            (df_, encoder_) = FeatureEngineering.encodings(cat_data, cat_data.columns, encoding_type)

                        rem_data=cat_data.loc[:,non_encoded_columns]
                        df = pd.concat([df_, rem_data], axis=1)
                        df = update_data(df)

                        save_project_encdoing(encoder_)

                        ProjectReports.insert_record_fe('Perform Encoding', encoding_type, '')
                        ProjectReports.insert_project_action_report(ProjectActions.ENCODING.value,
                                                                    ",".join(list(columns)))

                        logger.info(f'Perform Encoding:{encoding_type}')

                        return redirect('/eda/show')
                    except Exception as e:

                        logger.info(f'Perform Encoding:{encoding_type}')
                        ProjectReports.insert_record_fe('Perform Encoding', encoding_type, str(e), 0)

                        return render_template('fe/encoding.html', status="error", encoding_types=ENCODING_TYPES,
                                               msg="This encdoing can't performed, please select other method.",
                                               columns=list(df.columns[df.dtypes == 'object']), action=action)

                elif action == 'scaling':
                    try:
                        scaling_method = request.form['scaling_method']
                        columns = df.columns
                        if len(columns) <= 0:
                            raise Exception("Column can not be zero")

                        if session['target_column']:
                            columns = [col for col in columns if col != session['target_column']]

                        df[columns], scaler = FeatureEngineering.scaler(df[columns], scaling_method)
                        df = update_data(df)

                        save_project_scaler(scaler)
                        ProjectReports.insert_record_fe('Perform Scaling')
                        ProjectReports.insert_project_action_report(ProjectActions.SCALING.value)

                        return redirect('/eda/show')
                    except:
                        return render_template('fe/scaling.html', status="error", scaler_types=SUPPORTED_SCALING_TYPES,
                                               columns=list(df.columns[df.dtypes != 'object']))
                elif action == 'feature_selection':
                    return render_template('fe/feature_selection.html', data=data)
                elif action == 'dimension_reduction':
                    # Check this remove target column
                    try:
                        columns = list(df.columns)
                        if session['target_column']:
                            columns = [col for col in columns if col != session['target_column']]

                        df_ = df.loc[:, columns]
                        no_pca_selected = request.form['range']
                        df_, evr_, pca = FeatureEngineering.dimenstion_reduction(df_, len(df_.columns))
                        save_project_pca(pca)
                        ProjectReports.insert_record_fe('Dimention Reduction')
                        ProjectReports.insert_project_action_report(ProjectActions.PCA.value, no_pca_selected)
                        df_ = df_[:, :int(no_pca_selected)]
                        df_evr = pd.DataFrame()
                        data = pd.DataFrame(df_, columns=[f"Col_{col + 1}" for col in np.arange(0, df_.shape[1])])

                        if session['target_column']:
                            data[session['target_column']] = df.loc[:, session['target_column']]

                        df = update_data(data)
                        data = df.head(200).to_html()
                        return render_template('fe/dimension_reduction.html', status="success", action=action,
                                               data=data)
                    except Exception as e:
                        logger.error(e)
                        return render_template('fe/dimension_reduction.html', status="error", action=action, data=data)
                else:
                    return 'Non-Implemented Action'
            else:
                return redirect('/')
        else:
            return redirect(url_for('/'))

    except Exception as e:
        logger.error(e)
        return render_template('500.html', exception=e)
