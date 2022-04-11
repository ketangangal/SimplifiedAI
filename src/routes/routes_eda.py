from flask import Blueprint, request, render_template, session, redirect, url_for
from flask.wrappers import Response
from loguru import logger
from src.utils.common.data_helper import load_data
from src.utils.common.plotly_helper import PlotlyHelper
from src.utils.common.project_report_helper import ProjectReports
import numpy as np
from src.eda.eda_helper import EDA
from pandas_profiling import ProfileReport
from src.constants.constants import TWO_D_GRAPH_TYPES, TWO_D_GRAPH_TYPES_2
import plotly.figure_factory as ff
import json
import plotly
from src.utils.common.common_helper import immutable_multi_dict_to_str, get_numeric_categorical_columns
import os
from from_root import from_root
import pandas as pd

app_eda = Blueprint('eda', __name__)


@app_eda.route('/eda/<action>')
def eda(action):
    try:
        if 'pid' in session:
            df = load_data()
            if df is not None:
                if action == "data-summary":
                    ProjectReports.insert_record_eda('Redirect To Data Summary')
                    summary = EDA.five_point_summary(df)
                    data = summary.to_html()
                    dtypes = EDA.data_dtype_info(df)
                    return render_template('eda/5point.html', data=data, dtypes=dtypes.to_html(), count=len(df),
                                           column_count=df.shape[1])
                # elif action == "profiler":
                #     ProjectReports.insert_record_eda('Redirect To Profile Report')
                #     return render_template('eda/profiler.html', action=action)

                elif action == "show":
                    ProjectReports.insert_record_eda('Redirect To Show Dataset')
                    data = EDA.get_no_records(df, 100)
                    data = data.to_html()
                    topselected = True
                    bottomSelected = False
                    selectedCount = 100
                    return render_template('eda/showdataset.html', data=data, length=len(df),
                                           bottomSelected=bottomSelected, topselected=topselected, action=action,
                                           selectedCount=selectedCount, columns=df.columns)
                elif action == "missing":
                    ProjectReports.insert_record_eda('Redirect To Missing Value')
                    df = EDA.missing_cells_table(df)

                    if df is not None:

                        graphJSON = PlotlyHelper.barplot(df, x='Column', y='Missing values')
                        pie_graphJSON = PlotlyHelper.pieplot(df, names='Column', values='Missing values',
                                                             title='Missing Values')

                        data = df.drop('Column', axis=1)
                        data = data.to_html()
                        return render_template('eda/missing_values.html', action=action, data=data, barplot=graphJSON,
                                               pieplot=pie_graphJSON, contain_missing=True)
                    else:
                        return render_template('eda/missing_values.html', action=action, contain_missing=False)

                elif action == "outlier":
                    ProjectReports.insert_record_eda('Redirect To Outlier')
                    df = EDA.z_score_outlier_detection(df)
                    graphJSON = PlotlyHelper.barplot(df, x='Features', y='Total outliers')
                    pie_graphJSON = PlotlyHelper.pieplot(
                        df.sort_values(by='Total outliers', ascending=False).loc[: 10 if len(df) > 10 else len(df)-1, :],
                        names='Features', values='Total outliers', title='Top 10 Outliers')
                    data = df.to_html()
                    return render_template('eda/outliers.html', data=data, method='zscore', action=action,
                                           barplot=graphJSON, pieplot=pie_graphJSON)

                elif action == "correlation":
                    ProjectReports.insert_record_eda('Redirect To Correlation')
                    pearson_corr = EDA.correlation_report(df, 'pearson')
                    persion_data = list(np.around(np.array(pearson_corr.values), 2))
                    fig = ff.create_annotated_heatmap(persion_data, x=list(pearson_corr.columns),
                                                      y=list(pearson_corr.columns), colorscale='Viridis')
                    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    return render_template('eda/correlation.html', data=graphJSON, columns=list(pearson_corr.columns),
                                           action=action, method='pearson')

                elif action == "plots":
                    ProjectReports.insert_record_eda('Plots')
                    num_cols, cat_cols = get_numeric_categorical_columns(df)
                    if len(cat_cols) == 0:
                        graph_type_list = TWO_D_GRAPH_TYPES_2
                    else:
                        graph_type_list = TWO_D_GRAPH_TYPES

                    return render_template('eda/plots.html', columns=list(df.columns), x_list=list(df.columns),
                                           y_list=num_cols,
                                           graphs_2d=graph_type_list, action=action, x_column="", y_column="")
                else:
                    return render_template('eda/help.html')
            else:
                return redirect('/')

        else:
            return redirect(url_for('/'))
    except Exception as e:
        ProjectReports.insert_record_eda(e)
        logger.error(e)
        return render_template('500.html', exception=e)


@app_eda.route('/eda/<action>', methods=['POST'])
def eda_post(action):
    try:
        if 'pid' in session:
            df = load_data()
            if df is not None:
                graphJSON = None
                if action == "show":
                    range = request.form['range']
                    optradio = request.form['optradio']
                    columns_for_list = df.columns
                    columns = request.form.getlist('columns')
                    input_str = immutable_multi_dict_to_str(request.form)
                    ProjectReports.insert_record_eda('Show', input=input_str)

                    if len(columns) > 0:
                        df = df.loc[:, columns]

                    data = EDA.get_no_records(df, int(range), optradio)
                    data = data.to_html()
                    topselected = True if optradio == 'top' else False
                    bottomSelected = True if optradio == 'bottom' else False
                    return render_template('eda/showdataset.html', data=data, length=len(df),
                                           bottomSelected=bottomSelected, topselected=topselected, action=action,
                                           selectedCount=range, columns=columns_for_list)
                # elif action == "profiler":
                #     ProjectReports.insert_record_eda('Download  Profile Report')
                #
                #     pr = ProfileReport(df, explorative=True, minimal=True,
                #                        correlations={"cramers": {"calculate": False}})
                #
                #     report_path = os.path.join(from_root(), "artifacts", f"{session.get('id')}_report.html")
                #     pr.to_file(report_path)
                #     with open(report_path) as fp:
                #         content = fp.read()
                #
                #     return Response(
                #         content,
                #         mimetype="text/csv",
                #         headers={"Content-disposition": "attachment; filename=report.html"})

                elif action == "correlation":
                    method = request.form['method']
                    columns = request.form.getlist('columns')

                    input_str = immutable_multi_dict_to_str(request.form, True)
                    ProjectReports.insert_record_eda('Redirect To Correlation', input=input_str)

                    if method is not None:
                        # df=df.loc[:,columns]
                        _corr = EDA.correlation_report(df, method)
                        if len(columns) == 0:
                            columns = _corr.columns

                        _corr = _corr.loc[:, columns]
                        _data = list(np.around(np.array(_corr.values), 2))
                        fig = ff.create_annotated_heatmap(_data, x=list(_corr.columns),
                                                          y=list(_corr.index), colorscale='Viridis')
                        # fig = ff.create_annotated_heatmap(_data, colorscale='Viridis')
                        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                        return render_template('eda/correlation.html', data=graphJSON,
                                               columns=list(df.select_dtypes(exclude='object').columns), action=action,
                                               method=method)
                    else:
                        return render_template('eda/help.html')

                elif action == "outlier":
                    method = request.form['method']
                    print(method)
                    lower = 25
                    upper = 75
                    if method == "iqr":
                        lower = request.form['lower']
                        upper = request.form['upper']
                        df = EDA.outlier_detection_iqr(df, int(lower), int(upper))
                        print(df)
                    else:
                        df = EDA.z_score_outlier_detection(df)
                        print('missed')

                    input_str = immutable_multi_dict_to_str(request.form, True)
                    ProjectReports.insert_record_eda('Redirect To Outlier', input=input_str)

                    graphJSON = PlotlyHelper.barplot(df, x='Features', y='Total outliers')

                    pie_graphJSON = PlotlyHelper.pieplot(
                        df.sort_values(by='Total outliers', ascending=False).loc[: 10 if len(df) > 10 else len(df)-1,:],
                        names='Features', values='Total outliers', title='Top 10 Outliers')

                    data = df.to_html()
                    return render_template('eda/outliers.html', data=data, method=method, action=action, lower=lower,
                                           upper=upper, barplot=graphJSON, pieplot=pie_graphJSON)

                elif action == "plots":
                    """All Polots for all kind of features????"""
                    selected_graph_type = request.form['graph']

                    input_str = immutable_multi_dict_to_str(request.form)
                    ProjectReports.insert_record_eda('Plot', input=input_str)
                    num_cols, cat_cols = get_numeric_categorical_columns(df)
                    if len(cat_cols) == 0:
                        graph_type_list = TWO_D_GRAPH_TYPES_2
                    else:
                        graph_type_list = TWO_D_GRAPH_TYPES

                    if selected_graph_type == "Scatter Plot":
                        x_column = request.form['xcolumn']
                        y_column = request.form['ycolumn']
                        graphJSON = PlotlyHelper.scatterplot(df, x=x_column, y=y_column, title='Scatter Plot')

                    elif selected_graph_type == "Pie Chart":

                        x_column = request.form['xcolumn']
                        new_df = df.groupby(x_column).count()
                        temp_df = pd.DataFrame()

                        temp_df[x_column] = list(new_df.index)
                        temp_df['Count'] = list(new_df.iloc[:, 0])

                        graphJSON = PlotlyHelper.pieplot(temp_df, names=x_column, values='Count', title='Pie Chart')

                    elif selected_graph_type == "Bar Graph":
                        x_column = request.form['xcolumn']
                        new_df = df.groupby(x_column).count()
                        temp_df = pd.DataFrame()

                        temp_df[x_column] = list(new_df.index)
                        temp_df['Count'] = list(new_df.iloc[:, 0])

                        graphJSON = PlotlyHelper.barplot(temp_df, x=x_column, y='Count')

                    elif selected_graph_type == "Histogram":
                        x_column = request.form['xcolumn']
                        graphJSON = PlotlyHelper.histogram(df, x=x_column)

                    elif selected_graph_type == "Line Chart":
                        x_column = request.form['xcolumn']
                        y_column = request.form['ycolumn']
                        graphJSON = PlotlyHelper.line(df, x=x_column, y=y_column)

                    elif selected_graph_type == "Box Plot":
                        x_column = request.form['xcolumn']
                        y_column = request.form['ycolumn']
                        graphJSON = PlotlyHelper.boxplot(df, x=x_column, y=y_column)

                    elif selected_graph_type == "Dist Plot":
                        x_column = request.form['xcolumn']
                        y_column = request.form['ycolumn']
                        hist_data = []
                        category_list = list(df[y_column].unique())
                        for category in category_list:
                            hist_data.append(list(df[df[y_column] == category][x_column]))

                        graphJSON = PlotlyHelper.create_distplot(hist_data, category_list)

                    elif selected_graph_type == "Heat Map":
                        graphJSON = PlotlyHelper.heatmap(df)

                    return render_template('eda/plots.html', selected_graph_type=selected_graph_type,
                                           columns=list(df.columns), graphs_2d=graph_type_list,
                                           action=action, graphJSON=graphJSON)
                else:
                    return render_template('eda/help.html')
            else:
                """Manage This"""
                pass

        else:
            return redirect(url_for('/'))
    except Exception as e:
        ProjectReports.insert_record_eda(e)
        return render_template('500.html', exception=e)


@app_eda.route('/x_y_columns', methods=['GET', 'POST'])
def x_y_columns():
    try:
        if 'pid' in session:
            graph_selected = request.args.get('graph_selected')
            df = load_data()
            if df is not None:
                num_cols, cat_cols = get_numeric_categorical_columns(df)
                if graph_selected == "Bar Graph":
                    return render_template('eda/x_y_columns.html', x_list=list(cat_cols),
                                           graph_selected=graph_selected)
                elif graph_selected == "Histogram":
                    return render_template('eda/x_y_columns.html', x_list=list(df.columns), y_list=[],
                                           graph_selected=graph_selected)
                elif graph_selected == "Scatter Plot":
                    return render_template('eda/x_y_columns.html', x_list=list(num_cols), y_list=list(num_cols),
                                           graph_selected=graph_selected)
                elif graph_selected == "Pie Chart":
                    return render_template('eda/x_y_columns.html', x_list=list(cat_cols),
                                           graph_selected=graph_selected)
                elif graph_selected == "Line Chart":
                    return render_template('eda/x_y_columns.html', x_list=list(num_cols), y_list=list(num_cols),
                                           graph_selected=graph_selected)
                elif graph_selected == "Box Plot":
                    return render_template('eda/x_y_columns.html', x_list=list(cat_cols), y_list=list(num_cols),
                                           graph_selected=graph_selected)
                elif graph_selected == "Dist Plot":
                    return render_template('eda/x_y_columns.html', x_list=list(num_cols), y_list=list(cat_cols),
                                           graph_selected=graph_selected)
                elif graph_selected == "Heat Map":
                    return render_template('eda/x_y_columns.html', graph_selected=graph_selected)
                else:
                    return redirect(url_for('/eda/help'))
            else:
                """Manage This"""
                pass
        else:
            return redirect(url_for('/'))
    except Exception as e:
        ProjectReports.insert_record_eda(e)
        return render_template('500.html', exception=e)
