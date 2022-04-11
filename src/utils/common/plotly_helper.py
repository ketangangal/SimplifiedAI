import json
import plotly
import plotly.express as px
from src.eda.eda_helper import EDA
from loguru import logger
import os
from from_root import from_root
from src.utils.common.common_helper import read_config
import plotly.figure_factory as ff

config_args = read_config("./config.yaml")

log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")


class PlotlyHelper:
    
    @staticmethod
    def create_distplot(hist_data, group_labels):
        try:
            fig = ff.create_distplot(hist_data, group_labels, bin_size=1.0,curve_type="kde")
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("BarPlot Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)
        
    @staticmethod
    def barplot(df, x, y):
        try:
            fig = px.bar(df, x=x, y=y)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("BarPlot Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)

    @staticmethod
    def pieplot(df, names, values, title=''):
        try:
            fig = px.pie(df, names=names, values=values, title=title)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("PiePlot Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)

    @staticmethod
    def scatterplot(df, x, y, title=''):
        try:
            fig = px.scatter(df, x=x, y=y, title=title)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("ScatterPlot Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)

    @staticmethod
    def histogram(df, x, bin=20):
        try:
            fig = px.histogram(df, x=x, nbins=bin)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("Histogram Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)

    @staticmethod
    def line(df, x, y, bin=20):
        try:
            fig = px.line(df, x=x, y=y)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("linePlot Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)

    @staticmethod
    def boxplot(df, x, y):
        try:
            fig = px.box(df, x=x, y=y)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("BoxPlot Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)
            
    @staticmethod
    def boxplot_single(df, x):
        try:
            fig = px.box(df, x=x)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("BoxPlot Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)

    @staticmethod
    def distplot(df, x, y):
        try:
            fig = px.histogram(df, x=x, y=y, marginal='violin', hover_data=df.columns)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("DistPlot Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)

    @staticmethod
    def heatmap(df):
        try:
            pearson_corr = EDA.correlation_report(df, 'pearson')
            fig = px.imshow(pearson_corr)
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            logger.info("Heatmap Implemented!")
            return graphJSON
        except Exception as e:
            logger.error(e)
