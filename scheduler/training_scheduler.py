import os
import pickle
import pymongo
from src.model.custom.classification_models import ClassificationModels
from src.model.custom.regression_models import RegressionModels
from src.model.custom.clustering_models import ClusteringModels
from src.utils.databases.mysql_helper import MySqlHelper
import pandas as pd
from from_root import from_root
from emailSender.Sender import email_sender

mysql = MySqlHelper.get_connection_obj()


def load_model(pid):
    model_path = os.path.join(from_root(), 'artifacts', pid, 'model_temp.pkl')
    model = pickle.load(open(model_path, 'rb'))
    return model


def load_data(pid):
    path = os.path.join(from_root(), 'src', 'data', pid + '.csv')
    if os.path.exists(path):
        print('taking data from local')
        df = pd.read_csv(path)
        return df
    else:
        print('fetching data ')
        CONNECTION_URL = f"mongodb+srv://vishal:123@auto-neuron.euorq.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
        client = pymongo.MongoClient(CONNECTION_URL)
        dataBase = client["Auto-neuron"]
        collection = dataBase[pid]
        df = pd.DataFrame(list(collection.find()))
        df.drop(['_id'], axis=1, inplace=True)
        return df


def save_model(pid, model, name):
    path = os.path.join(from_root(), 'artifacts', pid)
    if not os.path.exists(path):
        os.mkdir(path)
    file_name = os.path.join(path, name)
    pickle.dump(model, open(file_name, 'wb'))


def train_model(model_name=None, target=None, pid=None):
    if target is None:

        X = load_data(pid)

        model_params = {}
        model = load_model(pid)

        for key, value in model.get_params().items():
            model_params[key] = value

        if model_name == "KMeans":
            train_model_fun = ClusteringModels.kmeans_clustering

        elif model_name == "DBSCAN":
            train_model_fun = ClusteringModels.dbscan_clustering

        elif model_name == "AgglomerativeClustering":
            train_model_fun = ClusteringModels.agglomerative_clustering
        else:
            return 'Non Implemented mtd'

        trained_model, y_pred = train_model_fun(X, True, **model_params)

        """Save Trained Model"""
        save_model(pid, trained_model, 'model.pkl')
        query = f'''Update tblProjects Set Model_Trained=1 Where Pid="{pid}"'''
        mysql.update_record(query)
        query = f''' UPDATE tblProject_scheduler set train_status=1 where ProjectId="{pid}"'''
        mysql.update_record(query)

    else:
        df = load_data(pid)
        X = df.drop(target, axis=1)
        y = df[target]

        model_params = {}

        model = load_model(pid)
        for key, value in model.get_params().items():
            model_params[key] = value

        if model_name == "LinearRegression":
            train_model_fun = RegressionModels.linear_regression_regressor
        elif model_name == "Ridge":
            train_model_fun = RegressionModels.ridge_regressor
        elif model_name == "Lasso":
            train_model_fun = RegressionModels.lasso_regressor
        elif model_name == "ElasticNet":
            train_model_fun = RegressionModels.elastic_net_regressor
        elif model_name == "DecisionTreeRegressor":
            train_model_fun = RegressionModels.decision_tree_regressor
        elif model_name == "RandomForestRegressor":
            train_model_fun = RegressionModels.random_forest_regressor
        elif model_name == "SVR":
            train_model_fun = RegressionModels.support_vector_regressor
        elif model_name == "AdaBoostRegressor":
            train_model_fun = RegressionModels.ada_boost_regressor
        elif model_name == "GradientBoostingRegressor":
            train_model_fun = RegressionModels.gradient_boosting_regressor
        elif model_name == "LogisticRegression":
            train_model_fun = ClassificationModels.logistic_regression_classifier
        elif model_name == "SVC":
            train_model_fun = ClassificationModels.support_vector_classifier
        elif model_name == "KNeighborsClassifier":
            train_model_fun = ClassificationModels.k_neighbors_classifier
        elif model_name == "DecisionTreeClassifier":
            train_model_fun = ClassificationModels.decision_tree_classifier
        elif model_name == "RandomForestClassifier":
            train_model_fun = ClassificationModels.random_forest_classifier
        elif model_name == "AdaBoostClassifier":
            train_model_fun = ClassificationModels.ada_boost_classifier
        elif model_name == "GradientBoostClassifier":
            train_model_fun = ClassificationModels.gradient_boosting_classifier
        else:
            return 'Non-Implemented Action'

        trained_model = train_model_fun(X, y, True, **model_params)
        """Save Trained Model"""
        save_model(pid, trained_model, 'model.pkl')

        query = f'''Update tblProjects Set Model_Trained=1 Where Pid="{pid}"'''
        mysql.update_record(query)
        query = f''' UPDATE tbleProject_scheduler set train_status=1 where ProjectId="{pid}"'''
        mysql.update_record(query)


def check_schedule_model():
    # It will on root level
    # sort table data on the basis date and time
    # ascending order
    # start scheduler 1 by 1 after 1 hr
    # train result email
    mysql = MySqlHelper.get_connection_obj()
    query = f""" select a.pid ProjectId , a.TargetColumn TargetName, 
                                a.Model_Name ModelName, 
                                b.datetime_,
                                a.Model_Trained, 
                                b.train_status ,
                                b.email, 
                                b.deleted
                                from tblProjects as a
                               join tblProject_scheduler as b on a.Pid = b.ProjectId
                               where b.datetime_ < NOW() and b.train_status = 0 and a.Model_Trained=0 and deleted = 0
                               order by datetime_"""

    results = mysql.fetch_all(query)
    if len(results) == 0:
        return 'No Task Remaing for scheduling'
    else:
        for process in results:
            train_model(model_name=process[2], target=process[1], pid=process[0])
            email_check = email_sender(process[6], 1)
            if email_check:
                mysql.update_record(f"""DELETE FROM tblProject_scheduler WHERE ProjectID = '{process[0]}'""")

    return 'Done'
