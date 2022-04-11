import pymongo
from src.utils.databases.mysql_helper import MySqlHelper
import pandas as pd
import os
from from_root import from_root
from loguru import logger
from datetime import datetime

logger.remove()
path = os.path.join(from_root(), 'logger', 'logs', 'logs.log')
logger.add(path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")


def get_data():
    mysql = MySqlHelper.get_connection_obj()
    query = "SELECT Pid from tblProjects where pid like 'PID%';"
    pid = mysql.fetch_all(query)
    return pid


def delete_data_from_mongo(projectId=None):
    try:
        CONNECTION_URL = f"mongodb+srv://vishal:123@auto-neuron.euorq.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
        client = pymongo.MongoClient(CONNECTION_URL)
        dataBase = client["Auto-neuron"]
        collection = dataBase[projectId]
        if collection.drop() is None:
            current_data = dataBase.list_collection_names()
            if projectId in current_data:
                return 'Still present inside mongodb'
            else:
                return 'Deleted', dataBase
    except Exception as e:
        logger.error(e.__str__())
        return e.__str__()


def upload_checkpoint(projectId=None, data_path=None):
    try:
        data = pd.read_csv(data_path)
        check, dataBase = delete_data_from_mongo(projectId)
        if check == 'Deleted':
            collection = dataBase[projectId]
            collection.insert_many(data.to_dict('records'))
            return 'SuccessFully Replaced'
        elif check == 'Still present inside mongodb':
            return 'Still present inside mongodb'
        else:
            return 'unidentified Error'
    except Exception as e:
        logger.error(e.__str__())
        return e.__str__()


def get_user_details(projectId=None):
    mysql = MySqlHelper.get_connection_obj()
    query = f"""select Pid,Name,Id,UserId,CreateDate from auto_neuron.tblProjects 
               where UserId = (select UserId from auto_neuron.tblProjects 
               where Pid = '{projectId}' and IsActive = 1)
               """
    result = mysql.fetch_all(query)
    return result


def get_names_from_files(path=None):
    try:
        backup_files = []
        normal_files = []
        result = os.listdir(path)
        for i in result[1:]:
            if i.replace('.csv', '').endswith('_backup'):
                backup_files.append(i.replace('.csv', ''))
            else:
                normal_files.append(i.replace('.csv', ''))

        logger.info(f'Main File Names In Data Folder : {normal_files}')
        logger.info(f'Backup File Names In Data Folder : {backup_files}')
        return backup_files, normal_files
    except Exception as e:
        logger.error(e.__str__())
        return e.__str__()


def file_path(path=None, backup=None, normal=None):
    try:
        backup_data_path = []
        normal_data_path = []

        for i in backup:
            backup_data_path.append(os.path.join(path, i + '.csv'))
        for i in normal:
            normal_data_path.append(os.path.join(path, i + '.csv'))

        return normal_data_path, backup_data_path
    except Exception as e:
        logger.error(e.__str__())
        return e.__str__()


def data_updater(path=os.path.join(from_root(), 'src', 'data')):
    try:
        logger.info(f'Scheduler Initialized at Date : {datetime.now().date}  Time {datetime.now().time}')
        backup, normal = get_names_from_files(path)
        normal_data_path, backup_data_path = file_path(path, backup, normal)

        for pid, data_path in zip(normal, normal_data_path):
            logger.info(f'Project ID {pid} Data Path {data_path}')
            result = upload_checkpoint(pid, data_path)
            logger.info(f'Result {result}')
            if result == 'SuccessFully Replaced':
                os.remove(data_path)
            logger.info('Data Updated in Mongo DB!')

        for pid, data_path in zip(backup, backup_data_path):
            os.remove(data_path)
        logger.info('Backup Files Removed From System!')
    except Exception as e:
        logger.error(e.__str__())
        return e.__str__()
