from flask import Flask, redirect, url_for, render_template, request, session, send_from_directory, flash
from werkzeug.wrappers import Response
import re
from scheduler.scheduler import data_updater
from src.constants.constants import PROJECT_TYPES, ProjectActions
from src.utils.common.plotly_helper import PlotlyHelper
from src.utils.databases.mysql_helper import MySqlHelper
from werkzeug.utils import secure_filename
import os
from scheduler.training_scheduler import check_schedule_model
from src.utils.common.common_helper import decrypt, read_config, unique_id_generator, Hashing, encrypt, \
    remove_temp_files
from src.utils.databases.mongo_helper import MongoHelper
from src.constants.constants import ALL_MODELS, TIMEZONE
from src.utils.common.data_helper import load_data, update_data
from src.utils.common.cloud_helper import aws_s3_helper
from src.utils.common.cloud_helper import gcp_browser_storage
from src.utils.common.cloud_helper import azure_data_helper
from src.utils.common.database_helper import mysql_data_helper, mongo_data_helper
from src.utils.common.database_helper import cassandra_connector
from src.utils.common.project_report_helper import ProjectReports
from src.routes.routes_api import app_api
from loguru import logger
from src.routes.routes_eda import app_eda
from src.routes.routes_dp import app_dp
from src.routes.routes_fe import app_fe
from src.routes.routes_training import app_training
from from_root import from_root
import numpy as np
import pandas as pd
import zipfile
import pathlib
import io
import time
import atexit
from apscheduler.schedulers.background import BackgroundScheduler

# Yaml Config File
config_args = read_config("./config.yaml")
log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])

logger.remove()

logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")
logger.info('Fetching Data from configuration file')

# SQL Connection code
host = config_args['secrets']['host']
port = config_args['secrets']['port']
user = config_args['secrets']['user']
password = config_args['secrets']['password']
database = config_args['secrets']['database']

# Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=data_updater, trigger="interval", hours=24)
scheduler.add_job(func=check_schedule_model, trigger="interval", hours=1)
scheduler.start()
#
# # Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

# DataBase Initilazation
logger.info('Initializing Databases')
mysql = MySqlHelper.get_connection_obj()
mongodb = MongoHelper()

template_dir = config_args['dir_structure']['template_dir']
static_dir = config_args['dir_structure']['static_dir']

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)
logger.info('App Started')

# Routes (API,EDA,DP,FE,Training)
app.register_blueprint(app_api)
app.register_blueprint(app_eda)
app.register_blueprint(app_dp)
app.register_blueprint(app_fe)
app.register_blueprint(app_training)

app.secret_key = config_args['secrets']['key']
app.config["UPLOAD_FOLDER"] = config_args['dir_structure']['upload_folder']
app.config["MAX_CONTENT_PATH"] = config_args['secrets']['MAX_CONTENT_PATH']


@app.context_processor
def context_processor():
    loggedin = False
    if 'loggedin' in session:
        loggedin = True
    return dict(loggedin=loggedin)


@app.route('/contact', methods=['GET'], )
def contact():
    try:
        developers = [{
            'id': "two",
            'name': 'Pankaj Malhan',
            'src': 'dev1.jfif',
            'desc': 'DEVELOPMENT CONTRIBUTER',
            'twitter': 'https://twitter.com/pankajmalhan30',
            'linkedin': 'https://www.linkedin.com/in/pankaj-kumar-choudhary-a2b930a8/',
            'github': 'https://github.com/pankajmalhan'
        }, {
            'id': "two",
            'name': 'Ketan Gangal',
            'src': 'dev2.jfif',
            'desc': 'DEVELOPMENT CONTRIBUTER',
            'twitter': 'https://twitter.com/ketan_gangal',
            'linkedin': 'https://www.linkedin.com/in/ketan-gangal/',
            'github': 'https://github.com/ketangangal'
        }, {
            'id': "two",
            'name': 'Vishal Singh',
            'src': 'dev3.jpg',
            'desc': 'DEVELOPMENT CONTRIBUTER',
            'twitter': 'https://twitter.com/17VishalSingh',
            'linkedin': 'https://www.linkedin.com/in/vishalsingh1719/',
            'github': 'https://github.com/vishalsingh17'
        }, {
            'id': "two",
            'name': 'Supreeth Shetty',
            'src': 'pp.jpg',
            'desc': 'DEVELOPMENT CONTRIBUTER',
            'twitter': 'https://twitter.com/Supreet09657830',
            'linkedin': 'https://www.linkedin.com/in/supreeth-s-shetty-302268170/',
            'github': 'https://github.com/Supreeth-Shetty'
        }, {
            'id': "two",
            'name': 'Anshu Narayan',
            'src': 'dev5.jpg',
            'desc': 'DEVELOPMENT CONTRIBUTER',
            'twitter': 'https://twitter.com/narayan_anshu',
            'linkedin': 'www.linkedin.com/in/anshu-narayan-36235791',
            'github': "https://github.com/anshyan"
        }, {
            'id': "two",
            'name': 'Rohan Bagulwar',
            'src': 'rohan.jpg',
            'desc': 'DEVELOPMENT CONTRIBUTER',
            'twitter': 'https://www.linkedin.com/in/rohan-bagulwar',
            'linkedin': 'https://www.linkedin.com/in/rohan-bagulwar',
            'github': "https://github.com/Rohanbagulwar"
        }]
        return render_template('contact.html', developers=developers)
    except Exception as e:
        logger.error(e)
        return render_template('500.html', exception=e)


# Index Route
@app.route('/', methods=['GET', 'POST'], )
def index():
    try:
        if 'loggedin' in session:
            query = f'''
            select tp.Id,tp.Name,tp.Description,tp.Cassandra_Table_Name,
            (Select ts.Name from tblProjectReports join tblProjectStatus as ts on ts.Id=ModuleId   Where ProjectId=tp.Id  LIMIT 1) as ModuleName,
            ts.Indetifier,tp.Pid,tp.TargetColumn,tpy.Name
            from tblProjects as tp
            join tblProjectType as tpy
                on tpy.Id=tp.ProjecTtype
            join tblProjectStatus as ts
                on ts.Id=tp.Status
            where tp.UserId={session.get('id')} and tp.IsActive=1
            order by 1 desc'''

            projects = mysql.fetch_all(query)
            project_lists = []

            for project in projects:
                projectid = encrypt(f"{project[6]}&{project[0]}").decode("utf-8")
                project_lists.append(project + (projectid,))

            logger.info("Project Initilazation Completed")
            return render_template('index.html', projects=project_lists)
        else:
            return redirect(url_for('contact'))
    except Exception as e:
        logger.error(e)
        return render_template('500.html', exception=e)


@app.route('/project', methods=['GET', 'POST'])
def project():
    try:
        if 'loggedin' in session:
            download_status = None
            file_path = None
            if request.method == "GET":
                return render_template('new_project.html', loggedin=True, project_types=PROJECT_TYPES)
            else:
                source_type = request.form['source_type']
                f = None
                ALLOWED_EXTENSIONS = ['csv', 'tsv', 'json', 'xlsx']

                if source_type == 'uploadFile':
                    name = request.form['project_name']
                    description = request.form['project_desc']
                    project_type = request.form['project_type']
                    logger.info(source_type, name, description)
                    if len(request.files) > 0:
                        f = request.files['file']

                    message = ''
                    if not name.strip():
                        message = 'Please enter project name'
                    elif not description.strip():
                        message = 'Please enter project description'
                    elif f.filename.strip() == '':
                        message = 'Please select a file to upload'
                    elif f.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                        message = 'This file format is not allowed, please select mentioned one'

                    if message:
                        logger.info(message)
                        return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)

                    filename = secure_filename(f.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    f.save(file_path)
                    timestamp = round(time.time() * 1000)
                    name = name.replace(" ", "_")
                    table_name = f"{name}_{timestamp}"

                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif file_path.endswith('.tsv'):
                        df = pd.read_csv(file_path, sep='\t')
                    elif file_path.endswith('.json'):
                        df = pd.read_json(file_path)
                    elif file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                    else:
                        message = 'This file format is currently not supported'
                        logger.info(message)
                        return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)

                    remove_temp_files([file_path])
                    project_id = unique_id_generator()
                    logger.info(f'Pushing {filename} to mongodb')
                    inserted_rows = mongodb.create_new_project(project_id, df)

                    if inserted_rows > 0:
                        userId = session.get('id')
                        status = 1
                        query = f"""INSERT INTO tblProjects (UserId, Name, Description, Status, 
                       Cassandra_Table_Name,Pid,ProjectType) VALUES
                       ("{userId}", "{name}", "{description}", "1", "{table_name}","{project_id}","{project_type}")"""

                        rowcount = mysql.insert_record(query)
                        if rowcount > 0:
                            logger.info('Project Created!!')
                            flash("Success!!")
                            return redirect(url_for('index'))
                        else:
                            message = "Error while creating new Project"
                            logger.info(message)
                            return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)
                    else:
                        message = "Error while creating new Project"
                        logger.info(message)
                        return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)

                elif source_type == 'uploadResource':
                    name = request.form['project_name']
                    description = request.form['project_desc']
                    resource_type = request.form['resource_type']

                    if not name.strip():
                        message = 'Please enter project name'
                        logger.info(message)
                        return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)
                    elif not description.strip():
                        message = 'Please enter project description'
                        logger.info(message)
                        return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)

                    if resource_type == "awsS3bucket":
                        region_name = request.form['region_name']
                        aws_access_key_id = request.form['aws_access_key_id']
                        aws_secret_access_key = request.form['aws_secret_access_key']
                        bucket_name = request.form['bucket_name']
                        file_name = request.form['file_name']
                        if file_name.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                            message = 'This file format is not allowed, please select mentioned one'
                            logger.info(message)
                            return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)

                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                        aws_s3 = aws_s3_helper(region_name, aws_access_key_id, aws_secret_access_key)
                        logger.info("Validating User's AWS Credentials!!")
                        conn_msg = aws_s3.check_connection(bucket_name, file_name)
                        if conn_msg != 'Successful':
                            logger.info("AWS Connection Not Successful")
                            return render_template('new_project.html', msg=conn_msg, project_types=PROJECT_TYPES)
                        logger.info("AWS Connection Successful!!")
                        download_status = aws_s3.download_file_from_s3(bucket_name, file_name, file_path)
                        logger.info(download_status)

                    elif resource_type == "gcpStorage":
                        credentials_file = request.files['GCP_credentials_file']
                        bucket_name = request.form['bucket_name']
                        file_name = request.form['file_name']
                        if file_name.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                            message = 'This file format is not allowed, please select mentioned one'
                            return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)

                        credentials_filename = secure_filename(credentials_file.filename)
                        credentials_file_path = os.path.join(app.config['UPLOAD_FOLDER'], credentials_filename)
                        credentials_file.save(credentials_file_path)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                        logger.info(credentials_file_path, file_path, file_name, bucket_name)
                        gcp = gcp_browser_storage(credentials_file_path)
                        logger.info("Validating User's GCP Credentials!!")
                        conn_msg = gcp.check_connection(bucket_name, file_name)
                        remove_temp_files([credentials_file_path])
                        logger.info(conn_msg)
                        if conn_msg != 'Successful':
                            logger.info("GCP Connection Not Successful")
                            return render_template('new_project.html', msg=conn_msg, project_types=PROJECT_TYPES)
                        logger.info("GCP Connection Successful")
                        download_status = gcp.download_file_from_bucket(file_name, file_path, bucket_name)
                        logger.info(download_status)

                    elif resource_type == "mySql":
                        host = request.form['host']
                        port = request.form['port']
                        user = request.form['user']
                        password = request.form['password']
                        database = request.form['database']
                        table_name = request.form['table_name']
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], (table_name + ".csv"))

                        mysql_data = mysql_data_helper(host, port, user, password, database)
                        logger.info("Validating User's Mysql Credentials!!")
                        conn_msg = mysql_data.check_connection(table_name)
                        if conn_msg != 'Successful':
                            logger.info("User's Msql Connection Not Successful")
                            return render_template('new_project.html', msg=conn_msg, project_types=PROJECT_TYPES)
                        logger.info("User's Msql Connection Successful")
                        download_status = mysql_data.retrive_dataset_from_table(table_name, file_path)
                        logger.info(download_status)

                    elif resource_type == "cassandra":
                        secure_connect_bundle = request.files['secure_connect_bundle']
                        client_id = request.form['client_id']
                        client_secret = request.form['client_secret']
                        keyspace = request.form['keyspace']
                        table_name = request.form['table_name']
                        data_in_tabular = request.form['data_in_tabular']
                        secure_connect_bundle_filename = secure_filename(secure_connect_bundle.filename)
                        secure_connect_bundle_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                                                       secure_connect_bundle_filename)
                        secure_connect_bundle.save(secure_connect_bundle_file_path)
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], (table_name + ".csv"))
                        cassandra_db = cassandra_connector(secure_connect_bundle_file_path,
                                                           client_id, client_secret, keyspace)
                        logger.info("Validating User's Cassandra Credentials!!")
                        conn_msg = cassandra_db.check_connection(table_name)
                        remove_temp_files([secure_connect_bundle_file_path])
                        if conn_msg != 'Successful':
                            logger.info("User's Cassandra Connection Not Successful")
                            return render_template('new_project.html', msg=conn_msg, project_types=PROJECT_TYPES)

                        logger.info("User's Cassandra Connection Successful")
                        if data_in_tabular == 'true':
                            download_status = cassandra_db.retrive_table(table_name, file_path)
                            logger.info(download_status)
                        elif data_in_tabular == 'false':
                            download_status = cassandra_db.retrive_uploded_dataset(table_name, file_path)
                            logger.info(download_status)

                    elif resource_type == "mongodb":
                        mongo_db_url = request.form['mongo_db_url']
                        mongo_database = request.form['mongo_database']
                        collection = request.form['collection']
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], (collection + ".csv"))
                        mongo_helper = mongo_data_helper(mongo_db_url)
                        logger.info("Validating User's MongoDB Credentials!!")
                        conn_msg = mongo_helper.check_connection(mongo_database, collection)
                        if conn_msg != 'Successful':
                            logger.info("User's MongoDB Connection Not Successful")
                            return render_template('new_project.html', msg=conn_msg, project_types=PROJECT_TYPES)
                        logger.info("User's MongoDB Connection Successful")
                        download_status = mongo_helper.retrive_dataset(mongo_database, collection, file_path)
                        logger.info(download_status)

                    elif resource_type == "azureStorage":
                        azure_connection_string = request.form['azure_connection_string']
                        container_name = request.form['container_name']
                        file_name = request.form['file_name']
                        if file_name.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
                            message = 'This file format is not allowed, please select mentioned one'
                            return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)

                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                        azure_helper = azure_data_helper(azure_connection_string)
                        conn_msg = azure_helper.check_connection(container_name, file_name)
                        logger.info("Validating User's Azure Credentials!!")
                        if conn_msg != 'Successful':
                            logger.info("User's Azure Connection Not Successful")
                            return render_template('new_project.html', msg=conn_msg)

                        logger.info("User's Azure Connection Successful")
                        download_status = azure_helper.download_file(container_name, file_name, file_path)
                        logger.info(download_status)

                    else:
                        return render_template('new_project.html', msg="Select Any Various Resource Type!!",
                                               project_types=PROJECT_TYPES)

                    if download_status == 'Successful':
                        timestamp = round(time.time() * 1000)
                        name = name.replace(" ", "_")
                        table_name = f"{name}_{timestamp}"

                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        elif file_path.endswith('.tsv'):
                            df = pd.read_csv(file_path, sep='\t')
                        elif file_path.endswith('.json'):
                            df = pd.read_json(file_path)
                        elif file_path.endswith('.xlsx'):
                            df = pd.read_excel(file_path)
                        else:
                            msg = 'This file format is currently not supported'
                            logger.info(msg)
                            return render_template('new_project.html', msg=msg)

                        remove_temp_files([file_path])
                        project_id = unique_id_generator()
                        logger.info(f'Pushing user dataset to mongodb')
                        inserted_rows = mongodb.create_new_project(project_id, df)

                        if inserted_rows > 0:
                            userId = session.get('id')
                            status = 1
                            query = f"""INSERT INTO tblProjects (UserId, Name, Description, Status, 
                                               Cassandra_Table_Name,Pid) VALUES
                                               ("{userId}", "{name}", "{description}", "1", "{table_name}","{project_id}")"""

                            rowcount = mysql.insert_record(query)
                            if rowcount > 0:
                                logger.info('Project Created!!')
                                flash("Success!!")
                                return redirect(url_for('index'))
                            else:
                                message = "Error while creating new Project"
                                logger.error(message)
                                return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)
                        else:
                            message = "Error while creating new Project"
                            logger.error(message)
                            return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)
                    else:
                        message = "Error while creating new Project"
                        logger.error(message)
                        return render_template('new_project.html', msg=message, project_types=PROJECT_TYPES)
        else:
            return redirect(url_for('login'))

    except Exception as e:
        logger.error(e)
        return render_template('new_project.html', project_types=PROJECT_TYPES, msg=e.__str__())


# Make Prediction Route
@app.route('/prediction_file/<action>', methods=['GET', 'POST'])
def prediction(action):
    if 'loggedin' in session:
        try:
            if request.method == "POST":
                download_status = None
                file_path = None
                source_type = request.form['source_type']

                if source_type == 'uploadFile':
                    ALLOWED_EXTENSIONS = ['csv', 'tsv', 'json', 'xlsx']
                    f = request.files['file']
                    filename = secure_filename(f.filename)
                    file_path = os.path.join('artifacts', f'{action}', filename)
                    f.save(file_path)

                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif file_path.endswith('.tsv'):
                        df = pd.read_csv(file_path, sep='\t')
                    elif file_path.endswith('.json'):
                        df = pd.read_json(file_path)
                    elif file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                    else:
                        msg = 'This file format is currently not supported'
                        return render_template('prediction.html', msg=msg)

                    return redirect(url_for('index'))

                elif source_type == 'uploadResource':
                    resource_type = request.form['resource_type']

                    if resource_type == "awsS3bucket":
                        region_name = request.form['region_name']
                        aws_access_key_id = request.form['aws_access_key_id']
                        aws_secret_access_key = request.form['aws_secret_access_key']
                        bucket_name = request.form['bucket_name']
                        file_name = request.form['file_name']
                        file_path = os.path.join('src/temp_data_store', file_name)
                        aws_s3 = aws_s3_helper(region_name, aws_access_key_id, aws_secret_access_key)
                        conn_msg = aws_s3.check_connection(bucket_name, file_name)
                        if conn_msg != 'Successful':
                            logger.info(conn_msg)
                            return render_template('prediction.html', msg=conn_msg)

                        download_status = aws_s3.download_file_from_s3(bucket_name, file_name, file_path)
                        logger.info(resource_type, download_status, file_path)

                    elif resource_type == "gcpStorage":
                        credentials_file = request.files['GCP_credentials_file']
                        bucket_name = request.form['bucket_name']
                        file_name = request.form['file_name']
                        credentials_filename = secure_filename(credentials_file.filename)
                        credentials_file_path = os.path.join(app.config['UPLOAD_FOLDER'], credentials_filename)
                        credentials_file.save(credentials_file_path)
                        file_path = os.path.join('src/temp_data_store', file_name)
                        logger.info(credentials_file_path, file_path, file_name, bucket_name)
                        gcp = gcp_browser_storage(credentials_file_path)
                        conn_msg = gcp.check_connection(bucket_name, file_name)
                        logger.info(conn_msg)
                        if conn_msg != 'Successful':
                            logger.info(conn_msg)
                            return render_template('prediction.html', msg=conn_msg)

                        download_status = gcp.download_file_from_bucket(file_name, file_path, bucket_name)
                        logger.info(download_status)

                    elif resource_type == "mySql":
                        host = request.form['host']
                        port = request.form['port']
                        user = request.form['user']
                        password = request.form['password']
                        database = request.form['database']
                        table_name = request.form['table_name']
                        file_path = os.path.join('src/temp_data_store', table_name)
                        logger.info(file_path)

                        mysql_data = mysql_data_helper(host, port, user, password, database)
                        conn_msg = mysql_data.check_connection(table_name)
                        logger.info(conn_msg)
                        if conn_msg != 'Successful':
                            logger.info(conn_msg)
                            return render_template('prediction.html', msg=conn_msg)

                        download_status = mysql_data.retrive_dataset_from_table(table_name, file_path)
                        logger.info(download_status)

                    elif resource_type == "cassandra":
                        secure_connect_bundle = request.files['secure_connect_bundle']
                        client_id = request.form['client_id']
                        client_secret = request.form['client_secret']
                        keyspace = request.form['keyspace']
                        table_name = request.form['table_name']
                        data_in_tabular = request.form['data_in_tabular']
                        secure_connect_bundle_filename = secure_filename(secure_connect_bundle.filename)
                        secure_connect_bundle_file_path = os.path.join(r'src/temp_data_store',
                                                                       secure_connect_bundle_filename)
                        secure_connect_bundle.save(secure_connect_bundle_file_path)
                        file_path = os.path.join('src/temp_data_store', f"{table_name}.csv")
                        logger.info(secure_connect_bundle_file_path, file_path)

                        cassandra_db = cassandra_connector(secure_connect_bundle_file_path, client_id, client_secret,
                                                           keyspace)
                        conn_msg = cassandra_db.check_connection(table_name)
                        logger.info(conn_msg)
                        if conn_msg != 'Successful':
                            logger.info(conn_msg)
                            return render_template('prediction.html', msg=conn_msg)

                        if data_in_tabular == 'true':
                            download_status = cassandra_db.retrive_table(table_name, file_path)
                            logger.info(download_status)
                        elif data_in_tabular == 'false':
                            download_status = cassandra_db.retrive_uploded_dataset(table_name, file_path)
                            logger.info(download_status)

                    elif resource_type == "mongodb":
                        mongo_db_url = request.form['mongo_db_url']
                        mongo_database = request.form['mongo_database']
                        collection = request.form['collection']
                        file_path = os.path.join('src/temp_data_store', f"{collection}.csv")
                        mongo_helper = mongo_data_helper(mongo_db_url)
                        conn_msg = mongo_helper.check_connection(mongo_database, collection)
                        if conn_msg != 'Successful':
                            logger.info(conn_msg)
                            return render_template('prediction.html', msg=conn_msg)

                        download_status = mongo_helper.retrive_dataset(mongo_database, collection, file_path)
                        logger.info(download_status)

                    elif resource_type == "azureStorage":
                        azure_connection_string = request.form['azure_connection_string']
                        container_name = request.form['container_name']
                        file_name = request.form['file_name']
                        file_path = os.path.join('src/temp_data_store', file_name)
                        azure_helper = azure_data_helper(azure_connection_string)
                        conn_msg = azure_helper.check_connection(container_name, file_name)

                        if conn_msg != 'Successful':
                            logger.info(conn_msg)
                            return render_template('prediction.html', msg=conn_msg)

                        download_status = azure_helper.download_file(container_name, file_name, file_path)
                        logger.info(download_status)
                    else:
                        return None

                    if download_status == 'Successful':

                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        elif file_path.endswith('.tsv'):
                            df = pd.read_csv(file_path, sep='\t')
                        elif file_path.endswith('.json'):
                            df = pd.read_json(file_path)
                        elif file_path.endswith('.xlsx'):
                            df = pd.read_excel(file_path)
                        else:
                            msg = 'This file format is currently not supported'
                            logger.error(msg)
                            return render_template('prediction.html', msg=msg)

                        return redirect(url_for('index'))
                    else:
                        return render_template('prediction.html', loggedin=True, data={'pid': action},
                                               msg="Failed to download the file!!")
            else:
                return render_template('prediction.html', loggedin=True)

        except Exception as e:
            logger.error(e)
            return render_template('prediction.html', loggedin=True, msg=e.__str__())
    else:
        return redirect(url_for('login'))


# Login Page Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = None
    if 'loggedin' in session:
        logger.info('Redirect To Main Page')
        return redirect('/')
    else:
        if request.method == "GET":
            logger.info('Login Template Rendering')
            return render_template('login.html')
        else:
            if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
                email = request.form['email']
                password = request.form['password']
                account = mysql.fetch_one(
                    f'SELECT * FROM tblUsers WHERE Email = "{email}" AND Password = "{Hashing.hash_value(password)}"')
                if account:
                    session['loggedin'] = True
                    session['id'] = account[0]
                    session['username'] = account[1]
                    logger.info('Login Successful')
                    return redirect('/')
                else:
                    msg = 'Incorrect username / password !'
                    logger.error(msg)
            return render_template('login.html', msg=msg)


# SignUp Page Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if 'loggedin' in session:
        return redirect(url_for('index'))
    else:
        if request.method == "GET":
            logger.info('Signup Template Rendering')
            return render_template('signup.html')
        else:
            msg = None
            if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
                username = request.form['username']
                password = request.form['password']
                confirm_password = request.form['confirm-password']
                email = request.form['email']
                account = mysql.fetch_one(f'SELECT * FROM tblUsers WHERE Email = "{email}"')
                logger.info('Checking Database')
                if account:
                    msg = 'EmailId already exists !'
                elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                    msg = 'Invalid email address !'
                elif not re.match(r'[A-Za-z0-9]+', username):
                    msg = 'Username must contain only characters and numbers !'
                elif not username or not password or not email:
                    msg = 'Please fill out the form !'
                elif confirm_password != password:
                    msg = 'Password and Confirm password are not same!'
                else:
                    hashed_password = Hashing.hash_value(password)
                    rowcount = mysql.insert_record(
                        f'INSERT INTO tblUsers (Name, Email, Password, AuthToken) VALUES ("{username}", "{email}", "{hashed_password}", "pankajtest")')
                    if rowcount > 0:
                        return redirect(url_for('login'))
            elif request.method == 'POST':
                msg = 'Please fill out the form !'
                logger.error(msg)
            logger.info(msg)
            return render_template('signup.html', msg=msg)


# ExportFile Route
@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    # note that we set the 404 status explicitly
    return render_template('500.html', msg=str(e)), 500


@app.route('/export-resources/<pid>', methods=['GET', 'POST'])
def exportResources(pid):
    try:
        if 'loggedin' in session:
            if request.method == 'GET':
                folder_path = os.path.join(from_root(), config_args['dir_structure']['artifacts_dir'], pid)
                if not os.path.exists(folder_path):
                    return render_template('export-resources.html', status="error",
                                           msg="No resources found to export, please train your model first")
                logger.info('Redirect To Export Reources Page')
                return render_template('export-resources.html', status="success", pid=pid)
            else:
                folder_path = os.path.join(from_root(), config_args['dir_structure']['artifacts_dir'], pid)

                """Get Projects Actions"""
                query_ = f"""
                        Select tblProjectActions.Name , Input,Output from  tblProject_Actions_Reports 
                        Join tblProjectActions on tblProject_Actions_Reports.ProjectActionId=tblProjectActions.Id
                        join tblProjects on tblProjects.Id=tblProject_Actions_Reports.ProjectId
                        where PId="{pid}"
                        """
                action_performed = mysql.fetch_all(query_)

                """Save Actions file"""
                if len(action_performed) > 0:
                    df = pd.DataFrame(action_performed, columns=['Action', 'Input', 'Output'])
                    df.to_csv(os.path.join(folder_path, 'actions.csv'))

                base_path = pathlib.Path(folder_path)
                data = io.BytesIO()
                with zipfile.ZipFile(data, mode='w') as z:
                    for f_name in base_path.iterdir():
                        z.write(f_name)
                data.seek(0)
                return Response(
                    data,
                    mimetype='application/zip',
                    headers={"Content-disposition": f"attachment; filename=data.zip"})
        else:
            return redirect(url_for('login'))
    except Exception as e:
        logger.info(e)
        return render_template('export-resources.html', status="error", msg=str(e))


@app.route('/exportFile/<pid>/<project_name>', methods=['GET'])
def exportForm(pid, project_name):
    if 'loggedin' in session:
        logger.info('Redirect To Export File Page')
        return render_template('exportFile.html',
                               data={"project_name": project_name, "project_id": pid})
    else:
        return redirect(url_for('login'))


@app.route('/exportFile/<project_id>/<project_name>', methods=['POST'])
def exportFile(project_id, project_name):
    try:
        file_path = None
        if 'loggedin' in session:
            logger.info('Export File in Process')
            fileType = request.form['fileType']

            if fileType != "":
                download_status, file_path = mongodb.download_collection_data(project_id, 'csv')
                logger.info(f'Temporary File Created!!, {project_name}.csv')
                if download_status != "Successful":
                    render_template('exportFile.html',
                                    data={"project_name": project_name, "project_id": project_id},
                                    msg="OOPS something went wrong!!")

            if fileType == 'csv':
                content = pd.read_csv(file_path)
                remove_temp_files([file_path])
                logger.info(f'Temporary File Deleted!!, {project_name}.csv')
                logger.info('Exported to CSV Sucessful')
                return Response(content.to_csv(index=False), mimetype="text/csv",
                                headers={"Content-disposition": f"attachment; filename={project_name}.csv"})

            elif fileType == 'tsv':
                content = pd.read_csv(file_path)
                remove_temp_files([file_path])
                logger.info(f'Temporary File Deleted!!, {project_name}.tsv')
                logger.info('Exported to TSV Sucessful')
                return Response(content.to_csv(sep='\t', index=False), mimetype="text/tsv",
                                headers={"Content-disposition": f"attachment; filename={project_name}.tsv"})

            elif fileType == 'xlsx':
                content = pd.read_csv(file_path)
                remove_temp_files([file_path])
                logger.info(f'Temporary File Deleted!!, {project_name}.xlsx')
                content.to_excel(os.path.join(app.config["UPLOAD_FOLDER"], f'{project_name}.xlsx'), index=False)
                logger.info('Exported to XLSX Sucessful')
                return send_from_directory(directory=app.config["UPLOAD_FOLDER"], path=f'{project_name}.xlsx',
                                           as_attachment=True)

            elif fileType == 'json':
                content = pd.read_csv(file_path)
                remove_temp_files([file_path])
                logger.info(f'Temporary File Deleted!!, {project_name}.json')
                logger.info('Exported to JSON Sucessful')
                return Response(content.to_json(), mimetype="text/json",
                                headers={"Content-disposition": f"attachment; filename={project_name}.json"})
            else:
                return render_template('exportFile.html', data={"project_name": project_name, "project_id": project_id},
                                       msg="Select Any File Type!!")
        else:
            return redirect(url_for('login'))
    except Exception as e:
        logger.error(e)
        return render_template('exportFile.html', data={"project_name": project_name, "project_id": project_id},
                               msg=e.__str__())


@app.route('/exportProject/<project_name>/<project_id>', methods=['GET', 'POST'])
def exportCloudDatabaseFile(project_name, project_id):
    try:
        download_status = None
        if 'loggedin' in session:
            logger.info('Export File')
            source_type = request.form['source_type']

            if source_type == 'uploadCloud':
                cloudType = request.form['cloudType']

                if cloudType == 'awsS3bucket':
                    region_name = request.form['region_name']
                    aws_access_key_id = request.form['aws_access_key_id']
                    aws_secret_access_key = request.form['aws_secret_access_key']
                    bucket_name = request.form['aws_bucket_name']
                    file_type = request.form['fileTypeAws']

                    aws_s3 = aws_s3_helper(region_name, aws_access_key_id, aws_secret_access_key)
                    logger.info("Validating User AWS S3 Credentials")
                    conn_msg = aws_s3.check_connection(bucket_name, 'none')

                    if conn_msg != 'File does not exist!!':
                        logger.info("AWS S3 Connection Not Successful!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=conn_msg)
                    logger.info("AWS S3 Connection Successful!!")
                    logger.info("Looking For User File!!")
                    download_status, file_path = mongodb.download_collection_data(project_id, file_type)
                    if download_status != "Successful":
                        logger.info("Could'nt Download The File!!")
                        render_template('exportFile.html',
                                        data={"project_name": project_name, "project_id": project_id},
                                        msg="OOPS something went wrong!!")

                    timestamp = round(time.time() * 1000)
                    upload_status = aws_s3.push_file_to_s3(bucket_name, file_path,
                                                           f'{project_name}_{timestamp}.{file_type}')
                    remove_temp_files([file_path])
                    if upload_status != 'Successful':
                        logger.info("Could'nt Upload The File To s3 Bucket!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=upload_status)
                    message = f"{project_name}_{timestamp}.{file_type} pushed to AWS S3 {bucket_name} bucket"
                    logger.info(message)
                    return render_template('exportFile.html',
                                           data={"project_name": project_name, "project_id": project_id}, msg=message)

                elif cloudType == 'azureStorage':
                    azure_connection_string = request.form['azure_connection_string']
                    container_name = request.form['container_name']
                    file_type = request.form['fileTypeAzure']
                    azure_helper = azure_data_helper(azure_connection_string)
                    logger.info("Validating User Azure Credentials")
                    conn_msg = azure_helper.check_connection(container_name, 'none')
                    if conn_msg != 'File does not exist!!':
                        logger.info("AzureStorage Connection Not Successful!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=conn_msg)
                    logger.info("AzureStorage Connection Successful!!")
                    download_status, file_path = mongodb.download_collection_data(project_id, file_type)

                    logger.info("Looking For User File!!")
                    if download_status != "Successful":
                        logger.info("Could'nt Download The File!!")
                        render_template('exportFile.html',
                                        data={"project_name": project_name, "project_id": project_id},
                                        msg="OOPS something went wrong!!")

                    timestamp = round(time.time() * 1000)
                    upload_status = azure_helper.upload_file(file_path, container_name,
                                                             f'{project_name}_{timestamp}.{file_type}')
                    remove_temp_files([file_path])
                    if upload_status != 'Successful':
                        logger.info("Could'nt Upload The File To Azure Container!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=upload_status)
                    message = f"{project_name}_{timestamp}.{file_type} pushed to Azure {container_name} container"
                    logger.info(message)
                    return render_template('exportFile.html',
                                           data={"project_name": project_name, "project_id": project_id}, msg=message)

                elif cloudType == 'gcpStorage':
                    credentials_file = request.files['GCP_credentials_file']
                    bucket_name = request.form['gcp_bucket_name']
                    file_type = request.form['fileTypeGcp']
                    credentials_filename = secure_filename(credentials_file.filename)
                    credentials_file_path = os.path.join(app.config['UPLOAD_FOLDER'], credentials_filename)
                    credentials_file.save(credentials_file_path)
                    gcp = gcp_browser_storage(credentials_file_path)
                    logger.info("Validating User Azure Credentials")
                    conn_msg = gcp.check_connection(bucket_name, 'none')
                    remove_temp_files([credentials_file_path])
                    if conn_msg != 'File does not exist!!':
                        logger.info("GCPStorage Connection Not Successful!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=conn_msg)
                    logger.info("GCPStorage Connection Successful!!")
                    download_status, file_path = mongodb.download_collection_data(project_id, file_type)
                    logger.info("Looking For User File!!")
                    if download_status != "Successful":
                        logger.info("Could'nt Download The File!!")
                        render_template('exportFile.html',
                                        data={"project_name": project_name, "project_id": project_id},
                                        msg="OOPS something went wrong!!")

                    timestamp = round(time.time() * 1000)
                    upload_status = gcp.upload_to_bucket(f'{project_name}_{timestamp}.{file_type}',
                                                         file_path, bucket_name)
                    remove_temp_files([file_path])
                    if upload_status != 'Successful':
                        logger.info("Could'nt Upload The File To GCP Container!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=upload_status)
                    message = f"{project_name}_{timestamp}.{file_type} pushed to GCP Storage{bucket_name} bucket"
                    logger.info(message)
                    return render_template('exportFile.html',
                                           data={"project_name": project_name, "project_id": project_id}, msg=message)
                else:
                    return render_template('exportFile.html',
                                           data={"project_name": project_name, "project_id": project_id},
                                           msg="Select Any Cloud Type!!")

            elif source_type == 'uploadDatabase':
                databaseType = request.form['databaseType']

                if databaseType == 'mySql':
                    host = request.form['host']
                    port = request.form['port']
                    user = request.form['user']
                    password = request.form['password']
                    database = request.form['database']

                    mysql_data = mysql_data_helper(host, port, user, password, database)
                    logger.info("Validating User Mysql Credentials")
                    conn_msg = mysql_data.check_connection('none')
                    if conn_msg != "table does not exist!!":
                        logger.info("Users Mysql Connection Not Successful!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=conn_msg)
                    logger.info("Users Mysql Connection Successful!!")
                    logger.info("Looking For User File!!")
                    download_status, file_path = mongodb.download_collection_data(project_id, "csv")

                    if download_status != "Successful":
                        render_template('exportFile.html',
                                        data={"project_name": project_name, "project_id": project_id},
                                        msg="OOPS something went wrong!!")

                    timestamp = round(time.time() * 1000)
                    upload_status = mysql_data.push_file_to_table(file_path, f'{project_name}_{timestamp}')
                    remove_temp_files([file_path])
                    if download_status != 'Successful' or upload_status != 'Successful':
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=upload_status)

                    message = f'{project_name}_{timestamp} table created in {database} database'
                    logger.info(message)
                    return render_template('exportFile.html',
                                           data={"project_name": project_name, "project_id": project_id}, msg=message)

                elif databaseType == 'cassandra':
                    secure_connect_bundle = request.files['secure_connect_bundle']
                    client_id = request.form['client_id']
                    client_secret = request.form['client_secret']
                    keyspace = request.form['keyspace']
                    secure_connect_bundle_filename = secure_filename(secure_connect_bundle.filename)
                    secure_connect_bundle_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                                                   secure_connect_bundle_filename)
                    secure_connect_bundle.save(secure_connect_bundle_file_path)

                    cassandra_db = cassandra_connector(secure_connect_bundle_file_path,
                                                       client_id, client_secret, keyspace)

                    logger.info("Validating User Cassandra Credentials")
                    conn_msg = cassandra_db.check_connection('none')
                    remove_temp_files([secure_connect_bundle_file_path])
                    if conn_msg != 'table does not exist!!':
                        logger.info("Users Cassandra Connection Not Successful!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=conn_msg)
                    logger.info("Users Cassandra Connection Successful!!")
                    logger.info("Looking For User File!!")
                    download_status, file_path = mongodb.download_collection_data(project_id, "csv")

                    if download_status != "Successful":
                        render_template('exportFile.html',
                                        data={"project_name": project_name, "project_id": project_id},
                                        msg="OOPS something went wrong!!")

                    timestamp = round(time.time() * 1000)
                    upload_status = cassandra_db.push_dataframe_to_table(pd.read_csv(file_path),
                                                                         f'{project_name}_{timestamp}')
                    remove_temp_files([file_path])
                    if download_status != 'Successful' or upload_status != 'Successful':
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=upload_status)
                    message = f'{project_name}_{timestamp} table created in {keyspace} keyspace'
                    logger.info(message)
                    return render_template('exportFile.html',
                                           data={"project_name": project_name, "project_id": project_id}, msg=message)

                elif databaseType == 'mongodb':
                    mongo_db_url = request.form['mongo_db_url']
                    mongo_database = request.form['mongo_database']
                    mongo_helper = mongo_data_helper(mongo_db_url)
                    conn_msg = mongo_helper.check_connection(mongo_database, 'none')
                    logger.info("Validating User Cassandra Credentials")
                    if conn_msg != "collection does not exits!!":
                        logger.info("Users MongoDB Connection Not Successful!!")
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=conn_msg)
                    logger.info("Users MongoDB Connection Successful!!")
                    download_status, file_path = mongodb.download_collection_data(project_id, "csv")
                    if download_status != "Successful":
                        render_template('exportFile.html',
                                        data={"project_name": project_name, "project_id": project_id},
                                        msg="OOPS something went wrong!!")
                    logger.info("Looking For User File!!")
                    timestamp = round(time.time() * 1000)
                    upload_status = mongo_helper.push_dataset(mongo_database, f'{project_name}_{timestamp}', file_path)
                    remove_temp_files([file_path])
                    if download_status != 'Successful' or upload_status != 'Successful':
                        return render_template('exportFile.html',
                                               data={"project_name": project_name, "project_id": project_id},
                                               msg=upload_status)

                    message = f'{project_name}_{timestamp} collection created in {mongo_database} database'
                    logger.info(message)
                    return render_template('exportFile.html',
                                           data={"project_name": project_name, "project_id": project_id}, msg=message)
                else:
                    return render_template('exportFile.html',
                                           data={"project_name": project_name, "project_id": project_id},
                                           msg="Select Any Database Type!!")
            else:
                return render_template('exportFile.html',
                                       data={"project_name": project_name, "project_id": project_id},
                                       msg="Select Any Cloud or Database")
        else:
            return redirect(url_for('login'))
    except Exception as e:
        logger.info(e)
        return render_template('exportFile.html', data={"project_name": project_name},
                               msg="OOPS Something Went Wrong!!")


@app.route('/projectReport/<id>', methods=['GET', 'POST'])
def projectReport(id):
    if 'loggedin' in session:
        logger.info('Redirect To Project Report Page')
        records, projectStatus = ProjectReports.get_record_by_pid(id, None)

        graphJSON = ""
        pie_graphJSON = ""

        df = pd.DataFrame(records)
        if df is not None:
            df_counts = pd.DataFrame(df.groupby('Module Name').count()).reset_index(level=0)
            y = list(pd.DataFrame(df.groupby('Module Name').count()).reset_index(level=0).columns)[-1]
            df_counts['Total Actions'] = df_counts[y]
            graphJSON = PlotlyHelper.barplot(df_counts, x='Module Name', y=df_counts['Total Actions'])
            pie_graphJSON = PlotlyHelper.pieplot(df_counts, names='Module Name', values='Total Actions', title='')
        return render_template('projectReport.html', data={"id": id, "moduleId": None}, records=records.to_html(),
                               projectStatus=projectStatus, graphJSON=graphJSON, pie_graphJSON=pie_graphJSON)
    else:
        return redirect(url_for('login'))


@app.route('/deletePage/<id>', methods=['GET'])
def renderDeleteProject(id):
    if 'loggedin' in session:
        logger.info('Redirect To Delete Project Page')
        return render_template('deleteProject.html', data={"id": id})
    else:
        return redirect(url_for('login'))


@app.route('/target-column', methods=['GET', 'POST'])
def setTargetColumn():
    try:
        if 'loggedin' in session and 'id' in session and session['project_type'] != 3 and session[
            'target_column'] is None:
            logger.info('Redirect To Target Column Page')
            df = load_data()
            columns = list(df.columns)
            if request.method == "GET":
                return render_template('target_column.html', columns=columns)
            else:
                status = "error"
                id = session.get('pid')
                target_column = request.form['column']
                rows_count = mysql.delete_record(f'UPDATE tblProjects SET TargetColumn="{target_column}" WHERE Id={id}')
                status = "success"
                # add buttom here
                if status == "success":
                    session['target_column'] = target_column
                    return redirect('/module')
                else:
                    return redirect('/module')

        else:
            logger.info('Redirect To Home Page')
            return redirect('/module')
    except Exception as e:
        logger.error(f'{e}, Occur occurred in target-columns.')
        return render_template('500.html', exception=e)
        # return redirect('/')


@app.route('/deleteProject/<id>', methods=['GET'])
def deleteProject(id):
    if 'loggedin' in session:
        try:
            if id:
                mysql.delete_record(f'UPDATE tblProjects SET IsActive=0 WHERE Pid="{id}"')
                logger.info('Data Successfully Deleted From Database')
                mongodb.drop_collection(id)
                # log.info(log_type='INFO', log_message='Data Successfully Deleted From Database')
                return redirect(url_for('index'))
            else:
                logger.info('Redirect to index invalid id')
                return redirect(url_for('index'))
        except Exception as ex:
            logger.info(str(ex))
            return render_template('500.html', exception=ex)

    else:
        logger.info('Login Needed')
        return redirect(url_for('login'))


"""[summary]
Route for logout
Raises:
    Exception: [description]
Returns:
    [type]: [description]
"""


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    session.pop('pid', None)
    session.pop('project_name', None)
    session.pop('project_type', None)
    session.pop('target_column', None)
    logger.info('Thanks For Using System!')
    return redirect(url_for('contact'))


"""[summary]
Entry Point on Any Project when click on project name
Raises:
    Exception: [description]
Returns:
    [type]: [description]
"""


@app.route('/stream/<pid>')
def stream(pid):
    try:
        data = decrypt(pid)
        if data:
            values = data.split("&")
            session['pid'] = values[1]
            query_ = f"Select ProjectType, TargetColumn,Name from tblProjects  where id={session['pid']}"
            info = mysql.fetch_one(query_)
            if info:
                session['project_name'] = values[0]
                session['project_type'] = info[0]
                if info[0] != 3:
                    session['target_column'] = info[1]
                else:
                    session['target_column'] = None

            mongodb.get_collection_data(values[0])
            return redirect(url_for('module'))
        else:
            return redirect(url_for('/'))
    except Exception as e:
        logger.error(e)
        return render_template('500.html', exception=e)


@app.route('/module')
def module():
    try:
        if 'pid' in session:
            logger.info(f'Inside {session["project_name"]}')
            return render_template('help.html')
        else:
            logger.info('Redirected to login')
            return redirect(url_for('/'))
    except Exception as e:
        logger.error(e)
        return render_template('500.html', exception=e)


@app.route('/systemlogs/<action>', methods=['GET'])
def systemlogs(action):
    try:
        if action == 'terminal':
            lines = []
            path = os.path.join(from_root(), 'logger', 'logs', 'logs.log')
            with open(path) as file_in:
                for line in file_in:
                    lines.append(line)
            file_in.close()
            return render_template('systemlogs/terminal.html', logs=lines)
        else:
            return 'Not Visible'
    except Exception as e:
        logger.error(f"{e} In System Logs API")
        return render_template('500.html', exception=e)


@app.route('/history/actions', methods=['GET'])
def history():
    try:
        my_collection = mysql.fetch_all(f''' Select Name, Input,Output,ActionDate 
        from tblProject_Actions_Reports 
        Join tblProjectActions on tblProject_Actions_Reports.ProjectActionId=tblProjectActions.Id 
        where ProjectId ="{session['pid']}"''')

        data = ""
        if len(my_collection) > 0:
            df = pd.DataFrame(np.array(my_collection), columns=['Action', 'Input', 'Output', 'DateTime'])
            data = df.to_html()
        return render_template('history/actions.html', status="success", data=data)
    except Exception as e:
        logger.info(e)
        ProjectReports.insert_record_dp(f'Error In History Page :{str(e)}')
        return render_template('history/actions.html', status="error", msg=str(e))


@app.route('/custom-script', methods=['GET', 'POST'])
def custom_script():
    try:
        if 'loggedin' in session:
            df = load_data()
            if df is not None:
                logger.info('Redirect To Custom Script')
                ProjectReports.insert_record_fe('Redirect To Custom Script')
                data = df.head(100).to_html()
                if request.method == 'GET':
                    return render_template('custom-script.html', status="success", data=data)
                else:
                    df = load_data()
                    code = request.form['code']
                    # Double quote is not allowed
                    if 'import' in code:
                        return render_template('custom-script.html', status="error", msg="Import is not allowed")
                    if '"' in code:
                        return render_template('custom-script.html', status="error", msg="Double quote is not allowed")

                    if code is not None:
                        try:
                            globalsParameter = {'os': None, 'pd': pd, 'np': np}
                            localsParameter = {'df': df}
                            exec(code, globalsParameter, localsParameter)
                            update_data(df)
                            ProjectReports.insert_project_action_report(ProjectActions.CUSTOM_SCRIPT.value, code)
                            return redirect('/eda/show')
                        except Exception as e:
                            return render_template('custom-script.html', status="error",
                                                   msg="Code snippets is not valid")

                    else:
                        return render_template('custom-script.html', status="error", msg="Code snippets is not valid")
            else:
                return redirect(url_for('/'))
        else:
            return redirect(url_for('login'))
    except Exception as e:
        logger.error(e)
        ProjectReports.insert_record_fe(f'Error In Custom Script: {str(e)}')
        return render_template('custom-script.html', status="error", msg=str(e))


@app.route('/insights/<action>', methods=['GET', 'POST'])
def data_insights(action):
    try:
        if request.method == 'GET':
            if 'pid' in session and 'id' in session:
                df = load_data()
                if df is not None:
                    if action == "data_insights":
                        col_lst = list(df.columns)
                        return render_template('insights/data_insights.html', columns=list(df.columns), action=action)
                else:
                    return 'No Data'
            else:
                return redirect('/')

        elif request.method == 'POST':
            if 'pid' in session and 'id' in session:
                df = load_data()
                if df is not None:
                    if action == "data_insights":
                        columns = request.form.getlist('columns')
                        print(columns)
                        return render_template('insights/insights_.html', columns=columns, action=action)
                else:
                    return 'No Data'
            else:
                return redirect('/')

    except Exception as e:
        logger.error(e)


@app.route('/scheduler/<action>', methods=['GET'])
def scheduler_get(action):
    try:
        df = load_data()
        if 'loggedin' in session:
            if df is not None:
                if action == 'help':
                    return render_template('scheduler/help.html')

                if action == 'Training_scheduler':
                    # To get the trained
                    Model_Trained, model_name, TargetColumn, pid = mysql.fetch_one(
                        f"""select Model_Trained, Model_Name,TargetColumn, pid  from tblProjects Where Id={session.get('pid')}""")

                    query = f"""select a.pid ProjectId ,
                                            a.TargetColumn TargetName, 
                                            a.Model_Name ModelName, 
                                            a.Model_Trained, 
                                            b.train_status ,
                                            b.email, 
                                            b.datetime_,
                                            NOW()
                                            from tblProjects as a
                                            join tblProject_scheduler as b on a.Pid = b.ProjectId where a.pid ="{pid}"
                                            and b.deleted=0
                                           """

                    result = mysql.fetch_one(query)

                    if Model_Trained == 0:
                        if model_name is None:
                            return render_template('scheduler/retrain.html')

                        if result is None:
                            return render_template('scheduler/add_new_scheduler.html',
                                                   action=action,
                                                   model_name=model_name,
                                                   status="Success",
                                                   msg="Model is not selected, please select your model first",
                                                   TargetColumn=TargetColumn)

                        if result is not None:
                            responseData = [{
                                "project_id": pid,
                                "mode_names": model_name,
                                "target_col_name": TargetColumn,
                                "status": result[4],
                                "DateTime": result[6],
                                "email_send": result[5],
                                "CurrentDateTime": result[7]
                            }]

                            return render_template('scheduler/Training_scheduler.html', action=action,
                                                   responseData=responseData)
                        else:
                            return render_template('scheduler/add_new_scheduler.html', model_name=model_name,
                                                   TargetColumn=TargetColumn, action=action, ALL_MODELS=ALL_MODELS,
                                                   TIMEZONE=TIMEZONE)

                    if Model_Trained == 1:
                        # Retrain for scheduler
                        if result is None:
                            return render_template('scheduler/retrain.html')

                        if result is not None:
                            return render_template('500.html', exception="Something Went Wrong!")

                    else:
                        return render_template('scheduler/add_new_scheduler.html', action=action, ALL_MODELS=ALL_MODELS)

                if action == "add_scheduler":
                    return render_template('scheduler/add_new_scheduler.html', action=action, ALL_MODELS=ALL_MODELS,
                                           TIMEZONE=TIMEZONE)

                if action == 'deleteScheduler':
                    pid = mysql.fetch_one(f"""select pid from tblProjects Where Id={session.get('pid')}""")
                    query = f'DELETE FROM tblProject_scheduler WHERE ProjectId = "{pid[0]}" '
                    mysql.delete_record(query)
                    return redirect('/scheduler/Training_scheduler')
            else:
                return redirect('/')
        else:
            return redirect(url_for('login'))

    except Exception as e:
        logger.error(f"{e} In scheduler")
        return render_template('500.html', exception=e)


@app.route('/scheduler/<action>', methods=['POST'])
def scheduler_post(action):
    try:
        if 'loggedin' in session:
            Model_Trained, model_name, TargetColumn, pid = mysql.fetch_one(
                f"""select Model_Trained, Model_Name,TargetColumn, pid  from tblProjects Where Id={session.get('pid')}""")
            if action == 'help':
                return render_template('scheduler/help.html')

            if action == 'Training_scheduler':
                time_after = int(request.form['time_after'])
                email = request.form['email']

                query = f''' INSERT INTO tblProject_scheduler 
                             (ProjectId,datetime_,email,train_status,deleted)
                            values('{pid}',DATE_ADD(NOW(),INTERVAL {time_after} HOUR),'{email}' ,0,0) '''

                row_effected = mysql.update_record(query)
                return redirect('/scheduler/Training_scheduler')
        else:
            return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"{e} In scheduler")
        return render_template('500.html', exception=e)


if __name__ == '__main__':
    if mysql is None or mongodb is None:
        print("Not Able To connect With Database (Check Mongo and Mysql Connection)")
    else:
        app.run(host="0.0.0.0", port=8080, debug=True)
