import pickle
import uuid
import yaml
import hashlib
from cryptography.fernet import Fernet
import json
import os
import re
import pandas as pd
from flask import session
from pickle import dump
from from_root import from_root
from flask import send_file

from src.constants.model_params import Params_Mappings


def read_config(config):
    with open(config) as config:
        content = yaml.safe_load(config)

    return content


def unique_id_generator():
    random = uuid.uuid4()
    unique_id = "PID" + str(random)

    return unique_id


class Hashing:
    @staticmethod
    def hash_value(value):
        hash_object = hashlib.md5(value.encode('utf-8'))
        return hash_object.hexdigest()


# we will be encrypting the below string.
def encrypt(message):
    key = b'r7T4WUAHgeAFSwwWVauOdCDsvWugU4xWxlLR1OKayI4='
    fernet = Fernet(key)
    encMessage = fernet.encrypt(message.encode())
    return encMessage


def decrypt(message):
    key = b'r7T4WUAHgeAFSwwWVauOdCDsvWugU4xWxlLR1OKayI4='
    fernet = Fernet(key)
    encMessage = fernet.decrypt(message.encode())
    return encMessage.decode("utf-8")


def immutable_multi_dict_to_str(immutable_multi_dict, flat=False):
    input_str = immutable_multi_dict.to_dict(flat)
    input_str = {key: value if len(value) > 1 else value[0] for key, value in input_str.items()}
    return json.dumps(input_str)


def save_project_encdoing(encoder):
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'))
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, 'encoder.pkl')
    dump(encoder, open(file_name, 'wb'))


def save_project_scaler(encoder):
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'))
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, 'scaler.pkl')
    dump(encoder, open(file_name, 'wb'))


def save_project_pca(pca):
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'))
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, 'pca.pkl')
    dump(pca, open(file_name, 'wb'))


def load_project_pca():
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'), 'pca.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model
    else:
        return None
    

def save_project_model(model, name='model_temp.pkl'):
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'))
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, name)
    dump(model, open(file_name, 'wb'))


def load_project_model():
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'), 'model_temp.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model
    else:
        return None


def load_project_encdoing():
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'), 'encoder.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model
    else:
        return None


def load_project_scaler():
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'), 'scaler.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model
    else:
        return None



def save_prediction_result(df):
    path = os.path.join(from_root(), 'artifacts', session.get('project_name'))
    if not os.path.exists(path):
        os.mkdir(path)

    file_name = os.path.join(path, "prediction.csv")
    df.to_csv(file_name, index=False)


def load_prediction_result():
    file_name = os.path.join(from_root(), 'artifacts', session.get('project_name'), "prediction.csv")
    return send_file(file_name,
                     mimetype='text/csv',
                     attachment_filename='predictions.csv',
                     as_attachment=True)


def get_param_value(obj, value):
    if obj['dtype'] == "boolean":
        return Params_Mappings[value]
    elif obj['dtype'] == "string":
        return str(value)
    elif obj['dtype'] == "int":
        if obj['accept_none'] and value == "":
            return None
        else:
            return int(value)
    elif obj['dtype'] == "float":
        if obj['accept_none'] and value == "":
            return None
        else:
            return float(value)


def get_numeric_categorical_columns(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    return num_cols, cat_cols


def check_file_presence(project_id):
    try:
        path1 = os.path.join('src', 'data')
        path2 = os.path.join('src', 'temp_data_store')
        if f"{project_id}.csv" in os.listdir(path1):
            df = pd.read_csv(os.path.join(path1, f'{project_id}.csv'))
            return True, df
        elif re.findall(f"{project_id}.\w+", ",".join(os.listdir(path2))):
            file = (re.findall(f"{project_id}.\w+", ",".join(os.listdir(path2)))[0])
            if file.endswith('csv'):
                print(os.path.join(path2, file))
                df = pd.read_csv(os.path.join(path2, file))
            elif file.endswith('tsv'):
                df = pd.read_csv(os.path.join(path2, file))
            elif file.endswith('json'):
                df = pd.read_json(os.path.join(path2, file))
            elif file.endswith('xlsx'):
                df = pd.read_excel(os.path.join(path2, file))
            else:
                df = False, None
            return True, df
        else:
            return False, None

    except Exception as e:
        return False, None


def remove_temp_files(list_of_path):
    """
    remove_temp_files
    removes temp files from the specified paths
    params : list of paths

    """
    try:
        for path in list_of_path:
            os.remove(path)
            print('removed', path)

    except Exception as e:
        print(e)
