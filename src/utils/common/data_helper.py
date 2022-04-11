from flask import session
import pandas as pd
import csv
import json
import openpyxl
from src.utils.databases.mongo_helper import MongoHelper
import os
from src.utils.common.common_helper import read_config
from loguru import logger
from from_root import from_root

config_args = read_config("./config.yaml")

log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")

updated_time = None
mongodb = MongoHelper()


def get_filename():
    try:
        project_name = session.get('project_name')
        filename = os.path.join(os.path.join('src', 'data'), f"{project_name}.csv")
        logger.info(f"filename: {filename} obtained successfully!")
        return filename

    except Exception as e:
        logger.error(f"{e} occurred in Get Filename of Data Helper!")
    finally:
        pass


def load_data():
    try:
        filename = get_filename()

        # if not os.path.exists(filename):
        #     mongodb = MongoHelper()
        #     mongodb.download_collection_data()

        df = pd.read_csv(filename)
        logger.info(f"DataFrame loaded successfully!")
        return df

    except Exception as e:
        logger.error(f"{e} occurred in Load Data of Data Helper!")


def update_data(df):
    try:
        filename = get_filename()
        os.remove(filename)
        df.to_csv(filename, index=False)
        logger.info(f"DataFrame updated successfully!")
        return df

    except Exception as e:
        logger.error(f"{e} occurred in Update Data of Data Helper!")
    finally:
        pass


def to_tsv():
    try:
        filename = get_filename()
        df = pd.read_csv(filename)
        filename = filename.rsplit('.', 1)[0]
        df.to_csv(filename + '.tsv', sep='\t')
        logger.info(f"DataFrame converted to TSV successfully!")
    except Exception as e:
        logger.error(f"{e} occurred in Convert to TSV of Data Helper!")
    finally:
        pass


def to_excel():
    try:
        filename = get_filename()
        df = pd.read_csv(filename)
        filename = filename.rsplit('.', 1)[0]

        # saving xlsx file
        GFG = pd.ExcelWriter(filename + '.xlsx')
        df.to_excel(GFG, index=False, header=True)

        GFG.save()
        logger.info(f"DataFrame converted to Excel successfully!")
    except Exception as e:
        logger.error(f"{e} occurred in Convert to Excel of Data Helper!")
    finally:
        pass


def to_json():
    try:
        filename = get_filename()
        df = pd.read_csv(filename)
        df = df.to_json(orient='records', lines=True)
        logger.info(f"DataFrame converted to JSON successfully!")
        return df

    except Exception as e:
        logger.error(f"{e} occurred in Convert to JSON of Data Helper!")
    finally:
        pass


def csv_to_json(csvFilePath, jsonFilePath=None):
    try:
        jsonArray = []

        # read csv file
        with open(csvFilePath, encoding='utf-8') as csvf:
            # load csv file data using csv library's dictionary reader
            csvReader = csv.DictReader(csvf)

            # convert each csv row into python dict
            for row in csvReader:
                # add this python dict to json array
                jsonArray.append(row)

        # convert python jsonArray to JSON String and write to file
        # with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        #     jsonString = json.dumps(jsonArray, indent=4)
        #     jsonf.write(jsonString)

        jsonString = json.dumps(jsonArray, indent=4)
        logger.info(f"DataFrame converted to JSON successfully!")
        return jsonString

    except Exception as e:
        logger.error(f"{e} occurred in Convert to JSON of Data Helper!")


def csv_to_excel(csv_file=None, excel_file=None):
    try:
        csv_file = csv_file
        excel_file = csv_file.rsplit('.', 1)[0]

        csv_data = []
        with open(csv_file) as file_obj:
            reader = csv.reader(file_obj)
            for row in reader:
                csv_data.append(row)

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        for row in csv_data:
            sheet.append(row)
        workbook.save(excel_file)
        logger.info(f"DataFrame converted to Excel successfully!")
        return workbook

    except Exception as e:
        logger.error(f"{e} occurred in Convert to Excel of Data Helper!")
