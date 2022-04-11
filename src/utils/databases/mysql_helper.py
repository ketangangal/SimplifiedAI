import mysql.connector as connector
import mysql.connector.pooling
from src.utils.common.common_helper import read_config
import os
from loguru import logger
from from_root import from_root

config_path = os.path.join(from_root(), "config.yaml")
config_args = read_config(config_path)
log_path = os.path.join(from_root(), config_args['logs']['logger'], config_args['logs']['generallogs_file'])
logger.add(sink=log_path, format="[{time:YYYY-MM-DD HH:mm:ss.SSS} - {level} - {module} ] - {message}", level="INFO")


"""
    [summary]
        Mysql Helper for all operations related to mysql
    Returns:
        [type]: [None]
"""


class MySqlHelper:
    connection_obj = None

    def __init__(self, host, port, user, password, database):
        """
        [summary]: Constructor
        Args:
            host ([type]): [description]
            port ([type]): [description]
            user ([type]): [description]
            password ([type]): [description]
            database ([type]): [description]
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.isconnected = False
        self.connection = None

        dbconfig = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }
        self.pool = self.create_pool(dbconfig, "auto_neuron_pool", 3)
        # self.connect_todb()

    @staticmethod
    def get_connection_obj():
        try:
            if MySqlHelper.connection_obj is None:

                host = config_args['secrets']['host']
                port = config_args['secrets']['port']
                user = config_args['secrets']['user']
                password = config_args['secrets']['password']
                database = config_args['secrets']['database']

                obj = MySqlHelper(host, port, user, password, database)
                MySqlHelper.connection_obj = obj
                return obj
            else:
                return MySqlHelper.connection_obj
        except Exception as e:
            logger.error(e)

    def connect_todb(self):
        self.connection = connector.connect(host=self.host, port=self.port, user=self.user,
                                            password=self.password, database=self.database, use_pure=True)
        self.isconnected = True

    def create_pool(self, dbconfig, pool_name="mypool", pool_size=3):
        """[summary]
                Create a connection pool, after created, the request of connecting 
                MySQL could get a connection from this pool instead of request to 
                create a connection.
        Args:
            pool_name (str, optional): [description]. Defaults to "mypool".
            pool_size (int, optional): [description]. Defaults to 3.

        Returns:
            [type]: [description]
        """
        pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name=pool_name,
            pool_size=pool_size,
            pool_reset_session=True,
            **dbconfig)
        return pool

    def close(self, conn, cursor):
        """
        A method used to close connection of mysql.
        :param conn: 
        :param cursor: 
        :return: 
        """
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

    def fetch_all(self, query):
        """
        [summary]: This function will return all record from table
        Args:
            query ([type]): [Select tabel query]

        Returns:
            [type]: [description]
        """
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor()

            cursor.execute(query)
            data = cursor.fetchall()
            return data

        except connector.Error as error:
            logger.error("Error: {}".format(error))

        finally:
            self.close(conn, cursor)

    def fetch_one(self, query):
        """
        [summary]:This method return single record from table
        Args:
            query ([type]): [Query to execute]

        Returns:
            [type]: [Data]
        """
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            data = cursor.fetchone()
            return data

        except connector.Error as error:
            logger.error("Error: {}".format(error))

        finally:
            self.close(conn, cursor)


    def delete_record(self, query):
        """
        [summary]: Function to delete record from table single or multiple
        Args:
            query ([type]): [Query to execute]

        Returns:
            [type]: [No of row effected]
        """
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            rowcount = cursor.rowcount
            conn.commit()
            self.close(conn, cursor)
            return rowcount

        except connector.Error as error:
            logger.error("Error: {}".format(error))

            
    def update_record(self, query):
        """
        [summary]: Function to delete record from table single or multiple
        Args:
            query ([type]): [Query to execute]

        Returns:
            [type]: [No of row effected]
        """
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            rowcount = cursor.rowcount
            return rowcount

        except connector.Error as error:
            logger.error("Error: {}".format(error))

        finally:
            conn.commit()
            self.close(conn, cursor)

    def insert_record(self, query):
        """
        [summary]:Insert record into table
        Args:
            query ([type]): [Query to execute]

        Returns:
            [type]: [1 if row inserted or 0 if not]
        """
        conn = None
        cursor = None
        try:
            conn = self.pool.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            rowcount = cursor.rowcount
            conn.commit()
            return rowcount

        except connector.Error as error:
            logger.error("Error: {}".format(error))

        finally:
            self.close(conn, cursor)
