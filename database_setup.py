import time

from sqlalchemy import create_engine, inspect, text
import pandas as pd
import re
from urllib.parse import quote
import sqlite3

class DataBase():
    def __init__(self):
        self.db_user = "root"
        self.db_password = quote("######")
        self.db_host = "localhost"
        self.db_port = 3306
        self.db_name = "text2sql"

        self.engine = create_engine(f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}")

        self.inspector = inspect(self.engine)

    def table_not_exist(self,table_name):
        return table_name not in self.inspector.get_table_names()

    def upload_table(self, table_name, df):
        try:
            df.to_sql(table_name, con=self.engine, if_exists="replace", index=False)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print('table_name: ', table_name, df.columns)
            print(df.head())

    def close_connection(self):
        self.engine.dispose()

    def execute(self, table_name, query):
        with self.engine.connect() as connection:
            query_type = query.strip().split()[0].upper()
            if query_type == 'SELECT':
                result = connection.execute(text(query))

                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return 1, df
            elif query_type == 'CREATE':
                pattern = r'(?i)\bCREATE\s+TABLE\s+[`"]?([\w]+(?:\.[\w]+)?)[`"]?'
                match = re.search(pattern, query)
                if match:
                    table_name = match.group(1)
                else:
                    print(f'Table name cannot be extracted from: \n{query}\n')
                    return
                table_name = table_name.lower()

                connection.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
                time.sleep(1)
                result = connection.execute(text(query))
                connection.commit()

                return 2, table_name
            else:
                print(f'Error in Query: {query_type}')
                return 0, None

    def get_all_rows(self, table_name):
        with self.engine.connect() as connection:
            query = f'Select * from {table_name}'
            result = connection.execute(text(query))
            connection.commit()


            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        
    def load_from_sqlite(self, dataset):
        sqlite_file_path = f'./dataset/{dataset}/{dataset}.sqlite'
        description_path = f'./dataset/{dataset}/database_description'
    
        sqlite_conn = sqlite3.connect(sqlite_file_path)
        
        sqlite_cursor = sqlite_conn.cursor()
        sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = sqlite_cursor.fetchall()
        tables_list = []
        table_names_list = []
        description_list = []

        for table_tuple in tables:
            table_name = table_tuple[0]
            
            if table_name.startswith('sqlite_'):
                continue
                
            print(f"Importing table: {table_name}")
            
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", sqlite_conn)
            #print(table_name)
            df_description = pd.read_csv(f'{description_path}/{table_name}.csv', header=0)
            #print(df_description)
            self.upload_table(table_name, df)
            tables_list.append(df)
            table_names_list.append(table_name)
            description_list.append(df_description)

        sqlite_conn.close()
        print("SQLite database import complete")    
        return tables_list, table_names_list, description_list
    
