import json
import os
from questions_input import parse_context
from database_setup import DataBase
import pandas as pd

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class DataLoader:
    def __init__(self, dataset):
        self._dataset = dataset
        self.data = self.load_json()
        self.count = 0
        self.db = DataBase()

    def load_json(self):
        with open(f'./dataset/{self._dataset}.json') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def get_index(self):
        return self.count + 1

    def get_table_from_path(self, table_dict, path):
        if self._dataset == 'wikitq':
            table_path_name = parse_context(path)
            try:
                table_path = f'./WikiTableQuestions/csv/{table_path_name[0]}-csv/{table_path_name[1]}.csv'
                table = pd.read_csv(table_path, header=0)
            except:
                table_path = f'./WikiTableQuestions/csv/{table_path_name[0]}-csv/{table_path_name[1]}.table'
                table = pd.read_table(table_path, sep='|')
                table.columns = table.columns.str.strip()
                table = table.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        if self._dataset == 'tabfact':
            table = pd.read_csv(f'TabFact/{path}', header=0)
        if self._dataset == 'finqa':
            table = pd.read_csv(f'FinQA/{path}', header=0)
        if self._dataset == 'ott-qa':
            table_path = f'OTT-QA/data/traindev_tables_tok/{table_dict["table_file_name"]}.json'
            print(f"Loading table from {table_path}")
            # Load the table JSON
            with open(table_path, 'r', encoding='utf-8') as f:
                table_json = json.load(f)

            # Extract headers and data
            headers = [header[0] for header in table_json['header']]

            # Initialize DataFrame with empty lists for each column
            df_data = {header: [] for header in headers}

            # Process each row in the data
            for row in table_json['data']:
                for col_idx, cell in enumerate(row):
                    col_name = headers[col_idx]
                    cell_text = cell[0]  # Base cell text

                    # Process links if they exist and are not empty
                    if len(cell) > 1 and cell[1]:
                        links = cell[1]
                        if links:
                            # Load the request data that contains the expanded text for links
                            try:
                                request_path = f'OTT-QA/data/traindev_request_tok/{table_dict["table_file_name"]}.json'
                                with open(request_path, 'r', encoding='utf-8') as f:
                                    request_json = json.load(f)

                                # For each link in the cell, append its expanded text without limiting length
                                expanded_texts = []
                                for link in links:
                                    if link in request_json:
                                        # No length restriction - use the full text
                                        expanded_text = request_json[link]
                                        expanded_texts.append(expanded_text)

                                if expanded_texts:
                                    cell_text += " | " + " | ".join(expanded_texts)
                            except Exception as e:
                                print(f"Error loading request data for {table_dict['table_file_name']}: {e}")

                    # Add the cell text to the corresponding column
                    df_data[col_name].append(cell_text)

            # Before creating the DataFrame, check and fix column lengths
            max_length = max(len(values) for values in df_data.values())
            for col in df_data:
                current_length = len(df_data[col])
                if current_length < max_length:
                    # Pad with empty strings or None values
                    df_data[col].extend([None] * (max_length - current_length))

            # Now create the DataFrame with columns of equal length
            table = pd.DataFrame(df_data)

        return table

    def get_table(self, index):
        table_dict = self.data[index]
        paragraphs = None
        if 'paragraphs' in table_dict:
            paragraphs = table_dict['paragraphs']
        table_id = table_dict['table_id']
        table = self.get_table_from_path(table_dict, table_dict['table_file_name'])
        table_name = table_dict['table_name']
        question = table_dict['question']
        answer = table_dict['target_value']
        return table_id, table_name, table, paragraphs, question, answer

    def get_question(self):
        table_dict = self.data[self.count]
        question = table_dict['question']
        return question
    '''
    def get_tables_and_description(self):

        tables, table_names, descriptions = self.db.load_from_sqlite(self._dataset)

        return tables, table_names, descriptions
    '''