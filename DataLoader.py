import json
import os
from questions_input import parse_context
from database_setup import DataBase
import pandas as pd
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
        return self.count+1

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
        return table
    
    def get_table(self):
        table_dict = self.data[self.count]

        table_id = table_dict['table_id']
        table = self.get_table_from_path(table_dict, table_dict['table_file_name'])
        table_name = table_dict['table_name']
        question = table_dict['question']
        answer = table_dict['target_value']
        return table_id, table_name, table, question, answer
    
    def get_question(self):
        table_dict = self.data[self.count]
        question = table_dict['question']
        return question
    
    def get_tables_and_description(self):
        base_path = './dataset/california_schools'
        csv_output_path = f'{base_path}/csv_output'
        description_path = f'{base_path}/database_description'
        tables, table_names, descriptions = self.db.load_from_sqlite(self._dataset)

        # csv_files = [f for f in os.listdir(csv_output_path) if f.endswith('.csv')]
        # for csv_file in csv_files:
        #     table_name = os.path.splitext(csv_file)[0]
            
        #     table_path = f'{csv_output_path}/{csv_file}'
        #     try:
        #         table = pd.read_csv(table_path)
        #         tables.append(table)
        #         table_names.append(table_name)
                
        #         description_file_path = f'{description_path}/{table_name}.csv'
        #         if os.path.exists(description_file_path):
        #             with open(description_file_path, 'r') as desc_file:
        #                 description_content = desc_file.read()
        #                 descriptions.append(description_content)
        #         else:
        #             descriptions.append(f"No description available for {table_name}")
                    
        #     except Exception as e:
        #         print(f"description")
        
        return tables, table_names, descriptions
