import os
import re
import time

import pandas as pd
import logging
from database_setup import DataBase

class HybridQAMulti:
    def __init__(self, question, tables, table_names, descriptions, gold_answer, dataset, model_name, model):
        '''
        :param dataset: Dataset used (wikitq, TabFact, TAT, FinQA, ...)
        :param table_names: List of table names [different from path -> e.g. csv-203/777.csv]
        :param tables: List of semi-structured tables as dataframes
        :param question: Question asked on the Tables
        :param gold_answer: Target answer
        :param database: The MySQL database loaded (sqlAlchemy), used to run SQL code
        '''

        # Configure logging
        logging.basicConfig(filename=f'logs/weaver_{model_name}_{dataset}_ab_veifyplan.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        self.database = DataBase()
        self.dataset = dataset
        self.question = question
        self.tables = tables
        self.gold_answer = gold_answer
        self.table_names = table_names
        self.model_name = model_name
        self.llm = model
        self.descriptions_list = descriptions
        self.number_of_rows = 5
        self.cols=[]
        
        self.set_pd()
        for i, table_name in enumerate(table_names):
            logging.info(table_name)
            logging.info(f'\n{self.tables[i].head(20).to_string()}')
        logging.info(question)
        logging.info(gold_answer)

    def set_pd(self):
        # Set options to display entire DataFrame without truncation
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', 1000)  # Adjust width to avoid wrapping
        pd.set_option('display.colheader_justify', 'left')  # Align column headers
        pd.set_option('display.float_format', '{:.2f}'.format)  # Format floats if needed

    def create_coldescription_prompt(self, table, table_name, description):
        table_description_prompt = f"""
            Give me the column name, data type, formatting needed in detail, and column descriptions in detail, for the context of question on the table.\n
            Also, give a small description of the table using table name and table data given.
            Table:
            table name: {table_name}

            
            columns: {table.columns}
            
            table:\n {table.head(self.number_of_rows).to_html()}
            
            Question: {self.question}
            """
        return table_description_prompt

    def get_relevant_cols(self, table, table_name, description): # Get relevant Columns
        prompt = f'''
                These are few columns which are alreasy available in different tables, as these is a multi table question, you need to provide the relevant columns for the question.
                Existing columns: {self.cols}
                Given column description, Table and Question return a list of columns that can be relevant to the solving the question (even if slightly relevant) given the table name and table:
                table name: {table_name}
                Description: {description.to_html()}
                table: \n {table.head(self.number_of_rows).to_html()}
                Question: {self.question}
    
                Example output: [ 'Score', 'Driver']
                Instructions:
                1. Do not provide any explanations, just give the cols as a list
                2. The list will be used to filter the table dataframe directly so take care of that.
                3. Give output in above format only and in case of no relevant columns, return an empty list. []

                Output:

                '''
        output = self.llm.call(prompt)
        print(output)
        return output
        
    def filter_table(self, table, table_name, description):
        #try:
        
        # Use only the first 10 rows of the description table
        # description_sample = description.head(10)
        # cols = self.get_relevant_cols(table, table_name, description_sample)
        cols1 = self.get_relevant_cols(table, table_name, description)
        self.cols.extend(eval(cols1))
        print(table_name)
        print(self.cols)
        
        final_columns = []
        for col in self.cols:
            if col in table.columns:
                final_columns.append(col)       
        description = description[
                            (description['Primary Key / Foreign Key'].astype(str).str.lower() == 'true') |
                            (description['Column Name'].isin(self.cols))
                        ]
        # if not final_columns:
        #     final_columns = table.columns
        filtered_table = table[final_columns]
        return filtered_table, final_columns, description
        # except Exception as e:
        #     print('Error filtering cols:\n')
        #     print(e)
        #     return table, table.columns, description
            
    def create_plan_prompt(self, descriptions_list):
        planner_prompt = ''
        few_shot = ''
        with open(f'prompts/{self.dataset}/few_shot_plan') as fs:
            text = fs.readlines()
        for txt in text:
            few_shot += '\n'+txt
        with open(f'prompts/{self.dataset}/planner_prompt') as f:
            lines = f.readlines()
        for line in lines:
            planner_prompt += line

        # Add few shot in plan -
        planner_prompt += few_shot

        # Final planning prompt
        planner_prompt += f"""
            Solve for this:
            """
            
        # Add all tables and their descriptions to the prompt
        for i, (table_name, table, description) in enumerate(zip(self.table_names, self.tables, descriptions_list)):
            planner_prompt += f"""
            Table {i+1}:
            1. Table name: {table_name}
            {table.head(self.number_of_rows).to_html()}
            2. column_descriptions: {description.to_html()}
            """

        planner_prompt += f"""
            Question: {self.question}

            Only give the step by step plan and remove any extra explanation or Code.
            Output format :
            Step 1: SQL - [Instruction that can be used to write MySQL query]
            Step 2: Either SQL or LLM
            Step 3: ...
            
            Plan:
            """
        return planner_prompt

    def create_execute_prompt(self, plan, descriptions_list):
        execute_prompt = ''

        with open(f'prompts/{self.dataset}/execute_prompt') as f:
            lines = f.readlines()
        for line in lines:
            execute_prompt += line
            
        # Add all tables and their descriptions to the prompt
        for i, (table_name, table, description) in enumerate(zip(self.table_names, self.tables, descriptions_list)):
            execute_prompt += f"""
            Table {i+1}:
            Table name: {table_name}
            Schema: {table.columns}
            Column Descriptions: 
            {description.to_html()}
            
            Table: (This is a Sample table and the actual table can have more rows than below provided)
            {table.head(self.number_of_rows).to_html()}
            """

        execute_prompt += f"""
            Question: {self.question}
            
            Plan:
            {plan}
            
            Give me code for solving the question, and no other explanations. Keep in mind the column data formats (string to appropriate data type, removing extra character, Null values) while writing Mysql code.
            """
        return execute_prompt

    def create_llmstep_prompt(self, llm_step, column_value):
        llm_prompt = f"""
            Given a column and step you need to perform on it -
            Column: {column_value}
            Step to solve the question: {llm_step}
            Question: {self.question}
            
            Instructions: 
            - Do not provide any explanation and Return only a list (separate values by '#') that can be added to a dataframe as a new column in a dataframe.
            - Any value should not be more than 3 words (or each value should be as short as possible).
            - Size of output column Should be same as input column.
            """
        return llm_prompt
        
    # Planning and Code writing
    def column_descriptions(self, cd_prompt):
        result = self.llm.call(cd_prompt)
        return result

    def planner(self, pl_prompt):
        plan = self.llm.call(pl_prompt)
        return plan
        
    def verify_plan(self, plan, descriptions_list):
        with open(f'prompts/{self.dataset}/verify_plan') as f:
            lines = f.readlines()
        verify_prompt = f'''
                Suppose you are an expert planner verification agent. 
                Verify if the given plan will be able to answer the Question asked on these tables.
                '''
                
        # Add all tables and their descriptions to the prompt
        for i, (table_name, table, description) in enumerate(zip(self.table_names, self.tables, descriptions_list)):
            verify_prompt += f"""
                Table {i+1}:
                Table name: {table_name}
                Table: {table.head(self.number_of_rows).to_html()}
                Column descriptions: {description.to_html()}
                """
                
        verify_prompt += f'''
                Question to Answer: {self.question}
                Old Plan:
                {plan}
                Is the given plan correct to answer the Question asked on these tables (check format issues and reasoning steps) should be able to guide the LLM to write correct code and get correct result.
                If the plan is not correct, provide better plan detailed on what needs to be done handling all kinds of values in the column.
                - Check if the MySQL step logic adheres to the column format. (Performs calculations and formatting and filtering in the table)
                - The LLM step's logic will help in getting the correct answer.
                If the original plan is correct then return that plan.
                
                Do not provide with code or other explanations, only the new plan.
                Output format:
                Step 1: Either SQL or LLM - ...
                Step 2: SQL or LLM - ...
                Step 3: SQL ...
    
                As given in original plan.
                '''
        new_plan = self.llm.call(verify_prompt)
        return new_plan
        
    def coder(self, ex_prompt):
        result = self.llm.call(ex_prompt)
        return result

    def verify_code(self, ex_prompt):
        # Add all tables to the prompt
        tables_html = ""
        for i, (table_name, table) in enumerate(zip(self.table_names, self.tables)):
            tables_html += f"""
                    Table {i+1}: {table_name}
                    {table.head(self.number_of_rows).to_html()}
                    """
                    
        ex_prompt += f"""
                    Check if the above code will not throw any error given the tables, make the changes and give new code.
                    Tables: 
                    {tables_html}
                    Do not provide any other explanations.
                    The new code format should be same as the above format for code given.
                    """
        result = self.llm.call(ex_prompt)
        return result

    def get_new_colname(self, step):
        match = re.search(r"(?<=column name: )(\S+)", step, re.IGNORECASE)
        if match:
            new_column_name = match.group(1)
            new_column_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', new_column_name)
            logging.info(f"New column name: {new_column_name}")
            return new_column_name
        else:
            logging.info("New column name not found.")
            return None
            
    def get_tablename(self, step):
        match = re.search(r"(?<=table name: )(\S+)", step, re.IGNORECASE)
        if match:
            name = match.group(1)
            name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', name)
            logging.info(f"Table name: {name}")
            return name
        else:
            logging.info("New column name not found.")
            return None
            
    def get_new_prompt(self, step):
        match = re.search(r'(?<=LLM prompt: )(.*)', step, re.IGNORECASE)
        if match:
            llm_prompt = match.group(1)
            logging.info(f"Prompt: {llm_prompt}")
            return llm_prompt
        else:
            logging.info("LLM prompt not found.")
            return None

    def get_prev_colname(self, step):
        match = re.search(r"(?<=to be used: )(.*)", step, re.IGNORECASE)
        if match:
            new_column_name = match.group(1)
            new_column_name = new_column_name.split(',')
            new_column_name = [col.strip() for col in new_column_name]
            return new_column_name
        else:
            logging.info("Prev column name not found.")
            return None
            
    def run_code(self, text):
        # Use first table as default
        df = self.tables[0]
        tmp_df = df
        table_name = self.table_names[0]
        
        
        # Split the input by steps
        steps = re.split(r"Step \d+", text)

        logging.info('-----------------EXTRACTED CODE------------------\n')
        for num, step in enumerate(steps):
            try:
                if 'SQL' in step[:20] or 'sql' in step[:20]:
                    pattern = r"\b(?:CREATE TABLE|SELECT)\b.*?;"

                    # Find all matches
                    matches = re.findall(pattern, step, re.DOTALL)
                    for match in matches:
                        logging.info('--------------------sql--------------------------')
                        logging.info(match)
                        flag, ele = self.database.execute(table_name, match)
                        if flag == 1:
                            tmp_df = ele
                        elif flag == 2:
                            table_name = ele
                            tmp_df = self.database.get_all_rows(table_name)
                        else:
                            logging.info('----------ERROR----------')
                        logging.info(f'\n{tmp_df.head(20).to_string()}')

                elif 'LLM' in step[:20] or 'llm' in step[:20]:
                    logging.info('-----------------------LLM---------------------------')
                    logging.info(step)
                    cols = self.get_prev_colname(step)
                    final_cols = []
                    for col in cols:
                        if col in df.columns:
                            final_cols.append(col)
                        else:
                            logging.info(f"Column '{col}' not found in the DataFrame.")

                    table_name = self.get_tablename(step)
                    step_prompt = self.get_new_prompt(step)
                    new_col_name = self.get_new_colname(step)
                    tmp_df = self.database.get_all_rows(table_name)

                    
                    batch_size = 10
                    new_col = []

                    for start in range(0, len(tmp_df), batch_size):
                        end = start + batch_size
                        batch_df = tmp_df.iloc[start:end]
                        batch_column_value = batch_df[final_cols]

                        
                        llm_prompt = self.create_llmstep_prompt(step_prompt, batch_column_value)
                        batch_col = self.llm.call(llm_prompt)

                        new_col.extend(batch_col.split('#'))

                    logging.info(f'\nNew col: {new_col}')
                    if len(new_col) == len(tmp_df) and new_col_name is not None:
                        tmp_df[new_col_name] = new_col
                        self.database.upload_table(table_name, tmp_df)
                        logging.info(f'LLM updated the table: {table_name}, with column: {new_col_name}')
                    else:
                        logging.info("LLM created column format is not correct")
                    logging.info(f'\n{tmp_df.head(20).to_string()}')
                else:
                    logging.info('Steps splitted')

                if tmp_df.shape[0] == 0:
                    return df
                df = tmp_df

            except Exception as e:
                logging.info(f'Error in step {num}')
                logging.info(e)
                print(e)
                return df

        return df

    def format_answer(self, final_table, question, gold_answer):
        # Create the prompt for the first API call
        with open(f'eval_prompts/{self.dataset}/extract_answer') as f:
            prompt = f.read()
        prompt += f'''
            Now extract for this:
            Table: {self.table_names[0]}
            {final_table.head(self.number_of_rows).to_html(index=False)}
            Question: {question}
            Answer: 
            '''
        # Call LLM to get the answer
        answer = self.llm.call(prompt)
        logging.info(f"Generated Answer: {answer}")

        # Create the prompt for the second API call to compare the answer with the gold answer
        with open(f'eval_prompts/{self.dataset}/format_answer') as fp:
            answer_formatting_prompt = fp.read()
        answer_formatting_prompt += f'''
                    Solve for this-
                    Answer: {answer}
                    Gold Answer: {gold_answer}
                    Your Output:
                    '''
        answer = self.llm.call(answer_formatting_prompt)

        return answer

    def compare(self, final_table):
        answer = self.format_answer(final_table, self.question, self.gold_answer)
        logging.info(f'Actual answer: {self.gold_answer}')

        # LLM based Answer Match Checking
        if answer == str(self.gold_answer):
            logging.info(f'Model Answer: {answer} and Gold answer {self.gold_answer} match')
            return answer, True
        else:
            logging.info(f'Model Answer: {answer} and Gold answer {self.gold_answer} do not match')
            return answer, False

    def solve(self):
        self.llm.count = 0
        start = time.time()
        
        self.cols = []
        for i in range(len(self.tables)):
            # Filter Table via Columns
            #self.database.upload_table(self.table_names[i], self.tables[i])
            
            filtered_table, cols, description = self.filter_table(self.tables[i], self.table_names[i], self.descriptions_list[i])
            # print(self.table_names[i])
            # print(self.descriptions_list[i])
            self.tables[i] = filtered_table
            self.descriptions_list[i] = description
            logging.info(f'Table {i+1} Relevant Columns: {cols}\n {self.tables[i].head(20)}')
            logging.info(f'Table {i+1} Relevant Descriptions: {description}')
        
        

        # Create a Plan for solving the query with all tables
        plan_prompt = self.create_plan_prompt(self.descriptions_list)
        plan = self.planner(plan_prompt)
        logging.info(f'Plan:\n\n{plan}')

        # Verify if the plan is correct
        verified_plan = self.verify_plan(plan, self.descriptions_list)
        logging.info(f'New Plan:\n\n {verified_plan}')

        # Generate Code for solving the query
        coder_prompt = self.create_execute_prompt(verified_plan, self.descriptions_list)
        code = self.coder(coder_prompt)

        # Verify if the code is correct
        #code = self.verify_code(code)
        #logging.info(f'New code: {code}')

        # Run the code
        final_table = self.run_code(code)
        logging.info(f'Final table:\n{final_table.head(20).to_string(index=False)}')
        end = time.time()
        logging.info(f'Time taken: {int(end-start)}s')
        logging.info(f'Number of API calls: {self.llm.count}')

        # Extract the model's answer from final table
        result, correct = self.compare(final_table)
        logging.info(f'Correct: {correct}')

        return result, correct, verified_plan, code
