# IMPORTS
import os
import argparse
import json
import time
from tqdm import tqdm
from weaverMulti import HybridQAMulti
from LLM import Call_OpenAI
from preprocess_table import clean_table
from DataLoader import DataLoader
import logging

def load_result_into_json(result, dataset, model_name, baseline=''):
    file_path = f'{baseline}Results/{model_name}_{dataset}.json'

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump([], file)
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except (json.JSONDecodeError, ValueError):
        # Handle file corruption or invalid JSON structure
        data = []
    data.append(result)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    return

def call_llm(model_name):
    if 'gpt' in model_name:
        model = Call_OpenAI(model_name)
    # elif 'gemini' in model_name:
    #     model = Call_Gemini(model_name)
    # elif 'deepseek' in model_name:
    #     model = Call_DeepSeek()
    # elif 'llama' in model_name:
    #     model = Call_Llama()
    else:
        print(f'Model not supported: {model_name}')

    return model
def check_baselines(model, table, table_name, question):
    logging.basicConfig(filename=f'baselinelogs/{dataset}_{model.model[:8]}.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    logging.info(f'Table: {table_name}\n{table.head(20)}\nQuestion: {question}')
    # Check baseline Only LLM
    prompt = f'''
            Given the Table and Question, answer the question from the table directly. The answer should be as short as possible.
            Table name: {table_name}
            Table: {table}    
            Question: {question}
            Your answer:
            '''
    result = model.call(prompt)
    return result

def format_answer(model, answer, gold_answer):
    logging.basicConfig(filename=f'baselinelogs/{dataset}_{model.model}.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    answer_formatting_prompt = f"""
            You will be given Answer and Gold Answer, you have to Convert the answer into a format of gold answer given above, if the content or meaning is same they should be same.
            Answer: {answer}
            Gold Answer: {gold_answer}
            Output:
            """
    logging.info(f'Generate Answer: {answer}')

    answer = model.call(answer_formatting_prompt)
    logging.info(f'Model Answer: {answer}')
    logging.info(f'Gold Answer: {gold_answer}')
    if answer == str(gold_answer):
        logging.info('Correct: True')
        return answer, True
    logging.info('Correct: False')
    return answer, False
def main(dataset, model_name):
    dataLoader = DataLoader(dataset)
    n = len(dataLoader)
    model = call_llm(model_name)
    print(model)
    total = 0
    
    # for description in descriptions:
    #     print(description)
    for index in range(n):
        tables, table_names, descriptions = dataLoader.get_tables_and_description()
        count = dataLoader.get_index()
        question = dataLoader.get_question()
        dataLoader.count += 1
        #table_name, table = clean_table(table_name, table)

        # Weaver - Our Approach
        print('-' * 40)
        print(count)

        hybridqa = HybridQAMulti(question, tables, table_names, descriptions, ' ', dataset, model_name, model)
                            

        model_answer, correct, plan, code = hybridqa.solve()

        result = {
                  'question':question,
                  'plan': plan,
                  'code': code,
                  'model_answer': model_answer,
                  'gold_answer': ' ',
                  'is_correct': correct
                  }
        # load_result_into_json(result, dataset, model_name)

        # LLM Baseline
        # model_answer = check_baselines(model, table, table_name, question)
        # model_answer, correct = format_answer(model, model_answer, answer)
        # result = {
        #     'table_id': table_id,
        #     'table_name': table_name,
        #     'question':question,
        #     'model_answer': model_answer,
        #     'gold_answer': answer,
        #     'is_correct': correct
        # }

        print(f'Correct: {correct}')
        load_result_into_json(result,dataset, model_name, 'baseline')
        if correct:
            total+=1
        print(total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # File path or name
    parser.add_argument('--dataset', type=str, default='california_schools',
                        choices=['wikitq', 'tabfact', 'tat', 'finqa', 'california_school'])

    parser.add_argument('--model', type=str, default='gpt-4o', # don't put llama as string in deepseek model name
                        choices=['gpt-4o-mini', 'gpt-4o', 'gemini-2.0-flash-exp', 'llama-70B', 'deepseek-r1-distill'])

    parser.add_argument('--save_dir', type=str, default='results/')
    args = parser.parse_args()
    dataset = args.dataset.lower()
    model = args.model
    main(dataset, model)

