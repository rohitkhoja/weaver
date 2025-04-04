# IMPORTS
import os
import argparse
import json
import time
from tqdm import tqdm
from weaver import HybridQA
from LLM import Call_OpenAI, Call_Gemini, Call_DeepSeek, Call_Llama
from preprocess_table import clean_table
from DataLoader import DataLoader

import logging
import multiprocessing
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
    elif 'gemini' in model_name:
        model = Call_Gemini(model_name)
    elif 'deepseek' in model_name:
        model = Call_DeepSeek()
    elif 'llama' in model_name:
        model = Call_Llama()
    else:
        print(f'Model not supported: {model_name}')

    return model
def check_baselines(model, table, table_name, question):
    logging.basicConfig(filename=f'baselinelogs/{dataset}_{model.model[:8]}.log', level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    logging.info(f'Table: {table_name}\n{table}\nQuestion: {question}')

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
def Weaver(chunk, dataset, model, model_name):
    dataLoader = DataLoader(dataset)
    n = len(dataLoader)
    print(f'Length of Dataset: {n} Queries')
    start, end = chunk[0], chunk[1]
    total = 0
    result = []
    for index in range(start, end):
        count = dataLoader.get_index()
        dataLoader.count += 1
        table_id, table_name, table, paragraphs, question, answer = dataLoader.get_table(index)
        table_name, table = clean_table(table_name, table)
        print('-' * 40)
        print(index)
        print(table_id)

        hybridqa = HybridQA(table_id, question, table, table_name, paragraphs, answer, dataset, model_name, model)

        model_answer, correct, plan, code = hybridqa.solve()

        output = {'table_id': table_id,
                  'table_name': table_name,
                  'question': question,
                  'plan': plan,
                  'code': code,
                  'model_answer': model_answer,
                  'gold_answer': answer,
                  'is_correct': correct
                  }
        result.append(output)

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

        if correct:
            total += 1
        print(total)
        print(count)
    print('Process Done.')
    return result
def main(n_processes, dataset, model_name):
    model = call_llm(model_name)
    print(model)
    with open(f'./dataset/{dataset}.json') as f:
        data = json.load(f)
    n = len(data)
    data_split = [[i, min(n, i+(n//n_processes))] for i in range(0, n, n // n_processes)]
    results = []

    with multiprocessing.Pool(processes=n_processes) as pool:
        async_results = [pool.apply_async(Weaver, args=(
            chunk,
            dataset,
            model,
            model_name
        )) for chunk in data_split]
        print(async_results[0])
        results.extend([r.get() for r in async_results])

        load_result_into_json(results, dataset, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # File path or name
    parser.add_argument('--dataset', type=str, default='ott-qa',
                        choices=['wikitq', 'tabfact', 'tat', 'finqa', 'large', 'wikitq_processed', 'ott-qa'])

    parser.add_argument('--model', type=str, default='gpt-4o', # don't put llama as string in deepseek model name
                        choices=['gpt-4o-mini', 'gpt-4o', 'gemini-2.0-flash-exp', 'llama-70B', 'deepseek-r1-distill'])

    parser.add_argument('--save_dir', type=str, default='results/')

    parser.add_argument('--n_processes', type=str, default=1)

    args = parser.parse_args()
    n_processes = int(args.n_processes)
    dataset = args.dataset.lower()
    model = args.model
    main(n_processes, dataset, model)