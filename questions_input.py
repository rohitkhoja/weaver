from collections import defaultdict
import re
import pandas as pd

def parse_context(context):
    match = re.match(r'csv/(\d+)-csv/(\d+).csv$', context)
    batch_id, data_id = match.groups()
    return batch_id, data_id

def question_set(file_name):
    table_to_examples = []
    count = 0
    file_path = './WikiTableQuestions/csv/200-csv/'+file_name
    with open(file_path, 'r') as fin:
        header = fin.readline().strip().split('\t')
        for line in fin:
            line = dict(zip(header, line.strip().split('\t')))

            context = parse_context(line['context'])
            file = [context[0], context[1]]
            line['context'] = file
            line['targetValue'] = line['targetValue'].split('|')
            if len(line['targetValue']) == 1:
                line['targetValue'] = line['targetValue'][0]

            table_to_examples.append(line)
            count += 1

    df = pd.DataFrame(table_to_examples)
    return df