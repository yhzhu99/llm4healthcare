import json
import os
from typing import Dict, List
from pathlib import Path
from datetime import datetime as dt
import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI
import pandas as pd

from config.config import *
from prompts.prompt import *

logging.basicConfig(filename=f'logs/{dt.now().strftime("%Y%m%d")}.log', level=logging.INFO, format='%(asctime)s\n%(message)s')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_llm(
    model: str,
    systemPrompt: str,
    userPrompt: str,
):
    client = OpenAI(api_key=OPENAI_API_KEY)
    result = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': systemPrompt},
            {'role': 'user', 'content': userPrompt},
        ],
    )
    return result.choices[0].message.content, result.usage.prompt_tokens, result.usage.completion_tokens

def format_input(
    patient: pd.DataFrame,
    dataset: str,
    form: str,
    features: List[str]
):
    # basic_features = FEATURES[dataset]['Basic']
    # demo_features = FEATURES[dataset]['Demographics']
    # lab_features = FEATURES[dataset]['Laboratory']
    # features = []
    # feature_values = {}
    # if dataset == 'mimic-iv':
    #     for feature in basic_features + demo_features:
    #         features.append(feature)
    #         feature_values[feature] = patient[feature].values[:visit + 1]
    #     for feature in lab_features['Categorical']:
    #         features.append(feature)
    #         columns = patient.columns[patient.columns.str.startswith(feature)]
    #         rows = [columns[res] for res in (patient[columns] == 1.0).values]
    #         values = [row.item().split('->')[-1] if len(row) > 0 else 'nan' for row in rows]
    #         feature_values[feature] = values[:visit + 1]
    #     for feature in lab_features['Numerical']:
    #         features.append(feature)
    #         feature_values[feature] = patient[feature].values[:visit + 1]
    # elif dataset == 'tjh':
    #     for feature in basic_features + demo_features + lab_features:
    #         features.append(feature)
    #         feature_values[feature] = patient[feature].values[:visit + 1]
    detail = ''
    if dataset == 'tjh':
        if form == 'string':
            for i, name in enumerate(features):
                detail += f'- {name}: \"{", ".join([str(visit[2 + i]) for visit in patient])}\"\n'
        elif form == 'batches':
            for i, visit in enumerate(patient):
                detail += f'Visit {i+1}:\n'
                for j, name in enumerate(features):
                    detail += f'- {name}: \"{visit[2 + j]}\"\n'
                detail += '\n'
    return detail

def run(
    config: Dict,
    dst_root: str='logits',
):
    logging.info(f'Running config: {config}\n\n')
    
    dataset = config['dataset']
    assert dataset in ['tjh', 'mimic-iv'], f'Unknown dataset: {dataset}'
    
    prompt_tokens = 0
    completion_tokens = 0
    
    if config['unit'] is True or config['reference_range'] is True:
        unit_range = ''
        unit_values = dict(json.load(open(UNIT[dataset])))
        range_values = dict(json.load(open(REFERENCE_RANGE[dataset])))
        for feature in unit_values.keys():
            unit_range += f'- {feature}: '
            if config['unit'] is True:
                unit_range = unit_range + unit_values[feature] + ' '
            if config['reference_range'] is True:
                unit_range = unit_range + range_values[feature]
            unit_range += '\n'
    else:
        unit_range = ''
        
    form = config['form']
    assert form in ['string', 'batches', 'list'], f'Unknown form: {form}'
    
    # if config['shot'] is True:
    #     example = open(EXAMPLE[dataset][prediction_format]).read() + '\n'
    # else:
    #     example = ''
    example = ''
    
    dataset_path = f'datasets/{dataset}/processed/fold_llm'
    task = config['task']
    assert task in ['outcome', 'los', 'readmission'], f'Unknown task: {task}'
    
    xs = pd.read_pickle(os.path.join(dataset_path, 'test_x.pkl'))
    ys = pd.read_pickle(os.path.join(dataset_path, 'test_y.pkl'))
    features = pd.read_pickle(os.path.join(dataset_path, 'all_features.pkl'))[2:]
    record_times = pd.read_pickle(os.path.join(dataset_path, 'test_x_record_times.pkl'))
    labels = []
    preds = []
 
    for x, y, record_time in zip(xs, ys, record_times):
    # patient = xs[0]
    # record_time = record_times[0]
        length = len(x)
        sex = 'male' if x[0][0] == 1 else 'female'
        age = x[0][1]
        detail = format_input(
            patient=x,
            dataset=dataset,
            form=form,
            features=features,
        )
        userPrompt = open('prompts/template.txt', 'r').read().format(
            INPUT_FORMAT_DESCRIPTION=INPUT_FORMAT_DESCRIPTION[form],
            TASK_DESCRIPTION_AND_RESPONSE_FORMAT=TASK_DESCRIPTION_AND_RESPONSE_FORMAT[task],
            UNIT_RANGE_CONTEXT=unit_range,
            EXAMPLE=example,
            SEX=sex,
            AGE=age,
            LENGTH=length,
            RECORD_TIME_LIST=', '.join(record_time),
            DETAIL=detail,
        )
    # with open('prompt.txt', 'w') as f:
    #     f.write(userPrompt)
        try:
            result, prompt_token, completion_token = query_llm(
                model=config['model'],
                systemPrompt=SYSTEMPROMPT,
                userPrompt=userPrompt
            )
        except Exception as e:
            # logging.info(f'PatientID: {patient.iloc[0]["PatientID"]}:\n')
            # logging.info(f'{e}')
            continue
        prompt_tokens += prompt_token
        completion_tokens += completion_token
        if task == 'outcome':
            labels.append(y[0][0])
        try:
            preds.append(float(result))
        except:
            preds.append(0.501)
            # logging.info(f'PatientID: {patient.iloc[0]["PatientID"]}:\n')
            # logging.info(f'UserPrompt:{userPrompt}\nResponse: {result}\n')
    
    logging.info(f'Prompts: {prompt_tokens}, Completions: {completion_tokens}, Total: {prompt_tokens + completion_tokens}\n\n')
    
    dst_path = os.path.join(dst_root, dataset, config['model'], form)
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    pd.to_pickle({
        'config': config,
        'preds': preds,
        'labels': labels,
    }, os.path.join(dst_path, dt.now().strftime("%Y%m%d-%H%M%S") + '.pkl'))

if __name__ == '__main__':
    for config in params:
        run(config)