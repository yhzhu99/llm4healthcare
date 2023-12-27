import json
import os
from typing import Dict
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
    prediction_format: str,
    visit: int,
):
    basic_features = FEATURES[dataset]['Basic']
    demo_features = FEATURES[dataset]['Demographics']
    lab_features = FEATURES[dataset]['Laboratory']
    features = []
    feature_values = {}
    if dataset == 'mimic-iv':
        for feature in basic_features + demo_features:
            features.append(feature)
            feature_values[feature] = patient[feature].values[:visit + 1]
        for feature in lab_features['Categorical']:
            features.append(feature)
            columns = patient.columns[patient.columns.str.startswith(feature)]
            rows = [columns[res] for res in (patient[columns] == 1.0).values]
            values = [row.item().split('->')[-1] if len(row) > 0 else 'nan' for row in rows]
            feature_values[feature] = values[:visit + 1]
        for feature in lab_features['Numerical']:
            features.append(feature)
            feature_values[feature] = patient[feature].values[:visit + 1]
    elif dataset == 'tjh':
        for feature in basic_features + demo_features + lab_features:
            features.append(feature)
            feature_values[feature] = patient[feature].values[:visit + 1]
    detail = ''
    if prediction_format == '1-1':
        for feature in features:
            detail += f'- {feature}: {feature_values[feature][visit]}\n'
    elif prediction_format == 'N-1_string':
        for feature in features:
            detail += f'- {feature}: \"{", ".join(list(map(str, feature_values[feature])))}\"\n'
    elif prediction_format == 'N-1_list':
        for feature in features:
            detail += f'- {feature}: [{", ".join(list(map(str, feature_values[feature])))}]\n'
    elif prediction_format == 'N-1_batches':
        for i in range(visit + 1):
            detail += f'Visit {i + 1}:\n'
            for feature in features:
                detail += f'- {feature}: {feature_values[feature][i]}\n'
            detail += '\n' if i < visit else ''
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
    
    if config['unit'] is True or config['range'] is True:
        unit_range = UNIT_RANGE_PROMPT
        unit_values = dict(json.load(open(UNIT[dataset])))
        range_values = dict(json.load(open(RANGE[dataset])))
        for feature in unit_values.keys():
            unit_range += f'- {feature}: '
            if config['unit'] is True:
                unit_range = unit_range + unit_values[feature] + ' '
            if config['range'] is True:
                unit_range = unit_range + range_values[feature]
            unit_range += "\n"
    else:
        unit_range = ''
        
    prediction = config['prediction']
    if prediction == '1-1':
        prediction_format = prediction
    elif prediction == 'N-1':
        prediction_format = prediction + f'_{config["format"]}'
    elif prediction == 'N-N':
        pass
    else:
        raise Exception(f'Unknown prediction type: {prediction}')
        
    if config['shot'] is True:
        example = open(EXAMPLE[dataset][prediction_format]).read() + '\n'
    else:
        example = ''
        
    patients = pd.read_csv(DATASETS_PATH[dataset])
    grouped_patients = [item[1] for item in patients.groupby(['PatientID'])]
    if config['forwardfill'] is True:
        filled_values = patients[patients.columns[8:]].median(skipna=True).to_dict()
        for patient in grouped_patients:
            patient.fillna(method='ffill', inplace=True)
            patient.fillna(value=filled_values, inplace=True)
    visit_range = range(max([len(patient) for patient in grouped_patients]))
    visits = max([len(patient) for patient in grouped_patients])
    labels = []
    preds = []
    
    for visit in range(visits - 1, visits):
        label = []
        pred = []
        if prediction_format == '1-1':
            length = '1 visit'
        else:
            length = f'{visit + 1} visit'
            length += 's' if visit > 0 else ''
        for patient in grouped_patients:
            if len(patient) <= visit:
                continue 
            detail = format_input(
                patient=patient,
                dataset=dataset,
                prediction_format=prediction_format,
                visit=visit,
            )
            # with open('prompt.txt', 'w') as f:
            #     f.write(open(USERPROMPT[prediction_format], 'r').read().format(
            #         length=length,
            #         detail=detail,
            #         unit_range=unit_range,
            #         example=example,
            #     ))
            userPrompt = open(USERPROMPT[prediction_format], 'r').read().format(
                length=length,
                detail=detail,
                unit_range=unit_range,
                example=example,
            )
            try:
                result, prompt_token, completion_token = query_llm(
                    model=config['model'],
                    systemPrompt=SYSTEMPROMPT,
                    userPrompt=userPrompt
                )
            except Exception as e:
                logging.info(f'PatientID: {patient.iloc[0]["PatientID"]}, Visit: {visit + 1}:\n')
                logging.info(f'{e}')
                continue
            label.append(float(patient.iloc[visit]['Outcome']))
            prompt_tokens += prompt_token
            completion_tokens += completion_token
            try:
                pred.append(float(result))
            except:
                pred.append(0.501)
                logging.info(f'PatientID: {patient.iloc[0]["PatientID"]}, Visit: {visit + 1}:\n')
                logging.info(f'UserPrompt:{userPrompt}\nResponse: {result}\n')
        labels.append(label)
        preds.append(pred)
    
    logging.info(f'Prompts: {prompt_tokens}, Completions: {completion_tokens}, Total: {prompt_tokens + completion_tokens}\n\n')
    if config['shot'] is True:
        shot = 'oneshot'
    else:
        shot = 'zeroshot'
    dst_path = os.path.join(dst_root, config['model'], prediction_format, shot)
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    pd.to_pickle({
        'config': config,
        'preds': preds,
        'labels': labels,
    }, os.path.join(dst_path, dt.now().strftime("%Y%m%d-%H%M%S") + '.pkl'))

if __name__ == '__main__':
    for config in params:
        run(config)