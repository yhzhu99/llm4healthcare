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

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
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
    patient: List,
    dataset: str,
    form: str,
    features: List[str],
):
    feature_values = {}
    numerical_features = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
    categorical_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']
    if dataset == 'mimic-iv':
        for i, feature in enumerate(features):
            if feature in numerical_features:
                feature_values[feature] = [str(visit[i]) for visit in patient]
        for categorical_feature in categorical_features:
            indexes = [i for i, f in enumerate(features) if f.startswith(categorical_feature)]
            feature_values[categorical_feature] = []
            for visit in patient:
                values = [visit[i] for i in indexes]
                if sum(values) == 0:
                    pass
                else:
                    for i in indexes:
                        if visit[i] == 1:
                            feature_values[categorical_feature].append(features[i].split('->')[-1])
                            break
        features = categorical_features + numerical_features
    elif dataset == 'tjh':
        for i, feature in enumerate(features):
            feature_values[feature] = [str(visit[i]) for visit in patient]

    detail = ''
    if form == 'string':
        for feature in features:
            detail += f'- {feature}: \"{", ".join(feature_values[feature])}\"\n'
    elif form == 'list':
        for feature in features:
            detail += f'- {feature}: [{", ".join(feature_values[feature])}]\n'
    elif form == 'batches':
        for i, visit in enumerate(patient):
            detail += f'Visit {i + 1}:\n'
            for feature in features:
                detail += f'- {feature}: {feature_values[feature][i]}\n'
            detail += '\n'
    return detail

def run(
    config: Dict,
    output_logits: bool=True,
    output_prompts: bool=False,
    logits_root: str='logits',
    prompts_root: str='logs',
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
    
    nshot = config['n_shot']
    # if config['shot'] is True:
    #     example = open(EXAMPLE[dataset][prediction_format]).read() + '\n'
    # else:
    #     example = ''
    example = ''
    
    dataset_path = f'datasets/{dataset}/processed/fold_llm'
    task = config['task']
    assert task in ['outcome', 'los', 'readmission'], f'Unknown task: {task}'
    
    xs = pd.read_pickle(os.path.join(dataset_path, 'test_x.pkl'))[:5]
    ys = pd.read_pickle(os.path.join(dataset_path, 'test_y.pkl'))[:5]
    pids = pd.read_pickle(os.path.join(dataset_path, 'test_pid.pkl'))[:5]
    features = pd.read_pickle(os.path.join(dataset_path, 'all_features.pkl'))[2:]
    record_times = pd.read_pickle(os.path.join(dataset_path, 'test_x_record_times.pkl'))
    labels = []
    preds = []
    
    if output_logits:
        logits_path = os.path.join(logits_root, dataset, task, config['model'])
        Path(logits_path).mkdir(parents=True, exist_ok=True)
        sub_dst_name = f'{form}_{str(nshot)}shot'
        if config['unit'] is True:
            sub_dst_name += '_unit'
        if config['reference_range'] is True:
            sub_dst_name += '_range'
        sub_logits_path = os.path.join(logits_path, sub_dst_name)
        Path(sub_logits_path).mkdir(parents=True, exist_ok=True)
    if output_prompts:
        prompts_path = os.path.join(prompts_root, dataset, task, config['model'])
        Path(prompts_path).mkdir(parents=True, exist_ok=True)
        sub_dst_name = f'{form}_{str(nshot)}shot'
        if config['unit'] is True:
            sub_dst_name += '_unit'
        if config['reference_range'] is True:
            sub_dst_name += '_range'
        sub_prompts_path = os.path.join(prompts_path, sub_dst_name)
        Path(sub_prompts_path).mkdir(parents=True, exist_ok=True)
 
    for x, y, pid, record_time in zip(xs, ys, pids, record_times):
        length = len(x)
        sex = 'male' if x[0][0] == 1 else 'female'
        age = x[0][1]
        x = [visit[2:] for visit in x]
        detail = format_input(
            patient=x,
            dataset=dataset,
            form=form,
            features=features,
        )
        userPrompt = USERPROMPT.format(
            INPUT_FORMAT_DESCRIPTION=INPUT_FORMAT_DESCRIPTION[form],
            TASK_DESCRIPTION_AND_RESPONSE_FORMAT=TASK_DESCRIPTION_AND_RESPONSE_FORMAT[task],
            UNIT_RANGE_CONTEXT=unit_range,
            EXAMPLE=example,
            SEX=sex,
            AGE=age,
            LENGTH=length,
            RECORD_TIME_LIST=', '.join(list(map(str, record_time))),
            DETAIL=detail,
            RESPONSE_FORMAT=RESPONSE_FORMAT[task],
        )
        if output_prompts:
            with open(os.path.join(sub_prompts_path, f'{round(pid)}.txt'), 'w') as f:
                f.write(userPrompt)
        if output_logits:
            try:
                result, prompt_token, completion_token = query_llm(
                    model=config['model'],
                    systemPrompt=SYSTEMPROMPT,
                    userPrompt=userPrompt
                )
            except Exception as e:
                # logging.info(f'PatientID: {patient.iloc[0]["PatientID"]}:\n')
                logging.info(f'{e}')
                continue
            prompt_tokens += prompt_token
            completion_tokens += completion_token
            if task == 'outcome':
                label = y[0][0]
            elif task == 'readmission':
                label = y[0][2]
            elif task == 'los':
                pass
            try:
                pred = float(result)
            except:
                pred = 0.501
                if result == 'I do not know':
                    pass
                else:
                    logging.info(f'PatientID: {round(pid)}:\nResponse: {result}\n')
            pd.to_pickle({
                'prompt': userPrompt,
                'label': label,
                'pred': pred,
            }, os.path.join(sub_logits_path, f'{round(pid)}.pkl'))
            labels.append(label)
            preds.append(pred)
    
    if output_logits:
        logging.info(f'Prompts: {prompt_tokens}, Completions: {completion_tokens}, Total: {prompt_tokens + completion_tokens}\n\n')    
        pd.to_pickle({
            'config': config,
            'preds': preds,
            'labels': labels,
        }, os.path.join(logits_path, dt.now().strftime("%Y%m%d-%H%M%S") + '.pkl'))

if __name__ == '__main__':
    for config in params:
        run(config, output_logits=True, output_prompts=False)