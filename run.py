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
import google.generativeai as genai
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
import pandas as pd

from config.config import *
from prompts.prompt import *

logging.basicConfig(filename=f'logs/{dt.now().strftime("%Y%m%d")}.log', level=logging.INFO, format='%(asctime)s\n%(message)s')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_llm(
    model: str,
    llm,
    systemPrompt: str,
    userPrompt: str,
):
    if model in ['gpt-4-1106-preview', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-1106']:
        try:
            result = llm.chat.completions.create(
                model=model,
                messages=[
                    {'role': 'system', 'content': systemPrompt},
                    {'role': 'user', 'content': userPrompt},
                ],
            )
        except Exception as e:
            logging.info(f'{e}')
            raise e
        return result.choices[0].message.content, result.usage.prompt_tokens, result.usage.completion_tokens
    elif model in ['gemini-pro']:
        try:
            response = llm.generate_content(systemPrompt + userPrompt)
        except Exception as e:
            logging.info(f'{e}')
            raise e
        return response.text, 0, 0
    elif model in ['llama2:70b']:
        try:
            response = llm(systemPrompt + userPrompt)
        except Exception as e:
            logging.info(f'{e}')
            raise e
        return response, 0, 0

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
                if 1 not in values:
                    feature_values[categorical_feature].append('unknown')
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
                value = feature_values[feature][i] if i < len(feature_values[feature]) else 'unknown'
                detail += f'- {feature}: {value}\n'
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
    
    prompt_tokens = 0
    completion_tokens = 0
    
    dataset = config['dataset']
    assert dataset in ['tjh', 'mimic-iv'], f'Unknown dataset: {dataset}'
    task = config['task']
    assert task in ['outcome', 'los', 'readmission', 'multitask'], f'Unknown task: {task}'
    time = config['time']
    if time == 0:
        time_des = 'upon-discharge'
    elif time == 1:
        time_des = '1month'
    elif time == 2:
        time_des = '6months'
    else:
        raise ValueError(f'Unknown time: {time}')
    
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
    if nshot == 0:
        example = ''
    elif nshot == 1:
        example = f'Here is an example of input information:\n'
        example += 'Example #1:'
        example += EXAMPLE[dataset][task][0] + '\n'
    else:
        example = f'Here are {nshot} examples of input information:\n'
        for i in range(nshot):
            example += f'Example #{i + 1}:'
            example += EXAMPLE[dataset][task][i] + '\n'
            
    if config.get('prompt_engineering') is True:
        example = COT[dataset]
        response_format = RESPONSE_FORMAT['cot']
    else:
        response_format = RESPONSE_FORMAT[task]
        
    if task == 'outcome':
        task_description = TASK_DESCRIPTION_AND_RESPONSE_FORMAT[task][time_des]
    else:
        task_description = TASK_DESCRIPTION_AND_RESPONSE_FORMAT[task]
    
    model = config['model']
    if model in ['gpt-4-1106-preview', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-1106']:
        llm = OpenAI(api_key=OPENAI_API_KEY)
    elif model in ['gemini-pro']:
        genai.configure(api_key=GOOGLE_API_KEY, transport='rest')
        llm = genai.GenerativeModel(model)
    elif model in ['llama2:70b']:
        llm = Ollama(model=model)
    else:
        raise ValueError(f'Unknown model: {model}')
    
    dataset_path = f'datasets/{dataset}/processed/fold_llm'
    impute = config.get('impute', 1)
    if impute in [1, 2]:
        xs = pd.read_pickle(os.path.join(dataset_path, 'test_x.pkl'))
    else:
        xs = pd.read_pickle(os.path.join(dataset_path, 'test_x_no_impute.pkl'))
    ys = pd.read_pickle(os.path.join(dataset_path, 'test_y.pkl'))
    pids = pd.read_pickle(os.path.join(dataset_path, 'test_pid.pkl'))
    features = pd.read_pickle(os.path.join(dataset_path, 'all_features.pkl'))[2:]
    record_times = pd.read_pickle(os.path.join(dataset_path, 'test_x_record_times.pkl'))
    labels = []
    preds = []
    
    if output_logits:
        logits_path = os.path.join(logits_root, dataset, task, model)
        Path(logits_path).mkdir(parents=True, exist_ok=True)
        sub_dst_name = f'{form}_{str(nshot)}shot_{time_des}'
        if config['unit'] is True:
            sub_dst_name += '_unit'
        if config['reference_range'] is True:
            sub_dst_name += '_range'
        if config.get('prompt_engineering') is True:
            sub_dst_name += '_cot'
        if impute == 0:
            sub_dst_name += '_no_impute'
        elif impute == 1:
            sub_dst_name += '_impute'
        elif impute == 2:
            sub_dst_name += '_impute_info'
        sub_logits_path = os.path.join(logits_path, sub_dst_name)
        Path(sub_logits_path).mkdir(parents=True, exist_ok=True)
    if output_prompts:
        prompts_path = os.path.join(prompts_root, dataset, task, model)
        Path(prompts_path).mkdir(parents=True, exist_ok=True)
        sub_dst_name = f'{form}_{str(nshot)}shot_{time_des}'
        if config['unit'] is True:
            sub_dst_name += '_unit'
        if config['reference_range'] is True:
            sub_dst_name += '_range'
        if config.get('prompt_engineering') is True:
            sub_dst_name += '_cot'
        if impute == 0:
            sub_dst_name += '_no_impute'
        elif impute == 1:
            sub_dst_name += '_impute'
        elif impute == 2:
            sub_dst_name += '_impute_info'
        sub_prompts_path = os.path.join(prompts_path, sub_dst_name)
        Path(sub_prompts_path).mkdir(parents=True, exist_ok=True)

    for x, y, pid, record_time in zip(xs, ys, pids, record_times):
        if isinstance(pid, float):
            pid = str(round(pid))
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
        input_format_description = INPUT_FORMAT_DESCRIPTION[form]
        if impute == 0:
            input_format_description += MISSING_VALUE_DESCRIPTION
        elif impute == 2:
            input_format_description += INSTRUCTING_MISSING_VALUE
        userPrompt = USERPROMPT.format(
            INPUT_FORMAT_DESCRIPTION=input_format_description,
            TASK_DESCRIPTION_AND_RESPONSE_FORMAT=task_description,
            UNIT_RANGE_CONTEXT=unit_range,
            EXAMPLE=example,
            SEX=sex,
            AGE=age,
            LENGTH=length,
            RECORD_TIME_LIST=', '.join(list(map(str, record_time))),
            DETAIL=detail,
            RESPONSE_FORMAT=response_format,
        )
        if output_prompts:
            with open(os.path.join(sub_prompts_path, f'{pid}.txt'), 'w') as f:
                f.write(userPrompt)
        if output_logits:
            try:
                result, prompt_token, completion_token = query_llm(
                    model=model,
                    llm=llm,
                    systemPrompt=SYSTEMPROMPT[dataset],
                    userPrompt=userPrompt
                )
            except Exception as e:
                # logging.info(f'PatientID: {patient.iloc[0]["PatientID"]}:\n')
                logging.info(f'Query LLM Exception: {e}')
                continue
            prompt_tokens += prompt_token
            completion_tokens += completion_token
            if task == 'outcome':
                label = y[0][0]
            elif task == 'readmission':
                label = y[0][2]
            elif task == 'los':
                label = [yi[1] for yi in y]
            elif task == 'multitask':
                label = [y[0][0], y[0][2]]
            else:
                raise ValueError(f'Unknown task: {task}')
            try:
                if config.get('prompt_engineering') is True:
                    pred = result
                elif task in ['los', 'multitask']:
                    pred = [float(p) for p in result.split(',')]
                else:
                    pred = float(result)
            except:
                if task == 'los':
                    pred = [0] * len(label)
                elif task == 'multitask':
                    pred = [0.501, 0.501]
                else:
                    pred = 0.501
                logging.info(f'PatientID: {pid}:\nResponse: {result}\n')
            pd.to_pickle({
                'prompt': userPrompt,
                'pred': pred,
                'label': label,
            }, os.path.join(sub_logits_path, f'{pid}.pkl'))
            labels.append(label)
            preds.append(pred)
    if output_logits:
        logging.info(f'Prompts: {prompt_tokens}, Completions: {completion_tokens}, Total: {prompt_tokens + completion_tokens}\n\n')    
        pd.to_pickle({
            'config': config,
            'preds': preds,
            'labels': labels,
        }, os.path.join(logits_path, sub_dst_name + '.pkl'))

if __name__ == '__main__':
    for config in params:
        run(config, output_logits=True, output_prompts=False)