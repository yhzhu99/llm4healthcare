import os
from pathlib import Path

import pandas as pd
import torch

from metrics import get_all_metrics

def export_performance(
    src_path: str,
    dst_root: str='performance',
):
    logits = pd.read_pickle(src_path)
    config = logits['config']
    if config['task'] == 'multitask':
        _labels = logits['labels']
        _preds = logits['preds']
        labels = []
        preds = []
        for label, pred in zip(_labels, _preds):
            if pred[0] != 0.501:
                labels.append(label)
                preds.append(pred)
        _labels, _preds, labels, preds = torch.tensor(_labels), torch.tensor(_preds), torch.tensor(labels), torch.tensor(preds)
        outcome_metrics = get_all_metrics(preds[:, 0], labels[:, 0], 'outcome', None)
        readmission_metrics = get_all_metrics(preds[:, 1], labels[:, 1], 'outcome', None)
        _outcome_metrics = get_all_metrics(_preds[:, 0], _labels[:, 0], 'outcome', None)
        _readmission_metrics = get_all_metrics(_preds[:, 1], _labels[:, 1], 'outcome', None)
        data = {'count': [len(_labels), len(labels)] * 2}
        data = dict(data, **{k: [v1, v2, v3, v4] for k, v1, v2, v3, v4 in zip(_outcome_metrics.keys(), _outcome_metrics.values(), outcome_metrics.values(), _readmission_metrics.values(), readmission_metrics.values())})
        performance = pd.DataFrame(data=data, index=['o all', 'o without unknown samples', 'r all', 'r without unknown samples'])
    else:
        _labels = logits['labels']
        _preds = logits['preds']
        _metrics = get_all_metrics(_preds, _labels, 'outcome', None)
        labels = []
        preds = []
        for label, pred in zip(_labels, _preds):
            if pred != 0.501:
                labels.append(label)
                preds.append(pred)
        metrics = get_all_metrics(preds, labels, 'outcome', None)
        data = {'count': [len(_labels), len(labels)]}
        data = dict(data, **{k: [v1, v2] for k, v1, v2 in zip(_metrics.keys(), _metrics.values(), metrics.values())})
    
        performance = pd.DataFrame(data=data, index=['all', 'without unknown samples'])
    
    time = config['time']
    if time == 0:
        time_des = 'upon-discharge'
    elif time == 1:
        time_des = '1month'
    elif time == 2:
        time_des = '6months'
    dst_path = os.path.join(dst_root, config['dataset'], config['task'], config['model'])
    sub_dst_name = f'{config["form"]}_{str(config["n_shot"])}shot_{time_des}'
    if config['unit'] is True:
        sub_dst_name += '_unit'
    if config['reference_range'] is True:
        sub_dst_name += '_range'
    if config.get('prompt_engineering') is True:
        sub_dst_name += '_cot'
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    performance.to_csv(os.path.join(dst_path, f'{sub_dst_name}.csv'))

if __name__ == '__main__':
    for file in [
        'logits/mimic-iv/multitask/gpt-4-1106-preview/string_1shot_upon-discharge_unit_range.pkl'
    ]:
        export_performance(file)