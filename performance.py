import os
from pathlib import Path

import pandas as pd

from metrics import get_all_metrics

def export_performance(
    src_path: str,
    dst_root: str='performance',
):
    logits = pd.read_pickle(src_path)
    config = logits['config']
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
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    performance.to_csv(os.path.join(dst_path, f'{sub_dst_name}.csv'))

if __name__ == '__main__':
    export_performance('logits/mimic-iv/outcome/gpt-4-1106-preview/20240103-113726.pkl')
    export_performance('/data/wangzx/llm4healthcare/logits/mimic-iv/outcome/gpt-4-1106-preview/20240103-115018.pkl')
