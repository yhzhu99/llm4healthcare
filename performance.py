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
    dst_path = os.path.join(dst_root, config['dataset'], config['task'], config['model'])
    os.makedirs(dst_path, exist_ok=True)
    performance.to_csv(os.path.join(dst_path, f'{Path(src_path).name}.csv'))

if __name__ == '__main__':
    export_performance('logits/tjh/outcome/gpt-4-1106-preview/string_0shot.pkl')
    export_performance('logits/tjh/outcome/gpt-4-1106-preview/string_0shot_unit_range.pkl')
