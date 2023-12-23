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
    labels = logits['labels']
    preds = logits['preds']
    performance = pd.DataFrame()
    for i, (label, pred) in enumerate(zip(labels, preds)):
        performance = pd.concat([performance, pd.DataFrame(data=get_all_metrics(
            preds=pred,
            labels=label,
            task='outcome',
            los_info=None
        ), index=[f'Visit {i + 1}'])], axis=0)
    dst_path = os.path.join(dst_root, config['model'], config['prediction'] + '_' + config['format'], 'oneshot' if config['shot'] else 'zeroshot')
    os.makedirs(dst_path, exist_ok=True)
    performance.to_csv(os.path.join(dst_path, f'{Path(dst_path).name}.csv'))

if __name__ == '__main__':
    export_performance('logits/gpt-3.5-turbo-16k/N-1_list/oneshot/20231218-150859.pkl')
