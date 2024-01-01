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
    performance = pd.DataFrame(data=get_all_metrics(
        preds=preds,
        labels=labels,
        task='outcome',
        los_info=None
    ), index=[0])
    dst_path = os.path.join(dst_root, config['dataset'], config['task'], config['model'])
    os.makedirs(dst_path, exist_ok=True)
    performance.to_csv(os.path.join(dst_path, f'{Path(dst_path).name}.csv'))

if __name__ == '__main__':
    export_performance('/data/wangzx/llm4healthcare/logits/tjh/outcome/gpt-4-1106-preview/20240102-001910.pkl')
