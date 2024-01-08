import os

import lightning as L
import pandas as pd

from configs.exp import hparams
from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline

def get_latest_file(path):
    # Get list of all files in the directory
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    # Get the file with the latest modification time
    latest_file = max(files, key=os.path.getctime)
    
    return latest_file

def run_ml_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})
    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm)
    perf = pipeline.test_performance
    return perf

def run_dl_experiment(config):
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    config.update({"los_info": los_config})

    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])
    # checkpoint
    # checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/best.ckpt'

    checkpoint_path = get_latest_file(f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints')

    print("checkpoint_path: ", checkpoint_path)

    if "time_aware" in config and config["time_aware"] == True:
        checkpoint_path = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}-ta/checkpoints/best.ckpt'

    # train/val/test
    pipeline = DlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=False, num_sanity_val_steps=0)
    trainer.test(pipeline, dm, ckpt_path=checkpoint_path)
    perf = pipeline.test_performance
    return perf

if __name__ == "__main__":
    best_hparams = hparams # [TO-SPECIFY]
    performance_table = {'dataset':[], 'task': [], 'model': [], 'fold': [], 'seed': [], 'accuracy': [], 'auroc': [], 'auprc': [], 'f1': [], 'minpse': []}
    for i in range(0, len(best_hparams)):
    # for i in range(0, 1):
        config = best_hparams[i]
        print(f"Testing... {i}/{len(best_hparams)}")
        run_func = run_ml_experiment if config["model"] in ["RF", "DT", "GBDT", "XGBoost", "CatBoost", "LR", "LightGBM"] else run_dl_experiment
        seeds = [0] # [0,1,2,3,4]
        folds = ['nshot']
        for fold in folds:
            config["fold"] = fold
            for seed in seeds:
                config["seed"] = seed
                perf = run_func(config)
                print(f"{config}, Test Performance: {perf}")

                if "time_aware" in config and config["time_aware"] == True:
                    model_name = config['model']+"_ta"
                else:
                    model_name = config['model']

                performance_table['dataset'].append(config['dataset'])
                performance_table['task'].append(config['task'])
                performance_table['model'].append(model_name)
                performance_table['fold'].append(config['fold'])
                performance_table['seed'].append(config['seed'])
                if config['task'] == 'outcome':
                    performance_table['accuracy'].append(perf['accuracy'])
                    performance_table['auroc'].append(perf['auroc'])
                    performance_table['auprc'].append(perf['auprc'])
                    performance_table['f1'].append(perf['f1'])
                    performance_table['minpse'].append(perf['minpse'])
    pd.DataFrame(performance_table).to_csv('ijcai24_ml_baselines_20240108.csv', index=False) # [TO-SPECIFY]
