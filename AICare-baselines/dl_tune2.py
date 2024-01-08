import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

import wandb
from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline

# import os
# os.environ['WANDB_MODE'] = 'offline'
# os.environ['WANDB_LOG_LEVEL'] = 'debug'

project_name = "aicare"

hydra.initialize(config_path="configs", version_base=None)
cfg = OmegaConf.to_container(hydra.compose(config_name="config"))

dataset_config = {
    'tjh': {'demo_dim': 2, 'lab_dim': 73},
    'cdsl': {'demo_dim': 2, 'lab_dim': 97},
    'mimic-iii': {'demo_dim': 2, 'lab_dim': 59},
    'mimic-iv': {'demo_dim': 2, 'lab_dim': 59},
}


sweep_id = "nal5p411"

def run_experiment():
    run = wandb.init(project=project_name, config=cfg)
    wandb_logger = WandbLogger(project=project_name, log_model=True) # log only the last (best) checkpoint
    config = wandb.config
    config.update(dataset_config[config['dataset']], allow_val_change=True)
    los_config = get_los_info(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}')
    main_metric = "mae" if config["task"] == "los" else "auprc"
    config.update({"los_info": los_config, "main_metric": main_metric})
    
    # data
    dm = EhrDataModule(f'datasets/{config["dataset"]}/processed/fold_{config["fold"]}', batch_size=config["batch_size"])

    # EarlyStop and checkpoint callback
    if config["task"] in ["outcome", "multitask"]:
        early_stopping_callback = EarlyStopping(monitor="auprc", patience=config["patience"], mode="max",)
        checkpoint_callback = ModelCheckpoint(monitor="auprc", mode="max")
    elif config["task"] == "los":
        early_stopping_callback = EarlyStopping(monitor="mae", patience=config["patience"], mode="min",)
        checkpoint_callback = ModelCheckpoint(monitor="mae", mode="min")

    L.seed_everything(config["seed"]) # seed for reproducibility

    # train/val/test
    pipeline = DlPipeline(config.as_dict())
    trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=config["epochs"], logger=wandb_logger, callbacks=[early_stopping_callback, checkpoint_callback], num_sanity_val_steps=0)
    trainer.fit(pipeline, dm)
    print("Best Score", checkpoint_callback.best_model_score)

if __name__ == "__main__":
   wandb.agent(sweep_id, function=run_experiment, project=project_name)
