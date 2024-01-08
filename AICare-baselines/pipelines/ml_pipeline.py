import os
from pathlib import Path

import pandas as pd
import lightning as L

import models
from datasets.loader.unpad import unpad_batch
from metrics import check_metric_is_better, get_all_metrics


class MlPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.task = config["task"]
        self.los_info = config["los_info"]
        self.model_name = config["model"]
        self.main_metric = config["main_metric"]
        self.cur_best_performance = {}

        model_class = getattr(models, self.model_name)
        self.model = model_class(**config)

        self.test_performance = {}
        self.test_outputs = {}
        checkpoint_folder = f'logs/train/{config["dataset"]}/{config["task"]}/{config["model"]}-fold{config["fold"]}-seed{config["seed"]}/checkpoints/'
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = os.path.join(checkpoint_folder, 'best.ckpt')

    def forward(self, x):
        pass
    def training_step(self, batch, batch_idx):
        # the batch is large enough to contain the whole training set
        x, y, lens, pid = batch
        x, y = unpad_batch(x, y, lens)
        self.model.fit(x, y) # y contains both [outcome, los]
    def validation_step(self, batch, batch_idx):
        x, y, lens, pid = batch
        x, y = unpad_batch(x, y, lens)
        y_hat = self.model.predict(x) # y_hat is the prediction results, outcome or los
        metrics = get_all_metrics(y_hat, y, self.task, self.los_info)
        # for k, v in metrics.items(): self.log(k, v)
        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.task):
            self.cur_best_performance = metrics
            for k, v in metrics.items(): self.log("best_"+k, v)
            pd.to_pickle(self.model, self.checkpoint_path)
        return main_score
    def test_step(self, batch, batch_idx):
        x, y, lens, pid = batch
        x, y = unpad_batch(x, y, lens)
        self.model = pd.read_pickle(self.checkpoint_path)
        y_hat = self.model.predict(x)
        self.test_performance = get_all_metrics(y_hat, y, self.task, self.los_info)
        self.test_outputs = {'preds': y_hat, 'labels': y}
        return self.test_performance
    def configure_optimizers(self):
        pass
