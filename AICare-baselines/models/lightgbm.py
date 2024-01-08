import lightgbm as lgb
from lightgbm import LGBMClassifier


class LightGBM():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        learning_rate: float, learning rate
        max_depth: int, depth of trees
        """
        task = params['task']
        self.task = task
        seed = params['seed']
        n_estimators = params['n_estimators']
        learning_rate = params['learning_rate']
        max_depth = params['max_depth']
        if task == "outcome":
            self.model = LGBMClassifier(random_state=seed, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, verbose=0)
        elif task == "los":
            pass
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")

    def fit(self, x, y):
        if self.task == "outcome":
            self.model.fit(x, y[:, 0])
        elif self.task == "los":
            # self.model.fit(x, y[:, 1])
            pass
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
    def predict(self, x):
        if self.task == "outcome":
            return self.model.predict_proba(x)[:, 1]
        elif self.task == "los":
            # return self.model.predict(x)
            pass
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
