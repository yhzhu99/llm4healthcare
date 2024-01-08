from sklearn.linear_model import LogisticRegression


class LR():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        max_depth: int, depth of trees
        """
        task = params['task']
        self.task = task
        seed = params['seed']
        if task == "outcome":
            self.model = LogisticRegression(random_state=seed, max_iter=200)
        elif task == "los":
            pass
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")

    def fit(self, x, y):
        if self.task == "outcome":
            self.model.fit(x, y[:, 0])
        elif self.task == "los":
            pass
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
    def predict(self, x):
        if self.task == "outcome":
            return self.model.predict(x)
        elif self.task == "los":
            pass
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
