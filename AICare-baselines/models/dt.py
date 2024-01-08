from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DT():
    def __init__(self, **params):
        """params is a dict
        seed: int, random seed
        n_estimators: int, number of trees
        max_depth: int, depth of trees
        """
        task = params['task']
        self.task = task
        seed = params['seed']
        max_depth = params['max_depth']
        if task == "outcome":
            self.model = DecisionTreeClassifier(random_state=seed, max_depth=max_depth)
        elif task == "los":
            self.model = DecisionTreeRegressor(random_state=seed,  max_depth=max_depth)
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")

    def fit(self, x, y):
        if self.task == "outcome":
            self.model.fit(x, y[:, 0])
        elif self.task == "los":
            self.model.fit(x, y[:, 1])
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
    def predict(self, x):
        if self.task == "outcome":
            return self.model.predict_proba(x)[:, 1]
        elif self.task == "los":
            return self.model.predict(x)
        else:
            raise ValueError("Task must be either 'outcome' or 'los'.")
