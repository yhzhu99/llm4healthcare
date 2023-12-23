from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score

def get_regression_metrics(preds, labels):
    # get regression metrics: mse, mae, rmse, r2
    mse = MeanSquaredError(squared=True)
    rmse = MeanSquaredError(squared=False)
    mae = MeanAbsoluteError()
    r2 = R2Score()

    mse(preds, labels)
    rmse(preds, labels)
    mae(preds, labels)
    r2(preds, labels)

    # return a dictionary
    return {
        "mse": mse.compute(),
        "rmse": rmse.compute(),
        "mae": mae.compute(),
        "r2": r2.compute(),
    }
