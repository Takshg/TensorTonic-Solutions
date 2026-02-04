import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)

    errs = np.abs(np.subtract(y_true, y_pred))
    loss_values = np.where(errs <= delta, 0.5*errs**2, delta*(errs-0.5*delta))
    return float(np.mean(loss_values))
    pass