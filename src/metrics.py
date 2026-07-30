import numpy as np


def brier_score(df, prob_col, outcome_col="home_win"):
    """Mean squared error between predicted probability and the binary outcome."""
    return float(((df[prob_col] - df[outcome_col]) ** 2).mean())


def log_loss(df, prob_col, outcome_col="home_win"):
    """Mean negative log-likelihood of the binary outcome under the predicted probability."""
    p = df[prob_col].clip(1e-6, 1 - 1e-6)
    y = df[outcome_col]
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
