import numpy as np


XGB_COMMON = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


def feature_matrix(data) -> np.ndarray:
    x = np.concatenate([data['morgan'], data['maccs'], data['physicochemical']], axis=1).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    return x
