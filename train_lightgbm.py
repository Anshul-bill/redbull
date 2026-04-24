"""LightGBM classifier factory used by the traditional model comparison."""
from lightgbm import LGBMClassifier

from config import RANDOM_STATE


def build_lightgbm() -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
    )
