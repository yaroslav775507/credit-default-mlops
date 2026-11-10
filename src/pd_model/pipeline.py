import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .schema import BILLS, FEATURES, PAYMENTS, STATUS, normalize_categories

CATEGORIES = ["SEX", "EDUCATION", "MARRIAGE", "AGE_GROUP"]
AGGREGATES = ["MAX_DELAY", "LATE_MONTHS", "MEAN_BILL", "MEAN_PAYMENT"]

# создаем возрастную группу, максимальный статус просрочки, число месяцев с просрочкой,
# среднюю сумму счёта и средний платёж


def engineer(frame):
    frame = normalize_categories(frame)
    frame["AGE_GROUP"] = pd.cut(
        frame["AGE"], [0, 25, 35, 45, 55, 120], labels=False, include_lowest=True
    )
    frame["MAX_DELAY"] = frame[STATUS].max(axis=1)
    frame["LATE_MONTHS"] = frame[STATUS].gt(0).sum(axis=1)
    frame["MEAN_BILL"] = frame[BILLS].mean(axis=1)
    frame["MEAN_PAYMENT"] = frame[PAYMENTS].mean(axis=1)
    return frame


# собирает три шага. Они выполняются последовательно при обучении и при прогнозе.
def make_pipeline():
    numeric = [name for name in [*FEATURES, *AGGREGATES] if name not in CATEGORIES]
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer)),
            (
                "preprocessing",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                [
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent"),
                                    ),
                                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            CATEGORIES,
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )
