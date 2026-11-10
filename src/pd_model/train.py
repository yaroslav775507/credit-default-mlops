import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from .pipeline import make_pipeline
from .schema import FEATURES, MODEL, TARGET, THRESHOLD

SEED = 42


def metrics(y, probabilities):
    predictions = probabilities >= THRESHOLD
    return {
        "roc_auc": roc_auc_score(y, probabilities),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
    }


def main():
    matplotlib.use("Agg")
    data = pd.read_csv("data/processed/clients.csv")
    train, test = train_test_split(
        data, test_size=0.2, random_state=SEED, stratify=data[TARGET]
    )
    x_train, x_test = train[FEATURES], test[FEATURES]
    # У логистической регрессии меньший C означает более сильную регуляризацию.
    # В бустинге max_leaf_nodes ограничивает число листьев в каждом дереве.
    search = GridSearchCV(
        make_pipeline(),
        [
            {"classifier__C": [0.01, 0.1, 1.0]},
            {
                "classifier": [HistGradientBoostingClassifier(random_state=SEED)],
                "classifier__max_leaf_nodes": [7, 15, 31],
            },
        ],
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        error_score="raise",
        n_jobs=1,
    )
    search.fit(x_train, train[TARGET])
    model = search.best_estimator_
    probability = model.predict_proba(x_test)[:, 1]
    scores = metrics(test[TARGET], probability)
    Path("reports").mkdir(exist_ok=True)
    Path(MODEL).parent.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("credit-default")
    candidates = []
    with mlflow.start_run(run_name="Подбор гиперпараметров") as parent:
        for index, params in enumerate(search.cv_results_["params"]):
            candidate = {
                "algorithm": (
                    type(params["classifier"]).__name__
                    if "classifier" in params
                    else "LogisticRegression"
                ),
                **{
                    key.split("__")[-1]: value
                    for key, value in params.items()
                    if key != "classifier"
                },
                "cv_roc_auc": float(search.cv_results_["mean_test_score"][index]),
                "cv_std": float(search.cv_results_["std_test_score"][index]),
                "rank": int(search.cv_results_["rank_test_score"][index]),
            }
            candidates.append(candidate)
            with mlflow.start_run(run_name=f"Конфигурация {index + 1}", nested=True):
                mlflow.log_params(
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in {"cv_roc_auc", "cv_std", "rank"}
                    }
                )
                mlflow.log_metrics(
                    {key: candidate[key] for key in ("cv_roc_auc", "cv_std", "rank")}
                )
        report = {
            "train_rows": len(train),
            "test_rows": len(test),
            "test_default_rate": float(test[TARGET].mean()),
            "threshold": THRESHOLD,
            "winner": candidates[search.best_index_],
            "test_metrics": scores,
            "candidates": candidates,
            "mlflow_run": parent.info.run_id,
        }
        Path("reports/metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        RocCurveDisplay.from_predictions(test[TARGET], probability)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Доля ложноположительных прогнозов")
        plt.ylabel("Доля истинноположительных прогнозов")
        plt.title("ROC-кривая модели дефолта")
        plt.tight_layout()
        plt.savefig("reports/roc.png", dpi=140)
        plt.close()
        mlflow.log_params({"seed": SEED, "threshold": THRESHOLD, "folds": 3})
        mlflow.log_params(
            {
                f"best_{key}": value
                for key, value in candidates[search.best_index_].items()
                if key not in {"cv_roc_auc", "cv_std", "rank"}
            }
        )
        mlflow.log_metrics({"cv_roc_auc": search.best_score_, **scores})
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifact("reports/roc.png")
        example = x_train.head(5)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            code_paths=["src/pd_model"],
            input_example=example,
            signature=infer_signature(example, model.predict(example)),
        )
    joblib.dump(model, MODEL)
    test.to_csv("data/processed/test.csv", index=False)
    pd.DataFrame({"probability": model.predict_proba(x_train)[:, 1]}).to_csv(
        "models/reference.csv", index=False
    )
    print("Результаты обучения и проверки модели", file=sys.stderr)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
