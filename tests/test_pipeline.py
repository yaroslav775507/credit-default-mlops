import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pd_model.api import app
from pd_model.data import prepare, validate
from pd_model.monitor import psi
from pd_model.pipeline import engineer, make_pipeline
from pd_model.schema import FEATURES, TARGET
from pd_model.train import metrics


@pytest.fixture
def raw():
    return pd.DataFrame(
        [
            {
                "ID": i + 1,
                **dict.fromkeys(FEATURES, 0),
                "LIMIT_BAL": 10000 * (i + 1),
                "SEX": 1 + i % 2,
                "AGE": 24 + i,
                "PAY_0": i,
                TARGET: i % 2,
            }
            for i in range(4)
        ]
    )


def test_validation_accepts_known_legacy_categories(raw):
    validate(raw)


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("AGE", [24, 25, 26, 160]),
        ("AGE", [24, 25, 26, None]),
        (TARGET, ["0", "1", "0", "1"]),
        ("ID", [1, 1, 3, 4]),
    ],
)
def test_validation_detects_anomalies(raw, column, values):
    raw[column] = values
    with pytest.raises(ValueError, match="Проверка данных не пройдена"):
        validate(raw)


def test_feature_engineering(raw):
    prepared = prepare(raw)
    assert "ID" not in prepared
    result = engineer(prepared)
    assert result["AGE_GROUP"].tolist() == [0, 0, 1, 1]
    assert result["LATE_MONTHS"].tolist() == [0, 1, 1, 1]
    assert result["EDUCATION"].eq(4).all()
    assert result["MARRIAGE"].eq(3).all()


def test_metrics_and_drift():
    assert metrics([0, 1], np.array([0.1, 0.9])) == {
        "roc_auc": 1,
        "precision": 1,
        "recall": 1,
        "f1": 1,
    }
    assert psi([0.1, 0.9], [0.1, 0.9]) == 0
    assert psi([0.1] * 50, [0.9] * 50) > 0


def test_api_loads_pipeline_and_scores(raw, tmp_path, monkeypatch):
    model = make_pipeline().fit(raw[FEATURES], raw[TARGET])
    path = tmp_path / "test.joblib"
    joblib.dump(model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    row = raw[FEATURES].iloc[0].to_dict()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=row)
        assert response.status_code == 200
        result = response.json()
        assert result["probability"] == pytest.approx(
            model.predict_proba(raw[FEATURES].iloc[:1])[0, 1]
        )
        assert result["prediction"] == int(result["probability"] >= 0.5)
        assert client.post("/predict", json={**row, "AGE": 160}).status_code == 422
