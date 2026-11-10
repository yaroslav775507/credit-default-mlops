import json
from contextlib import asynccontextmanager
from os import getenv
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import Body, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

from .schema import FEATURES, LIMITS, MODEL, THRESHOLD

# тестовые выгрузки из test.csv
PREDICT_EXAMPLES = json.loads(
    Path(__file__).with_name("predict_examples.json").read_text(encoding="utf-8")
)

Client = create_model(
    "Client",
    __config__=ConfigDict(extra="forbid"),
    **{
        name: (int, Field(strict=True, ge=minimum, le=maximum))
        for name, (minimum, maximum) in LIMITS.items()
    },
)


class Prediction(BaseModel):
    prediction: int = Field(description="Прогноз класса. 0 — без дефолта, 1 — дефолт.")
    probability: float = Field(description="Оценка вероятности дефолта от 0 до 1.")


@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(getenv("MODEL_PATH", MODEL))
    yield


app = FastAPI(title="Оценка риска кредитного дефолта", lifespan=lifespan)


@app.get("/health", summary="Проверить готовность сервиса")
def health():
    return {"status": "ok", "message": "Сервис готов к работе"}


@app.post("/predict", response_model=Prediction, summary="Оценить риск дефолта")
def predict(
    client: Annotated[Client, Body(openapi_examples=PREDICT_EXAMPLES)],
    request: Request,
):
    row = pd.DataFrame([client.model_dump()], columns=FEATURES)
    probability = float(request.app.state.model.predict_proba(row)[0, 1])
    return Prediction(prediction=int(probability >= THRESHOLD), probability=probability)
