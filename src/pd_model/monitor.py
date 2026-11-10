import argparse
import json
import sys
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .schema import FEATURES


def psi(reference, current, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    distributions = []
    for values in (reference, current):
        values = np.asarray(values)
        if not len(values) or not np.all((values >= 0) & (values <= 1)):
            raise ValueError("Нужны непустые наборы вероятностей в диапазоне от 0 до 1")
        counts = np.histogram(values, bins=edges)[0]
        smoothed = counts + 0.5
        distributions.append(smoothed / smoothed.sum())
    expected, actual = distributions
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def main():
    parser = argparse.ArgumentParser(description="Оценка дрейфа прогнозов по PSI")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Адрес API")
    parser.add_argument("--sample", type=int, default=300, help="Размер выборки")
    args = parser.parse_args()
    test = pd.read_csv("data/processed/test.csv", usecols=FEATURES)
    if not 1 <= args.sample <= len(test):
        parser.error(f"Параметр --sample должен быть в диапазоне от 1 до {len(test)}")
    probabilities = []
    for row in test.sample(args.sample, random_state=42).to_dict("records"):
        request = Request(
            args.url.rstrip("/") + "/predict",
            data=json.dumps(row).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=10) as response:
            probabilities.append(json.load(response)["probability"])
    reference = pd.read_csv("models/reference.csv")["probability"]
    print("Результат проверки дрейфа прогнозов", file=sys.stderr)
    print(json.dumps({"sample": args.sample, "psi": psi(reference, probabilities)}))


if __name__ == "__main__":
    main()
