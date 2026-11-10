import argparse
from pathlib import Path

import great_expectations as gx
import numpy as np
import pandas as pd

from .schema import BILLS, FEATURES, LIMITS, PAYMENTS, RAW, TARGET, normalize_categories


# проверяет входную таблицу
def validate(frame):
    dataset = gx.from_pandas(frame)
    dataset.set_config_value("interactive_evaluation", False)
    # строка должна содержать id, features, target
    dataset.expect_table_columns_to_match_set(["ID", *FEATURES, TARGET])
    dataset.expect_table_row_count_to_be_between(min_value=1)
    dataset.expect_column_values_to_be_unique("ID")
    for column, (minimum, maximum) in {
        "ID": (1, None),
        **LIMITS,
        TARGET: (0, 1),
    }.items():
        dataset.expect_column_values_to_not_be_null(column)
        dataset.expect_column_values_to_be_in_type_list(column, ["int64", "int32"])
        if minimum is not None or maximum is not None:
            dataset.expect_column_values_to_be_between(column, minimum, maximum)
    result = dataset.validate(catch_exceptions=True, only_return_failures=True)
    if not result.success:
        failed = [
            (
                item.expectation_config.expectation_type,
                item.expectation_config.kwargs.get("column"),
            )
            for item in result.results
        ]
        raise ValueError(f"Проверка данных не пройдена. Нарушенные правила: {failed}")


# читает CSV и исправляет тип денежных столбцов
def read_raw(path):
    frame = pd.read_csv(path)
    for column in ["LIMIT_BAL", *BILLS, *PAYMENTS]:
        if column not in frame:
            continue
        values = frame[column]
        if pd.api.types.is_float_dtype(values) and (
            np.isfinite(values).all()
            and values.eq(np.trunc(values)).all()
            # Все числа по модулю укладываются в выбранную границу
            # точного представления целых в float64
            and values.abs().lt(2**53).all()
        ):
            frame[column] = values.astype("int64")
    return frame


# Формирование таблицы для обучения
def prepare(frame):
    return normalize_categories(frame[[*FEATURES, TARGET]])


def main():
    parser = argparse.ArgumentParser(description="Проверка и подготовка данных UCI")
    parser.add_argument("action", choices=["validate", "prepare"])
    parser.add_argument("--input", default=RAW, help="Путь к исходному CSV")
    args = parser.parse_args()
    frame = read_raw(args.input)
    validate(frame)
    print(f"Строки, прошедшие проверку Great Expectations: {len(frame)}")
    # запись в таблицу
    if args.action == "prepare":
        path = Path("data/processed/clients.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        prepare(frame).to_csv(path, index=False)


if __name__ == "__main__":
    main()
