TARGET = "default.payment.next.month"
RAW = "data/UCI_Credit_Card.csv"
STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
LIMITS = {
    "LIMIT_BAL": (1, None),
    "SEX": (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3),
    "AGE": (18, 120),
    **dict.fromkeys(STATUS, (-2, 9)),
    **dict.fromkeys(BILLS, (None, None)),
    **dict.fromkeys(PAYMENTS, (0, None)),
}
FEATURES = list(LIMITS)
MODEL = "models/pd_model.joblib"
THRESHOLD = 0.5


def normalize_categories(frame):
    return frame.assign(
        EDUCATION=frame["EDUCATION"].replace({0: 4, 5: 4, 6: 4}),
        MARRIAGE=frame["MARRIAGE"].replace({0: 3}),
    )
