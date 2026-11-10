# Загрузка и очистка.

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def main():
    with urlopen(URL, timeout=60) as response:
        data = BytesIO(response.read())
    with ZipFile(data) as archive:
        sheet = archive.read("default of credit card clients.xls")
    frame = pd.read_excel(BytesIO(sheet), header=1)
    frame = frame.rename(
        columns={"default payment next month": "default.payment.next.month"}
    )
    path = Path("data/UCI_Credit_Card.csv")
    path.parent.mkdir(exist_ok=True)
    path.write_text(frame.to_csv(index=False))


if __name__ == "__main__":
    main()
