#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
test -f models/pd_model.joblib || {
    echo "Сначала обучите модель командой dvc repro." >&2
    exit 1
}
docker build -t credit-default:local .
exec docker run --rm -p "127.0.0.1:${PORT:-8000}:8000" credit-default:local
