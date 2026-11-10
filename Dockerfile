FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/ src/
COPY models/pd_model.joblib models/pd_model.joblib
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8000
USER 65534
CMD ["uvicorn", "pd_model.api:app", "--host", "0.0.0.0", "--port", "8000"]
