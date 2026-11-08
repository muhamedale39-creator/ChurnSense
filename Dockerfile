FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn==0.27.1 fastapi==0.109.2

# Copy the saved model artifacts and API script
COPY churn_artifacts.joblib .
COPY main.py .

# Expose the API port
EXPOSE 8000

# Run the Uvicorn server in exec form for clean shutdowns
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]