FROM python:3.12-slim
# libgomp is needed by LightGBM / XGBoost; curl is used by the health check
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py carprep.py model.pkl model_meta.json ./
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
