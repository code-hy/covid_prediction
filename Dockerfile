FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG TRAIN=false
COPY train.py .
# Only run training at build time if explicitly requested via --build-arg TRAIN=true
RUN if [ "$TRAIN" = "true" ] ; then python3 train.py ; else echo "Skipping training during build (TRAIN=$TRAIN)"; fi
COPY covid_forecast_models.pkl .
COPY cleaned_covid_data.csv .
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]