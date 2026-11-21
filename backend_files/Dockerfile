FROM python:3.10-slim

WORKDIR /app

# libgomp is required by XGBoost / scikit-learn on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app and the serialized model
COPY . .

EXPOSE 7860

# Serve the Flask app with gunicorn on port 7860
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "app:superkart_api"]
