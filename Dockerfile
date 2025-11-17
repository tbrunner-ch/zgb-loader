FROM python:3.11-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# Requirements kopieren und installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Rest des Codes kopieren
COPY . .

# Standard-Command: unser Loader-Script ausführen
CMD ["python", "load_zgb.py"]
