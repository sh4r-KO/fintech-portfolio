# BE/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .   
# copies BE/* into /app → /app/DataManagement exists

ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8080"]
