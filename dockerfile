FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY static/ static/
COPY templates/ templates/
COPY app.py .

ENV FLASK_APP=app.py
ENV IMAGE_FOLDER=/app/static/images
EXPOSE 8000

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]