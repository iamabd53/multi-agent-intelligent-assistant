FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV OPENROUTER_API_KEY=""
EXPOSE 7860
CMD ["python", "app/main.py"]