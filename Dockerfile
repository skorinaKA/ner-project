FROM python:3.12-slim

WORKDIR /app

# Установка зависимостей
# COPY requirements.txt .
COPY . .
RUN pip install -r requirements.txt

# Копирование приложения
COPY . .

# Создание директорий
RUN mkdir -p models data
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# скачивание моделей spacy
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download xx_ent_wiki_sm

EXPOSE 8000

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]