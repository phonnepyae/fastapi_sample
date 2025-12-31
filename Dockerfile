FROM python:3.12-slim AS builder

EXPOSE $PORT

WORKDIR /app


COPY . .


RUN pip install pipenv


RUN pipenv install --system --deploy


CMD ["python","main.py"]
