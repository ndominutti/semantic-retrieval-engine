FROM python:3.11-slim

ARG APP_PORT=8000
ENV APP_PORT=$APP_PORT

ENV ENV=$ENV

RUN pip install --upgrade pip

RUN apt-get update && \
    apt-get install -y python3-dev build-essential curl && \
    apt-get install -y git &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#Only for PoC
RUN git clone http://github.com/wayfair/WANDS.git

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ./
WORKDIR .

CMD uvicorn main:app --host 0.0.0.0 --port $APP_PORT