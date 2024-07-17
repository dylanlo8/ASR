FROM ubuntu:22.04
FROM python:3.10.12

RUN apt-get update && \
    apt-get install -y python3 python3-pip

WORKDIR /src

COPY . /src

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV PYTHONUNBUFFERED=1

CMD [ "python3", "api_service.py" ]