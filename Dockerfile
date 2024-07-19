FROM ubuntu:22.04
FROM python:3.10.12

WORKDIR /src

COPY . /src

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD [ "python3", "api_service.py" ]