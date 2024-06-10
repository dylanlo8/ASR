FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y python3 python3-pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "python3", "main.py"]