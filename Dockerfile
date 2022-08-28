FROM python:3.8.13

ADD . xpyriments/

RUN pip install --upgrade pip
RUN pip install --default-timeout=100 -r xpyriments/requirements.txt