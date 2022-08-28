FROM python:3.8.13

ADD . churn/

RUN pip install --upgrade pip
RUN pip install --default-timeout=100 -r churn/requirements.txt