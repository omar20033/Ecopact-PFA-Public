FROM python:3.8.2
WORKDIR /main
COPY "./requirements.txt"

RUN pip install -r requirements.txt
EXPOSE 5000
COPY ./main
ENTRYPOINT ["python","main.py"]

