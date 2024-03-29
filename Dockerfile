FROM python:3.8.10

#copy the following files and folders
COPY ./app /app
COPY ./requirements.txt /requirements.txt
COPY ./pipelines /pipelines
COPY .env .env

RUN apt update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
#install whatever is necessary
RUN pip install --upgrade pip
RUN python3 -m pip --no-cache-dir install -r requirements.txt
#download pretrained model
RUN python3 -m pypyr /pipelines/ai-model-download

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--reload"]