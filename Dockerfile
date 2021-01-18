FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt 
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
EXPOSE 5001 
ENTRYPOINT [ "python" ] 
CMD [ "app.py" ] 