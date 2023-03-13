FROM python:3.8


WORKDIR /deteccion_de_rostros
COPY ./requirements.txt /deteccion_de_rostros/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /deteccion_de_rostros/requirements.txt

COPY . /deteccion_de_rostros/

CMD bash -c "while true; do sleep1; done"