# syntax=docker/dockerfile:1
FROM python:3.9-slim-bullseye
WORKDIR /app
COPY README.md README.md
COPY setup.cfg setup.cfg
COPY setup.py setup.py
COPY mosartwmpy mosartwmpy
RUN pip3 install -e .
VOLUME /data
WORKDIR /data
CMD [ "python3", "-c" , "from mosartwmpy import Model; m = Model(); m.initialize('config.yaml'); m.update_until();"]