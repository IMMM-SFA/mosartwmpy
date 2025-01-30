# To test this container locally, run:
# docker build -t mosartwmpy .
# docker run --rm -p 8888:8888 mosartwmpy

FROM ghcr.io/msd-live/jupyter/python-notebook:latest

# Install mosartwmpy
RUN pip install --upgrade pip
RUN pip install mosartwmpy
