# To test this container locally, run:
# docker build -t mosartwmpy .
# docker run --rm -p 8888:8888 mosartwmpy

FROM ghcr.io/msd-live/jupyter/python-notebook:dev

USER root

# Give permissions to jovyan to write to the matplotlib directory.
RUN mkdir -p /home/jovyan/.config/matplotlib && \
chmod 777 -R /home/jovyan/.config/matplotlib

# Install mosartwmpy
RUN pip install --upgrade pip
RUN pip install mosartwmpy

# Install tutorial data.
RUN python -c "from mosartwmpy.utilities.download_data import download_data; download_data('tutorial');"
