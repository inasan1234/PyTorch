FROM python:3.10
USER root

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip

# Install Basic Packages
RUN pip install numpy && \
    pip install ipykernel jupyter && \
    pip install torch && \
    pip install torchvision && \
    pip install pillow