# FROM python:3.11
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . /app
# CMD uvicorn --host=0.0.0.0 main:app --reload

FROM openjdk:11-jre-slim

# Install Python 3.11
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set the JAVA_HOME environment variable
ENV JAVA_HOME=/usr/local/openjdk-11

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

CMD uvicorn --host=0.0.0.0 main:app --reload