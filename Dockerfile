# Note: Since git repositories are cloned, an active internet connection is required

# The predictions are performed on Ubuntu 22.04
FROM ubuntu:22.04

# Set the working directory to /app
WORKDIR /application

# Set up specific apt package repositories (if needed)
RUN apt update

# Install git
RUN apt install -y -qq git

# Install python3 (which is needed for the evaluation of the scripts) and redirect python to python3
RUN apt install -y -qq python3 python3-pip python-is-python3

# Clone the supplementary web site containing the data and scripts
RUN git clone --depth=1 https://github.com/AI-4-SE/TwinsOrFalseFriends.git

# Install required packages
RUN cd TwinsOrFalseFriends \
    && pip install -q -r ./requirements.txt
