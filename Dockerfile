FROM nvidia/cuda:12.2.0-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

#For python3.9
RUN apt-get update && apt-get install --no-install-recommends -y python3.9 python3.9-dev python3.9-venv python3-pip python3-wheel build-essential && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

# For python3.1
#RUN apt update
#RUN apt-get install software-properties-common -y
#RUN add-apt-repository ppa:deadsnakes/ppa -y
#RUN apt update
#RUN apt upgrade -y
#RUN apt-get update && apt-get install --no-install-recommends -y python3.10 python3.10-venv python3-pip python3-wheel build-essential && \
#	apt-get clean && rm -rf /var/lib/apt/lists/*

# create and activate virtual environment
# using final folder name to avoid path issues with packages
RUN python3.9 -m venv /home/myuser/venv
#RUN python3.10 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

RUN pip3 install --no-cache-dir wheel

RUN apt-get update && apt-get install -y cuda-toolkit-12-2

#RUN apt update
#RUN apt-get install -y python3.9 python3-pip

RUN pip install jupyterlab==4.1.6
RUN pip install pandas==2.2.2
RUN pip install scikit-learn==1.4.2
RUN pip install matplotlib==3.8.4
RUN pip install seaborn==0.13.2

RUN pip install transformers==4.37.2
RUN pip install tensorflow==2.15.0
RUN pip install torch==2.3.0 
RUN pip install torchvision==0.18.0
RUN pip install torchaudio==2.3.0
RUN pip install unidecode==1.3.8
RUN pip install scipy==1.10.1
RUN pip install nltk==3.8.1
RUN pip install gensim==4.3.2
RUN pip install spacy==3.7.4
RUN pip install yake==0.4.8
RUN pip install ipywidgets==8.1.2

RUN pip install langchain-community==0.0.34
RUN pip install langchain==0.1.17
RUN pip install keybert==0.8.5

# Set the working directory inside the container
WORKDIR /app

# Expose the Jupyter port
EXPOSE 8888

# Command to run Jupyter Notebook
#CMD [ "/bin/bash" ]
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
