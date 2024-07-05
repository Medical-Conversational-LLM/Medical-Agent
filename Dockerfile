FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    vim \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    nginx \
    curl \ 
    build-essential \
    supervisor \
    && apt-get clean

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - 

RUN apt-get install -y nodejs

RUN node -v  && npm -v

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
&& /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
&& rm /tmp/miniconda.sh


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY . /app
WORKDIR /app

ENV PATH /opt/conda/bin:$PATH
RUN /opt/conda/bin/conda init bash

RUN cd client && npm install && npm run build

RUN conda env create -f environment.yml && conda clean -a -y

RUN echo "conda activate self-rag3" >> ~/.bashrc
ENV PATH /opt/conda/envs/self-rag3/bin:$PATH
ENV CONDA_DEFAULT_ENV $self-rag3

SHELL ["conda", "run", "-n", "self-rag3", "/bin/bash", "-c"]

RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/sites-enabled/nginx.conf
 
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

RUN mkdir -p /var/log/nginx /var/log/supervisor /var/log/gunicorn
 
EXPOSE 80/tcp 
EXPOSE 443/tcp 

ENTRYPOINT ["/entrypoint.sh"]


