FROM timesler/jupyter-dl-gpu:1.7

USER root

WORKDIR /home/jovyan/work

RUN apt-key del A4B469963BF863CC

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb && \
    rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update && apt-get install -y curl

RUN pip install --upgrade pip

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION="1.5.0"

RUN pip install --upgrade pip

RUN curl -sSSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false


COPY ./pyproject.toml* ./poetry.lock* ./initialize.py ./

RUN poetry install

RUN python initialize.py

CMD [ "streamlit", "run", "app.py" ]
