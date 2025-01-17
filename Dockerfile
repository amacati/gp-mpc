# Note
# 1. build the docker image named safe-control-gym
# docker build (--no-cache) -t safe-control-gym .
# 2. run the docker container
# docker run -it --gpus all --rm --name safe-control-gym -v /home/tsung/safe-control-gym/docker_results:/project/safe-control-gym/examples/hpo/hpo safe-control-gym
# 3. run experiment in the container
# bash examples/hpo/hpo.sh test 2 2 vizier cluster quadrotor_2D_attitude ilqr 100 False tracking False
# 4. push the image
# docker tag safe-control-gym:latest tsungyuan/safe-control-gym:latest
# docker push tsungyuan/safe-control-gym:latest

FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /project

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgmp-dev \
    libcdd-dev \
    cmake \
    g++ \
    make \
    python3-dev \
    python3-pip \
    git

# Clone and install safe-control-gym
RUN git clone https://<key>@github.com/middleyuan/safe-control-gym.git /project/safe-control-gym
RUN cd /project/safe-control-gym && git checkout benchmark && pip install -e .

WORKDIR /project

# Clone and set up ACADOS
RUN git clone https://github.com/acados/acados.git /project/acados && \
    cd /project/acados && \
    git submodule update --recursive --init && \
    mkdir -p build && \
    cd build && \
    cmake -DACADOS_WITH_QPOASES=ON -DACADOS_WITH_OPENMP=ON .. && \
    make install -j4

# Install ACADOS Python interface
RUN pip install -e /project/acados/interfaces/acados_template

# Set environment variables
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/project/acados/lib" >> ~/.bashrc && \
    echo "export ACADOS_SOURCE_DIR=/project/acados" >> ~/.bashrc

WORKDIR /project/safe-control-gym