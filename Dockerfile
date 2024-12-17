# Note
# 1. build the docker image named safe-control-gym
# docker build (--no-cache) -t safe-control-gym .
# 2. run the docker container
# docker run -it --gpus all --rm --name safe-control-gym -v /home/tsung/safe-control-gym/docker_results:/project/safe-control-gym/examples/hpo/hpo safe-control-gym test 2 2 vizier cluster quadrotor_2D_attitude ilqr 100 False tracking False
# 3. push the image
# docker tag safe-control-gym:latest tsungyuan/safe-control-gym:latest
# docker push tsungyuan/safe-control-gym:latest


FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /project

RUN apt-get update && apt-get install -y libgmp-dev libcdd-dev

RUN git clone https://<key>@github.com/middleyuan/safe-control-gym.git /project/safe-control-gym

RUN cd safe-control-gym && git checkout benchmark && pip install -e .

WORKDIR /project/safe-control-gym

ENTRYPOINT ["bash", "./examples/hpo/hpo.sh"]