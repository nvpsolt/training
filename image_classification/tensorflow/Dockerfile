FROM gitlab-master.nvidia.com:5005/dl/dgx/tensorflow:19.05-py3-stage


WORKDIR /research

# Mount data into the docker
ADD . /workspace/image_classification
ENV PYTHONPATH="/workspace/image_classification:${PYTHONPATH}"

WORKDIR /workspace/image_classification
RUN pip3 install -r requirements.txt