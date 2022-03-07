import os
import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role, Session
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import numpy as np
import sagemaker

sess = Session()

# for acoustic account & local deployment
role = 'arn:aws:iam::302145289873:role/service-role/AmazonSageMaker-ExecutionRole-20211223T112159'

# upload the trained pytorch model to S3
cnn14max_model_data = sess.upload_data(
    path="cnn14max.tar.gz", bucket=sess.default_bucket(), key_prefix="sound-event-model/pytorch"
)

model = PyTorchModel(
    entry_point = "inference.py",
    source_dir = "code",
    role = role,
    model_data = cnn14max_model_data,
    framework_version = "1.9.0",
    py_version = "py38",
    image_uri = '302145289873.dkr.ecr.ap-southeast-1.amazonaws.com/sound-event-detection:v1'
)

# set local_mode to False if you want to deploy on a remote
# SageMaker instance

local_mode = False
if local_mode:
    instance_type = "local"
else:
    instance_type = "ml.c4.xlarge"
predictor = model.deploy(
    initial_instance_count =  1,
    instance_type =instance_type,
    serializer = JSONSerializer(),
    deserializer = JSONDeserializer()
)

# TESTING
rng = np.random.default_rng()
audio = 2 * rng.random(size=44100, dtype=np.float32) - 1
data_dict = {"inputs" : audio.tolist()}
predictor.predict(data_dict)