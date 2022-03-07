import json
import os
import shutil
import tarfile

import boto3
import botocore
import numpy as np
import sagemaker

from inference import input_fn, model_fn, output_fn, predict_fn

def test(model_data):
    # load the model
    net = model_fn(model_data)

    # simulate some input data to test transform_fn
    rng = np.random.default_rng()
    audio = 2 * rng.random(size=44100, dtype=np.float32) - 1
    data_dict = {"inputs": audio.tolist()}

    # encode numpy array to binary stream
    serializer = sagemaker.serializers.JSONSerializer()

    jstr = serializer.serialize(data_dict)
    jstr = json.dumps(data_dict)

    # "send" the bin_stream to the endpoint for inference
    # inference container calls transform_fn to make an inference
    # and get the response body for the caller

    content_type = "application/json"
    input_object = input_fn(jstr, content_type)
    predictions = predict_fn(input_object, net)
    res = output_fn(predictions, content_type)
    print(res)
    return

if __name__ == "__main__":
    model_data = "../cnn14max.pth"
    test(model_data)