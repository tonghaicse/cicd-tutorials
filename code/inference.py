import os
import logging
import json
import numpy as np
from pann.sound_event_detection import CNN14Max

def model_fn(model_dir):
    cnn14max = CNN14Max()
    model_file = 'cnn14max.pth'
    logging.info('Loading the model file: {}'.format(model_file))
    cnn14max.load(model_file)
    return cnn14max

# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    # print('Request body: {}'.format(request_body))
    data = json.loads(request_body)["inputs"]
    data = np.array(data, dtype=np.float32)
    return data


# inference
def predict_fn(input_object, model):
    print('Input object: {}'.format(input_object))
    prediction = model.predict(input_object)
    return prediction


# postprocess
def output_fn(predictions, content_type):
    logging.info('Content Type: {}'.format(content_type))
    assert content_type == "application/json"
    return predictions.tolist()