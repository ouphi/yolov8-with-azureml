import os
import json
from ultralytics import YOLO

def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "best.pt"
    )
    model = YOLO(model_path)


def run(raw_data):
    image_url = json.loads(raw_data)["image_url"]
    results = model(image_url)
    result = results[0]
    serialized_result = json.loads(result.tojson())
    return serialized_result
