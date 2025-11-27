from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class InputData(BaseModel):
    sequence: list

@app.post("/predict")
def predict(data: InputData):
    seq = np.array(data.sequence, dtype=np.float32).reshape(1, -1, 2)

    interpreter.set_tensor(input_details[0]["index"], seq)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0][0]

    return {"predicted_wait_time": float(output)}
