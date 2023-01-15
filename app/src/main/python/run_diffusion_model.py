import json
from os.path import dirname, join
import numpy as np
import requests

def runModel(context, unconditional_context):
  #filename = join(dirname(__file__), "method.tflite")
  #filename = "/data/data/com.example.diffusionmodelsapp/files/diffusion_model_17.tflite"
  #print(filename)
  # Load the TFLite model and allocate tensors.
  #interpreter = tf.lite.Interpreter(model_path=filename)
  #interpreter.allocate_tensors()
  context = np.array(context)
  unconditional_context = np.array(unconditional_context)
  print(type(context.tolist()))

  data = json.dumps(
      {
          "signature_name": "serving_default",
          "inputs": {
              "batch_size": 1,
              "context": context.tolist(),
              "num_steps": 25,
              "unconditional_context": unconditional_context.tolist()
          }
  })

  headers = {
      "content-type": "application/json"
  }

  ADDRESS = "104.197.115.145"
  response = requests.post(
      f"http://{ADDRESS}:8501/v1/models/diffusion-model:predict", data=data, headers=headers
  )
  json_response = json.loads(response.text)


  print(type(json_response['outputs']))
  print(np.array(json_response['outputs']).shape)

  return np.array(json_response['outputs'])
