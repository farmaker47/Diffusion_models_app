import json
from os.path import dirname, join
import numpy as np
import requests

# Below implementation is in python because there were a lot of memory issues inside android

# Run the diffusion model.
# Change the ADDRESS parameter based on yours.
def runDiffusionModel(context, unconditional_context):
  #filename = join(dirname(__file__), "method.tflite")
  #filename = "/data/data/com.example.diffusionmodelsapp/files/diffusion_model_17.tflite"
  #print(filename)
  # Load the TFLite model and allocate tensors.
  #interpreter = tf.lite.Interpreter(model_path=filename)
  #interpreter.allocate_tensors()

  #context = np.array(context)
  #unconditional_context = np.array(unconditional_context)
  #print(type(context.tolist()))

  input_data_for_model = {
      'context' : np.array(context).tolist(),
      'unconditional_context' : np.array(unconditional_context).tolist()
  }
  '''
  print(type(np.array(context).tolist()))
  input_data_for_model = {
        'context' : 3,
        'unconditional_context' : 4
    }
  '''
  data = json.dumps(input_data_for_model)

  ADDRESS = "http://182b-35-237-48-189.ngrok.io/diffusion_model_inferenc"
  response = requests.post(ADDRESS, data=data)
  json_response = json.loads(response.text)
  json_response = json.loads(json_response)

  #print(json_response['output'])
  print(type(json_response['output']))
  print(np.array(json_response['output']).shape)

  return json_response['output'] #np.array(json_response['outputs'])

# Run the decoder model.
# Change the ADDRESS parameter based on yours.
def runDecoderModel(latent):
  latent = np.array(latent)

  data = json.dumps(
      {
          "signature_name": "serving_default",
          "inputs": {
              "latent": latent.tolist(),
          }
  })

  ADDRESS = "35.193.53.74"

  headers = {
      "content-type": "application/json"
  }

  response = requests.post(
      f"http://{ADDRESS}:8501/v1/models/decoder:predict", data=data, headers=headers
  )

  json_response = json.loads(response.text)


  print(type(json_response['outputs']))
  print(np.array(json_response['outputs']).shape)

  return np.array(json_response['outputs'])
