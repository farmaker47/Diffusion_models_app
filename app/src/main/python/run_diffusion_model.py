import json
from os.path import dirname, join
import numpy as np
import requests

# Below implementation is in python because there were a lot of memory issues inside android

# Run the diffusion model.
# Change the ADDRESS parameter based on yours.
def runDiffusionModel(context, unconditional_context):
  input_data_for_model = {
      'context' : np.array(context).tolist(),
      'unconditional_context' : np.array(unconditional_context).tolist()
  }
  data = json.dumps(input_data_for_model)

  ADDRESS = "http://2545-34-82-213-91.ngrok.io/diffusion_model_inferenc"
  response = requests.post(ADDRESS, data=data)
  json_response = json.loads(response.text)
  json_response = json.loads(json_response)

  #print(json_response['output'])
  print(type(json_response['output']))
  print(np.array(json_response['output']).shape)

  return json_response['output'] #np.array(json_response['outputs'])
