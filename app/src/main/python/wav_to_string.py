import numpy as np
#import tflite_runtime.interpreter as tflite

def getStringFromWav(string):
  #with open(pathWav, 'rb') as wav_file:
    #wav_data = wav_file.read()

  # load TFLite model and set params
  '''
  interpreter = tflite.Interpreter(model_path=pathTFLite)
  interpreter.allocate_tensors()
  input_index = interpreter.get_input_details()[0]["index"]
  print(type(input_index))
  output_index = interpreter.get_output_details()[0]["index"]

  # run inference
  interpreter.set_tensor(input_index, np.array(wav_data))
  interpreter.invoke()
  outputs = interpreter.get_tensor(output_index)
  print(type(outputs))
  '''
    
  return string