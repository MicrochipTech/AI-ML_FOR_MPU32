#######################################
#
# Name : img_reco_with_pressed_button.py
#
# Author : Hakim CHERIF - M69710 <hakim.cherif@microchip.com>
#
# Description :
#   Dynamic oject recognition demo controlled by the user button.
#   Press the button to launch the capture and the inference.
#######################################


# Copyright (C) 2022 Microchip Technology Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/usr/bin/env python3.8
import argparse
import subprocess

import cv2
import numpy as np
import time
from mpio import Input

import tflite_runtime.interpreter as tflite
from PIL import Image

input = Input("event0")

def load_labels(label_path):
  r"""Returns a list of labels"""
  with open(label_path, 'r') as f:
    return [line.strip() for line in f.readlines()]


def start_stream_server(stream_status):
  if stream_status:
    process_stream = subprocess.Popen(["exec python3 ./flask_webserver/video_stream_flask.py"], shell=True)
    time.sleep(8)
    return process_stream
  return 0


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
print("")
print("***********************************************************")
print("*** Welcome to the SAMA7G54-Ek Object Recognition demo  ***")
print("***        Using an USB Cam and the user button         ***")
print("***     Made with love by the MPU32 Marketing Team      ***")
print("***         Feel free to contact us if needed           ***")
print("***********************************************************")

id_button_pression = 0

model_path = 'mobilenet_v1_1.0_224_quant.tflite'
label_path = 'labels.txt'

parser = argparse.ArgumentParser()
parser.add_argument(
  '-s',
  '--stream',
  default=1,
  choices=('0', '1'),
  help='set value 1 to activate stream')
args = parser.parse_args()

start_stream_server(bool(int(args.stream)))


print("Loading module")

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print("Module loaded... running interpreter")

if __name__ == "__main__":
  model_path = 'mobilenet_v1_1.0_224_quant.tflite'
  label_path = 'labels.txt'

  print("Press the user button to launch the capture and inference")

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # get width and height
  input_shape = input_details[0]['shape']
  height = input_shape[1]
  width = input_shape[2]

  floating_model = input_details[0]['dtype'] == np.float32
  # process Stream
  try:
    while True:
      counter = 0
      locked = 1
      id_read = input.read()
      if id_read[3] == 148 and id_read[4] == 1:
        print("--------Button press detected-------------")
        while locked == 1:
          try:
            image = Image.open("img.png").resize((width, height))
            image.verify()
            locked = 0
          except Exception as e:
            pass

        input_data = np.expand_dims(image, axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        results = np.squeeze(predictions)

        top_k = results.argsort()[-1:][::-1]
        labels = load_labels(label_path)
        for i in top_k:
          if counter == 0:
            with open("result.txt", "w") as f:
              f.write(str(labels[i]))
            counter = 1
          if floating_model:

            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
          else:
            print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

        print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
        print('<======= Object Detected with Inference time ==========>')
        print(' ')
        id_button_pression = input.read()[0]
  except KeyboardInterrupt:
    print("\n***Thanks for using MPU32 Marketing's demo***")
    print("***Exiting***")
