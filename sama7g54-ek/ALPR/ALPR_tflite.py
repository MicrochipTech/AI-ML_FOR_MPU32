#!/usr/bin/env python3

#######################################
#
# Name : ALPR_tflite.py
#
# Author : Hakim CHERIF - M69710 <hakim.cherif@microchip.com>
#
# Description :
#   License Plate Detection and optical character recognition script
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

import argparse
import subprocess
import time
from os.path import exists

import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image, ImageDraw

SSD_PATH = 'models/ssdlite_ocr.tflite'
LPR_PATH = 'models/lprnet.tflite'
TEST_IMG = 'images/plaque_moto_2.jpg'
INFERENCE_THRESHOLD = 0.1


def start_stream_server(stream_status):
  """
  Start (or not) the streaming script thanks to a bash command. Returns the corresponding process PID.
  Argument : stream_status : boolean value.
  """
  if stream_status:
    process_stream = subprocess.Popen(["exec python3 ./flask_webserver/video_stream_flask.py"], shell=True)
    time.sleep(8)
    return process_stream
  return 0


def draw_image(image, results, size, labels=None):
  """
  Draw boxes around the detected license plates.
  """
  out_images = []
  result_size = len(results)
  for idx, obj in enumerate(results):
    #print(obj)
    # Prepare image for drawing
    draw = ImageDraw.Draw(image)

    # Prepare boundary box
    xmin, ymin, xmax, ymax = obj['bounding_box']
    xmin = int(xmin * size[1])
    xmax = int(xmax * size[1])
    ymin = int(ymin * size[0])
    ymax = int(ymax * size[0])

    # Draw rectangle to desired thickness
    for x in range(0, 4):
      draw.rectangle((ymin, xmin, ymax, xmax), outline=(255, 255, 0))

    # Annotate image with label and confidence score
    if labels is not None:
      display_str = labels[obj['class_id']] + ": " + str(round(obj['score'] * 100, 2)) + "%"
    else:
      display_str = str(obj['class_id']) + ": " + str(round(obj['score'] * 100, 2)) + "%"
    draw.text((obj['bounding_box'][0], obj['bounding_box'][1]), display_str)

    displayImage = np.asarray(image)
    out_images.append(image)
  image.show()
  image.save("img_box.jpg")
  return image


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all °output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results


def run_static(img_path):
  """
  Load the model and the input image, performs inference and display and save the output.
  """
  # Load TFLite model and allocate tensors.
  interpreter = Interpreter(model_path=SSD_PATH)
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  img_file = img_path
  locked = 1
  while locked == 1:
    try:
      img_in = Image.open(img_file).convert('RGB')
      img_in.verify()
      locked = 0
    except Exception as e:
      pass
  img_in = img_in.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
  input_image = img_in.copy()
  input_array = np.array(img_in, dtype=np.uint8)
  input_array = np.reshape(input_array, input_details[0]['shape']).astype(np.uint8)
  results = detect_objects(interpreter, input_array, INFERENCE_THRESHOLD)
  out_image = draw_image(img_in, results, img_in.size)
  if len(results) > 0:
    ymin, xmin = int(input_height * results[0]['bounding_box'][0]), int(input_width * results[0]['bounding_box'][1])
    ymax, xmax = int(input_height * results[0]['bounding_box'][2]), int(input_width * results[0]['bounding_box'][3])
    #print((xmin, xmax, ymin, ymax))
    img_in_lpr = input_image.crop((xmin, ymin, xmax, ymax))
    # Load TFLite model and allocate tensors.
    interpreter_lpr = Interpreter(model_path=LPR_PATH)
    interpreter_lpr.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter_lpr.get_input_details()
    output_details = interpreter_lpr.get_output_details()
    img_in_lpr = img_in_lpr.convert('RGB')
    img_in_lpr = img_in_lpr.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img_in_lpr.show()
    input_array = np.array(img_in_lpr, dtype=np.uint8)
    input_array = np.reshape(input_array, input_details[0]['shape'])
    interpreter_lpr.set_tensor(input_details[0]['index'], input_array)
    interpreter_lpr.invoke()

    tens_out = {'input': input_array}
    for i in range(len(output_details)):
      tens_out.update({output_details[i]['name']: interpreter_lpr.get_tensor(output_details[i]['index'])})

    char_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '<Anhui>': 10,
                 '<Beijing>': 11, '<Chongqing>': 12, '<Fujian>': 13, '<Gansu>': 14, '<Guangdong>': 15, '<Guangxi>': 16,
                 '<Guizhou>': 17, '<Hainan>': 18, '<Hebei>': 19, '<Heilongjiang>': 20, '<Henan>': 21, '<HongKong>': 22,
                 '<Hubei>': 23, '<Hunan>': 24, '<InnerMongolia>': 25, '<Jiangsu>': 26, '<Jiangxi>': 27, '<Jilin>': 28,
                 '<Liaoning>': 29, '<Macau>': 30, '<Ningxia>': 31, '<Qinghai>': 32, '<Shaanxi>': 33, '<Shandong>': 34,
                 '<Shanghai>': 35, '<Shanxi>': 36, '<Sichuan>': 37, '<Tianjin>': 38, '<Tibet>': 39, '<Xinjiang>': 40,
                 '<Yunnan>': 41, '<Zhejiang>': 42, '<police>': 43, 'A': 44, 'B': 45, 'C': 46, 'D': 47, 'E': 48, 'F': 49,
                 'G': 50, 'H': 51, 'I': 52, 'J': 53, 'K': 54, 'L': 55, 'M': 56, 'N': 57, 'O': 58, 'P': 59, 'Q': 60,
                 'R': 61, 'S': 62, 'T': 63, 'U': 64, 'V': 65, 'W': 66, 'X': 67, 'Y': 68, 'Z': 69, '_': 70}

    out = interpreter_lpr.get_tensor(output_details[0]['index'])
    out_char_codes = [np.argmax(out[0][i]) for i in range(out.shape[1])]

    out_char = []
    prev_char = None
    for i, char_code in enumerate(out_char_codes):
      if char_code == 70 or char_code == prev_char:
        continue
      prev_char = char_code
      for k, v in char_dict.items():
        if char_code == v:
          out_char.append(k)
          continue

    print(out_char)
    with open("LP.txt", "w") as f:
      for c in out_char:
        f.write(str(c))
      # for oc in out_char:
      #  f.write(oc+"\n")
    return 1
  else:
    print("***No License Plate detected***\n")
    with open("LP.txt", "w") as f:
        f.write("***No License Plate detected***")
    return 1


def run(stream_status):
  """
  Perform the license plate recognition
  """
  stream_process_id = start_stream_server(stream_status)
  savePath = 'img.png'
  run_completed = 1
  try:
    while True:
      if run_completed == 1 and exists(savePath):
        run_completed = 0
        run_completed = run_static(savePath)
  except KeyboardInterrupt:
    if stream_status:
      stream_process_id.terminate()
    print("\n***Thanks for using MPU32 Marketing's demo***")
    print("***Exiting***")
  except IOError:
    print('IO error')


if __name__ == "__main__":
  print("")
  print("***********************************************************")
  print("***        Welcome to the SAMA7G54-Ek ALPR demo         ***")
  print("***                 Using a USB Camera                  ***")
  print("***     Made with love by the MPU32 Marketing Team      ***")
  print("***         Feel free to contact us if needed           ***")
  print("***********************************************************")

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-i',
    '--image',
    help='image to be classified')
  parser.add_argument(
    '-s',
    '--stream',
    default=1,
    choices=('0', '1'),
    help='set value 1 to activate stream')

  args = parser.parse_args()

  if args.image:
    run_static(args.image)
    print("\n***Thanks for using our demo***")
  else:
    run(bool(int(args.stream)))
