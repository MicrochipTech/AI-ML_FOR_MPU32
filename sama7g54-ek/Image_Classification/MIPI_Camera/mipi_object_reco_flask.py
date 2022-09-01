#######################################
#
# Name : mipi_object_reco_flask.py
#
# Author : Hakim CHERIF - M69710 <hakim.cherif@microchip.com>
#
# Description :
#   Image classification/object recognition demo.
#   Running on SAMA7G54-Ek Board.Using a
#   MIPI Sensor and supporting a webserver.
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
import os
import threading
import queue

import numpy as np
import argparse
import subprocess
import time
import tflite_runtime.interpreter as tflite
import psutil

import PIL
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFINITION_PX = "640x480"

BENCHMARK = 0

with open("log.csv", 'w') as f:
  f.write("timestamp;id;inference_status;total_vm;total_cpu;python_vm;python_cpu;python_vm_mb\n")


def benchmarking(id_queue, inference_status_queue, stop_queue):
  current_pid = os.getpid()
  with open("log.csv", 'a') as g:
    while 1:
      try:
        stop = list(stop_queue.queue)[-1]
        if stop:
          raise KeyboardInterrupt
        id = list(id_queue.queue)[-1]
        inference_status = list(inference_status_queue.queue)[-1]
        vm = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=0.1)
        current_p = psutil.Process(current_pid)
        current_cpu = current_p.cpu_percent()
        current_vm = current_p.memory_percent()
        current_vm_mb = current_p.memory_info().rss >> 20
        timestamp = time.time()
        g.write(
          str(timestamp).replace('.', ',') + ";" + str(id) + ";" + inference_status + ";" + str(vm).replace(
            '.', ',') + ";" + str(cpu).replace('.', ',') + ";" + str(current_vm).replace('.', ',') + ";" + str(
            current_cpu).replace('.', ',') + ";" + str(current_vm_mb) + "\n")
      except queue.Empty:
        print("Queue empty")
        pass
      except KeyboardInterrupt:
        print("***Stopping benchmarking process***")
        break


def init_mipi_sensor(definition_px):
  """
  Configure the MIPI Sensor
  """
  if definition_px in ["640x480", "1640x1232", "1920x1080", "3264x2464"]:
    process_init_mipi = subprocess.Popen(["source mipi_bash_files/" + definition_px + ".sh"], shell=True)
    process_init_mipi.wait()
    return definition_px
  else:
    raise "Unknown definition/image format ! Exiting...."


def start_stream_server(stream_status):
  if stream_status:
    process_stream = subprocess.Popen(["exec python3 ./flask_webserver/video_stream_flask.py"], shell=True)
    time.sleep(8)
    return process_stream
  return 0


def load_labels(label_path):
  with open(label_path, 'r') as f:
    return [line.strip() for line in f.readlines()]


def load_model(model_path, num_threads):
  interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
  interpreter.allocate_tensors()
  return interpreter


def process_image(interpreter, img_path):
  """
  Process the image that will be given to the model
  """
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  locked = 1
  while locked == 1:
    try:
      img = Image.open(img_path).resize((width, height))
      img.verify()
      locked = 0
    except Exception as e:
      pass
  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  return input_data, input_details, output_details, floating_model


def inference(input_data, input_details, output_details, interpreter, inference_status_queue, benchmark_status):
  """
  Run the inference, measure of time execution
  """
  interpreter.set_tensor(input_details[0]['index'], input_data)
  if benchmark_status:
    inference_status_queue.put("inference")
  start_time = time.time()
  interpreter.invoke()
  stop_time = time.time()
  if benchmark_status:
    inference_status_queue.put("results_processing")

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)
  return results, start_time, stop_time


def process_results(results, floating_model, start_time, stop_time, benchmark_status):
  """
  Process output results from inference, print the results
  """

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  print("<---Processing done---->")
  counter = 0
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
  if benchmark_status:
    with open("log_time_ms.csv", "a") as h:
      h.write('{:.3f}\n'.format((stop_time - start_time) * 1000))
  time.sleep(3)


def capture_img(definition_px, stream_status):
  if stream_status == 0:
    process_capture = subprocess.Popen(["source mipi_bash_files/capture_" + definition_px + ".sh"], shell=True)
    process_capture.wait()
    print("Image Captured\n")


def run(definition_px, interpreter, img_path, stream_status, benchmark_status):
  stream_process_id = start_stream_server(stream_status)
  wb_process = subprocess.Popen(["exec python3 ./white_balance.py"], shell=True)
  if benchmark_status:
    id_queue = queue.Queue()
    inference_status_queue = queue.Queue()
    stop_queue = queue.Queue()
    id_queue.put(0)
    inference_status_queue.put(0)
    stop_queue.put(0)
    p_benchmark = threading.Thread(target=benchmarking, args=(id_queue, inference_status_queue, stop_queue,))
    p_benchmark.start()
  else:
    inference_status_queue = 0
    id_queue = 0
    stop_queue = 0
  i = 0
  try:
    while True:

      if benchmark_status:
        id_queue.put(i)
        inference_status_queue.put("image_capture")
      capture_img(definition_px, stream_status)
      if benchmark_status:
        inference_status_queue.put("image_processing")
      (input_data, input_details, output_details, floating_model) = process_image(interpreter, img_path)
      (results, start_time, stop_time) = inference(input_data, input_details, output_details, interpreter,
                                                   inference_status_queue, benchmark_status)
      process_results(results, floating_model, start_time, stop_time, benchmark_status)
      if benchmark_status:
        inference_status_queue.put("end_process")
      i = i + 1

  except KeyboardInterrupt:
    if stream_status:
      stream_process_id.terminate()
    if benchmark_status:
      stop_queue.put(1)
    wb_process.terminate()
    print("\n***Thanks for using MPU32 Marketing's demo***")
    print("***Exiting***")


def run_static(interpreter, img_path):
  (input_data, input_details, output_details, floating_model) = process_image(interpreter, img_path)
  (results, start_time, stop_time) = inference(input_data, input_details, output_details, interpreter, 0, 0)
  process_results(results, floating_model, start_time, stop_time, 0)


def init(definition_px):
  definition_px = init_mipi_sensor(definition_px)
  return definition_px


if __name__ == '__main__':

  print("")
  print("***********************************************************")
  print("*** Welcome to the SAMA7G54-Ek Object Recognition demo  ***")
  print("***               Using a MIPI CSI Sensor               ***")
  print("***     Made with love by the MPU32 Marketing Team      ***")
  print("***         Feel free to contact us if needed           ***")
  print("***********************************************************")

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '-i',
    '--image',
    help='image to be classified')
  parser.add_argument(
    '-m',
    '--model_file',
    default='mobilenet_v1_1.0_224_quant.tflite',
    help='.tflite model to be executed')
  parser.add_argument(
    '-l',
    '--label_file',
    default='labels.txt',
    help='name of file containing labels')
  parser.add_argument(
    '--input_mean',
    default=127.5, type=float,
    help='input_mean')
  parser.add_argument(
    '--input_std',
    default=127.5, type=float,
    help='input standard deviation')
  parser.add_argument(
    '-s',
    '--stream',
    default=1,
    choices=('0', '1'),
    help='set value 1 to activate stream')
  parser.add_argument(
    '-b',
    '--benchmarking',
    default=BENCHMARK,
    choices=('0', '1'),
    help='set value 1 to activate benchmark')

  parser.add_argument(
    '--num_threads', default=None, type=int, help='number of threads')
  args = parser.parse_args()

  if args.image:
    interpreter_tf = load_model(args.model_file, args.num_threads)
    run_static(interpreter_tf, args.image)
    print("\n***Thanks for using our demo***")
  else:
    definition = init(DEFINITION_PX)
    interpreter_tf = load_model(args.model_file, args.num_threads)
    run(definition, interpreter_tf, "img.png", bool(int(args.stream)), bool(int(args.benchmarking)))
