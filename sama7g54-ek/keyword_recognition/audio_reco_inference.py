#######################################
#
# Name : audio_reco_inference_button.py
#
# Author : Hakim CHERIF - M69710 <hakim.cherif@microchip.com>
#
# Description :
#   Audio Keywords recognition demo
#   Running on SAMA7G54-Ek.
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
import threading

from scipy.io import wavfile
from scipy import signal
from scipy.special import softmax

import numpy as np
import argparse
import subprocess
import time
import benchmarking
import queue

from tflite_runtime.interpreter import Interpreter

BENCHMARK = 0

VERBOSE_DEBUG = 0
VERBOSE_PERFORMANCES = 1


def run(benchmark_status):
  """
    Get audio input thanks to subprocess and arecord instruction from ALSA.
  """
  if benchmark_status:
    id_queue = queue.Queue()
    inference_status_queue = queue.Queue()
    stop_queue = queue.Queue()
    id_queue.put(0)
    inference_status_queue.put(0)
    stop_queue.put(0)
    p_benchmark = threading.Thread(target=benchmarking.benchmarking,
                                   args=(id_queue, inference_status_queue, stop_queue,))
    p_benchmark.start()
  else:
    inference_status_queue = 0
    id_queue = 0
    stop_queue = 0

  DEVICE = "plughw:0,4"
  FORMAT = "S32_LE"
  DURATION = 3
  RATE = 16000
  WAVE_OUTPUT_FILENAME = "recording.wav"
  CROP = 14000
  i = 0
  try:
    while True:
      if benchmark_status:
        id_queue.put(i)
        inference_status_queue.put("starting")
      print("\nWill begin audio recording soon")
      time.sleep(1)
      print("3...")
      time.sleep(1)
      print("2...")
      time.sleep(1)
      print("1...")
      if benchmark_status:
        inference_status_queue.put("recording")
      process = subprocess.Popen(
        ["arecord", "-D" + DEVICE, "-f" + FORMAT, "-d " + str(DURATION), "-r" + str(RATE), "-A 50", "-q",
         WAVE_OUTPUT_FILENAME])  # Starting to record
      time.sleep(1)
      print("Go !")
      process.wait()
      print("Recording done\n")
      if benchmark_status:
        inference_status_queue.put("extracting_clean_audio")
      rate_rec, waveform_rec = wavfile.read(WAVE_OUTPUT_FILENAME)
      zeros_completion = np.zeros(CROP, dtype=int)
      run_inference(np.concatenate(
        (zeros_completion, waveform_rec[CROP:])), inference_status_queue,
        benchmark_status)  # Extract the part of the signal without PCM commutation noise
      i = i + 1
  except KeyboardInterrupt:
    if benchmark_status:
      stop_queue.put(1)
    print("\n***Thanks for using MPU32 Marketing's demo***")
    print("***Exiting***")


def process_audio_data(waveform):
  """Process audio input.
  This function takes in raw audio data from a WAV file and does scaling
  and padding to 16000 length.
  """

  if VERBOSE_DEBUG:
    print("waveform:", waveform.shape, waveform.dtype, type(waveform))
    print(waveform[:5])

  # if stereo, pick the left channel
  if len(waveform.shape) == 2:
    print("Stereo detected. Picking one channel.")
    waveform = waveform.T[1]
  else:
    waveform = waveform

  if VERBOSE_DEBUG:
    print("After scaling:")
    print("waveform:", waveform.shape, waveform.dtype, type(waveform))
    print(waveform[:5])

  # normalise audio
  wabs = np.abs(waveform)
  wmax = np.max(wabs)
  waveform = waveform / wmax

  PTP = np.ptp(waveform)
  # print("peak-to-peak: %.4f. Adjust as needed." % (PTP,))

  # return None if too silent
  if PTP < 0.5:
    return []

  if VERBOSE_DEBUG:
    print("After normalisation:")
    print("waveform:", waveform.shape, waveform.dtype, type(waveform))
    print(waveform[:5])

  # scale and center
  waveform = 2.0 * (waveform - np.min(waveform)) / PTP - 1

  # extract 16000 len (1 second) of data
  max_index = np.argmax(waveform)
  start_index = max(0, max_index - 8000)
  end_index = min(max_index + 8000, waveform.shape[0])
  waveform = waveform[start_index:end_index]

  # Padding for files with less than 16000 samples
  if VERBOSE_DEBUG:
    print("After padding:")

  waveform_padded = np.zeros((16000,))
  waveform_padded[:waveform.shape[0]] = waveform

  if VERBOSE_DEBUG:
    print("waveform_padded:", waveform_padded.shape, waveform_padded.dtype, type(waveform_padded))
    print(waveform_padded[:5])

  return waveform_padded


def get_spectrogram(waveform, inference_status_queue, benchmark_status):
  if benchmark_status:
    inference_status_queue.put("audio_processing")
  waveform_padded = process_audio_data(waveform)
  if benchmark_status:
    inference_status_queue.put("spectrogram_extraction")

  if not len(waveform_padded):
    return []

  # compute spectrogram
  f, t, Zxx = signal.stft(waveform_padded, fs=16000, nperseg=255,
                          noverlap=124, nfft=256)
  # Output is complex, so take abs value
  spectrogram = np.abs(Zxx)

  if VERBOSE_DEBUG:
    print("spectrogram:", spectrogram.shape, type(spectrogram))
    print(spectrogram[0, 0])

  return spectrogram


def run_inference(waveform, inference_status_queue, benchmark_status):
  # get spectrogram data
  spectrogram = get_spectrogram(waveform, inference_status_queue, benchmark_status)
  if benchmark_status:
    inference_status_queue.put("preparing_inference")

  if not len(spectrogram):
    # disp.show_txt(0, 0, "Silent. Skip...", True)
    print("Too silent. Skipping...")
    # time.sleep(1)
    return

  spectrogram1 = np.reshape(spectrogram, (-1, spectrogram.shape[0], spectrogram.shape[1], 1))

  if VERBOSE_DEBUG:
    print("spectrogram1: %s, %s, %s" % (type(spectrogram1), spectrogram1.dtype, spectrogram1.shape))

  # load TF Lite model
  interpreter = Interpreter('simple_audio_model_numpy.tflite')
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # print("Input shape=", input_shape)
  input_data = spectrogram1.astype(np.float32)
  # print("Type de input data", type(input_data))
  interpreter.set_tensor(input_details[0]['index'], input_data)

  print("Running inference...")

  time_inference_start = time.time()
  # initial_cpu_charge=psutil.cpu_percent()
  if benchmark_status:
    inference_status_queue.put("inference")
  interpreter.invoke()
  time_inference_stop = time.time()
  if benchmark_status:
    inference_status_queue.put("results")
  print("Inference done in {:.2f} ms ".format((time_inference_stop - time_inference_start) * 1000))
  if benchmark_status:
    with open("log_time_ms.csv", "a") as h:
      h.write('{:.3f}\n'.format((time_inference_stop - time_inference_start) * 1000))
  output_data = interpreter.get_tensor(output_details[0]['index'])
  commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

  if VERBOSE_DEBUG:
    print(output_data[0])
  print(">>> Key Word Detected --> " + commands[np.argmax(output_data[0])].upper())
  print("See all the outputs bellow : ")
  for i in range(len(commands)):
    print("Score for label " + commands[i] + " is {:.2f} % ".format(softmax(output_data[0])[i] * 100))
  time.sleep(3)


if __name__ == '__main__':

  print("")
  print("***********************************************************")
  print("*** Welcome to the SAMA7G54-Ek Audio Recognition demo   ***")
  print("***     Made with love by the MPU32 Marketing Team      ***")
  print("***         Feel free to contact us if needed           ***")
  print("***********************************************************")

  # create parser
  descStr = """
  This program does ML inference on audio data.
  """
  parser = argparse.ArgumentParser(description=descStr)
  # add a mutually exclusive group of arguments
  group = parser.add_mutually_exclusive_group()

  # add expected arguments
  group.add_argument('--input', dest='wavfile_name', required=False)
  group.add_argument(
    '-b',
    '--benchmarking',
    default=BENCHMARK,
    choices=('0', '1'),
    help='set value 1 to activate benchmark')

  # parse args
  args = parser.parse_args()

  # disp = SSD1306_Display()

  # test WAV file
  if args.wavfile_name:
    wavfile_name = args.wavfile_name
    print("Reading the input wavefile :", wavfile_name)
    # get audio data
    rate, waveform = wavfile.read(wavfile_name)
    print("Reading of the file is successful")
    # run inference
    run_inference(waveform, 0, 0)
  else:
    print("Starting Audio processing\n")
    run(bool(int(args.benchmarking)))

  print("done.")
