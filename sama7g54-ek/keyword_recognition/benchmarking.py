#######################################
#
# Name : benchmarking.py
#
# Author : Hakim CHERIF - M69710 <hakim.cherif@microchip.com>
#
# Description :
#   Benchmarking of the ML Demo applications
#   Running on SAMA7G54-Ek Board.
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
import psutil
import os
import queue
import time


# with open("../log.csv", 'w') as f:
#   f.write("timestamp;id;inference_status;total_vm;total_cpu;python_vm;python_cpu;python_vm_mb\n")


def benchmarking(id_queue, inference_status_queue, stop_queue):
  with open("log.csv", 'w') as f:
    f.write("timestamp;id;inference_status;total_vm;total_cpu;python_vm;python_cpu;python_vm_mb\n")
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


def put(status_queue, status):
  status_queue.put(status)
