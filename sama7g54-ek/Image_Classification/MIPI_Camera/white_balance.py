import subprocess

from mpio import Input
input = Input("event0")


id_read = input.read()
if id_read[3] == 148 and id_read[4] == 1:
  print("***White balance")
  subprocess.Popen(["exec v4l2-ctl --set-ctrl=white_balance_automatic=0"],shell=True)
  subprocess.Popen(["exec v4l2-ctl --set-ctrl=do_white_balance=1"],shell=True)