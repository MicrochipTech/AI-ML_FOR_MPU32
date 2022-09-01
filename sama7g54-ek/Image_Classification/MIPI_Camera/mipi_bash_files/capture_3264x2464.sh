#!/bin/sh

#v4l2-ctl  pixelformat=VYUY,height=2464,width=3264
fswebcam -p VYUY -r 3264x2464 -S 0 -q img.png
