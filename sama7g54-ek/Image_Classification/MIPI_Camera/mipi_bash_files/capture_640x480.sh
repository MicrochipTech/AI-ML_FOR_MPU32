#!/bin/sh

#v4l2-ctl -v pixelformat=VYUY,height=480,width=640 --silent
fswebcam -p VYUY -r 640x480 -S 0 -q img.png

