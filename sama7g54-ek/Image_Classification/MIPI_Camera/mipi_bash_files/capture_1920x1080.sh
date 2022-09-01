#!/bin/sh

#v4l2-ctl  pixelformat=VYUY,height=1080,width=1920
fswebcam -p VYUY -r 1920x1080 -S 0 -q img.png

