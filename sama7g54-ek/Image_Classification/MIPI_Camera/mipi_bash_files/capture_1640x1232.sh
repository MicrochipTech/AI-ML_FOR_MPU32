#!/bin/sh

#v4l2-ctl  pixelformat=VYUY,height=1232,width=1640
fswebcam -p VYUY -r 1640x1232 -S 0 -q img.png

