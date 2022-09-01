
![microchip-logo](readme/mchp-logo.png)
# ALPR demo for the SAMA7G54-Ek board

This repository contains the source files needed to run an Automatic Chinese Licence Plate Recognition demo on the SAMA7G54-Evaluation Kit. 

This demo works either with an USB Camera. 

## Prepare the SD Card 

Follow the instructions provided [right-here](../) to flash your SD Card with a dedicated Linux4Sam distribution.

## Running the demo

To discover how to run the demo follow this tutorial :
> [ALPR demo - USB Camera - SAMA7G54-Ek Board - Tutorial](https://www.hackster.io/hakim-cherif/automatic-license-plate-recognition-demo-on-the-sama7g54-ek-11971b)


## Files and directories : 

| File or directory | Description |
|---|---|
|**ALPR_tflite.py**| Automatic License Plate Recognition Script |
|**img.png** & **img_box.jpg**|Blank images needed by the script|
|**LP.txt**| Blank text file needed by the script |
|**models/**| Machine Learning models used for the recognition |
|**flask_webserver/** | Scipts and files related to the webserver |
