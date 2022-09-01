
![microchip-logo](readme/mchp-logo.png)
# Image classification demo for the SAMA7G54-Ek board

This repository contains the source files needed to run the [TensorFlow Image classification](https://www.tensorflow.org/lite/examples/image_classification/overview) demo on the SAMA7G54-Evaluation Kit. 

This demo works either with a MIPI Camera, or with an USB Camera. 

## Prepare the SD Card 

Follow the instructions provided [right-here](../) to flash your SD Card with a dedicated Linux4Sam distribution.

## Running the demo

### Using a MIPI Sensor

To discover how to run the demo using a MIPI Camera Sensor, follow this tutorial :
> [Image classification demo - MIPI Camera - SAMA7G54-Ek Board - Tutorial](https://www.hackster.io/hakim-cherif/image-classification-demo-for-the-sama7g54-ek-board-mipi-5332c1)

### Using an USB Camera

To discover how to run the demo using an USB Camera, follow this tutorial :
> [Image classification demo - USB Camera - SAMA7G54-Ek Board - Tutorial](https://www.hackster.io/hakim-cherif/image-classification-demo-for-the-sama7g54-ek-board-usb-52a973)

## Files and directories : 

### MIPI_Camera/ :
| File or directory | Description |
|---|---|
|**mipi_object_reco_flask.py**| Object recognition using MIPI Camera and supporting a webserver |
|**idle_benchmarking.py**| Benchmark microprocessor's performances when it is idle |
|**white_balance.py**| Press the user button with a color checker card in front of the camera |
|**mobilenet_v1_1.0_224_quant.tflite** | Machine Learning model used for the recognition |
|**labels.txt** | The 1000 labels of the objects supported by the model |
|**images/**|Examples of images|
|**mipi_bash_files/** | Contains some bash files used for MIPI Camera initialisation and use purposes |
|**flask_webserver/** | Files needed to run the flask webserver |

---
### USB_Camera/ :

| File or directory | Description |
|---|---|
|**img_reco_with_pressed_button.py**| Object recognition when the user button is pressed|
|**infinite_camera_object_reco.py**|Perpetual object recognition (infinite loop) |
|**static_img_reco.py** | Static object recognition, performed by inputting an image file |
|**mobilenet_v1_1.0_224_quant.tflite** | Machine Learning model used for the recognition |
|**labels.txt** | The 1000 labels of the objects supported by the model |
|**images/**|Examples of images|
|**flask_webserver/** | Files needed to run the flask webserver |

---
