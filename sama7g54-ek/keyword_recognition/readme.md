
![microchip-logo](readme/mchp-logo.png)
# Keyword recognition demo for the SAMA7G54-Ek board

This repository contains the source files needed to run the [TensorFlow Simple audio recognition: Recognizing keywords](https://www.tensorflow.org/tutorials/audio/simple_audio) demo on the SAMA7G54-Evaluation Kit. 

## Prepare the SD Card 

Follow the instructions provided [right-here](../) to flash your SD Card with a dedicated Linux4Sam distribution.

## Running the demo

To discover how to run the demo, follow this tutorial :
> [Keyword recognition demo - SAMA7G54-Ek Board - Tutorial](https://www.hackster.io/hakim-cherif/keyword-recognition-demo-for-the-sama7g54-ek-board-5010fa)

## Files and directories : 

| File or directory | Description |
|---|---|
|**audio_reco_inference.py**| Perpetual audio recognition (infinite loop) |
|**audio_reco_inference**|Audio recognition launched by pressing the user button on the board.|
|**simple_audio_model_numpy.tflite**| Tensorflow Lite model used for the recognition |
|**audio_files/** | Contains few audio examples |
|**readme/** | Contains ressources used by this readme|
