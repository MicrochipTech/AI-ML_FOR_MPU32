
![microchip-logo](readme/mchp-logo.png)
# Machine Learning applications for the SAMA7G54 Evaluation Kit

This repository is dedicated to the machine learning applications on the [SAMA7G54-Evaluation Kit](https://www.microchip.com/en-us/development-tool/EV21H18A), using the Tensorflow framework.

## Prepare the SD Card 
To run the demos you will need to flash a SD Card with our custom [Linux4Sam SD Card image](sdcard.zip). (Empty image)

>Hint : This SD Card image supports the three demos presented bellow.

>Hint 2 :You can also download our [plug-and-play SD Card image](sd_cardmpu32demos.zip).

To flash the SD Card, follow this tutorial : https://www.linux4sam.org/bin/view/Linux4SAM/DemoSD

## Running the demos
Specific running instructions are provided for each demo.

## Supported Demos :

| Demo      | Description | Sensor used | Model type |
| ----------- | ----------- | ----------- | ----------- |
| [Keyword Recognition](keyword_recognition)      | 8 Keywords recognition demo       | Embedded PDMs (microphones) | CNN |
| [Object Recognition](Image_Classification) | Image recognition demo | MIPI CSI 2 Sensor or USB Camera | MobileNets|
|[ALPR](ALPR) | Automatic Chinese License Plate Recognition | USB Camera | MobileNets and OCR |

## File : menu_demos
The `menu_demos` file is bash file you can use to launch the demos.
To use it, run this commmand : 
```
./menu_demos
```
Once the menu is started, you should see something like : 
```
# ./menu_demos
***********************************************************
*** Welcome to the SAMA7G54-Ek Demos selection menu     ***
***     Made with love by the MPU32 Marketing Team      ***
***********************************************************
1) Object recognition demos
2) Keyword recognition demos
3) Automatic License Plate Recognition demo
4) Quit

Press [ENTER] to display the options if they are not shown
Please choose an option :

```
>Browse through the demos by typing the corresponding number and pressing [ENTER].


![SAMA7G5 image](readme/sama7g54.jpg)