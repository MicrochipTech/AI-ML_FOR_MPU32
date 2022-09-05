
![microchip-logo](readme/mchp-logo.png)
# Customize a Linux4Sam distribution for Machine Learning purposes

This directory contains files and folders that allow you to customize for machine learning purposes the [Microchip's Buildroot Distribution](https://github.com/linux4sam/buildroot-at91) that generates [Linux4SAM](https://www.linux4sam.org/) OS images.


## How to customize :

Copy the content of this directory to the folder containing the "**buildroot-at91/**" directory.

### Step 1 : Change Python Version

[Tensorflow Lite Runtime](https://pypi.org/project/tflite-runtime/) does not support latest Python version. This is why to use it, you need to change it to another python version, here it is the 3.8.6 python version :

Run :
```
$ python3 ./change_python_version_to_3.8.6.py
```
You should get :
```
***
***Removing current python version***
***Successfully removed current python version***
***Installing new version***
***Successfully changed python version***
```

### Step 2 : Add SciPy

[SciPy](https://scipy.org/) is a mathematic signal processing python library. If you need it you can add it to buildroot-at91 by running this command : 

```
$ git apply 0001-Add_SciPy_version_1.8.1.patch
```
Once these steps are done, you are ready to generate a Linux4SAM SD Card image that supports tflite_runtime and SciPy. 

![SAMA7G5 image](readme/sama7g54.jpg)