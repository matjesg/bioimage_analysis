# bioimage_analysis
Evaluation of biological effects from microscopy images

Official repository of our study "On the objectivity, reliability, and validity of deep learning enabled bioimage analyses". You can find a preprint of our paper on [Biorxiv](https://www.biorxiv.org/content/10.1101/473199v2).

We provide all source code and all source data of our study for the open science community. We also share all of our training datasets.

Furthermore, we will also provide two entire bioimage datasets (Lab-Wue1 and Lab-Mue), including all microscopy images and the annotations of all DL-based approaches, as soon as our paper is published.


## Local installation

To reproduce our experiments a Python (>= 3.6) installation is required. Run the following code to set up the environment
```
git clone https://github.com/matjesg/bioimage_analysis.git
cd bioimage_analysis
pip install requirements.txt
```

The code requires a TensorFlow 1 installation (tested on TensorFlow 1.14.0 and 1.15.2). For more information see the [TensorFlow install guide](https://www.tensorflow.org/install).