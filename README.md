# bioimage_analysis
Evaluation of biological effects from microscopy images

Official GitHub repository of our study "On the objectivity, reliability, and validity of deep learning enabled bioimage analyses". You can find a preprint of our paper on [Biorxiv](https://www.biorxiv.org/content/10.1101/473199v2), and all source data, our open-source model library, two entire bioimage datasets, as well as the source code in our [Dryad repository](https://doi.org/10.5061/dryad.4b8gtht9d). 

Here, we provide all source code that can be used to reproduce our analyses and a Jupyter Notebook, optimized for the use in Google Colab, to [train deep learning models](https://colab.research.google.com/github/matjesg/bioimage_analysis/blob/master/notebooks/dl1_train_and_select.ipynb) according to the strategies we investigate in our study. Furthermore, we also share all of our training datasets with the open science community.


## Local installation

To reproduce our experiments a Python (>= 3.6) installation is required. Run the following code to set up the environment
```
git clone https://github.com/matjesg/bioimage_analysis.git
cd bioimage_analysis
pip install requirements.txt
```

The code requires a TensorFlow 1 installation (tested on TensorFlow 1.14.0 and 1.15.2). For more information see the [TensorFlow install guide](https://www.tensorflow.org/install).
