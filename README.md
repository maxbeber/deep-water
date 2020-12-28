# Deep Water

This projects track changes in water level using satellite imagery and deep learning.

A demo video is available [here](https://drive.google.com/file/d/1iATFNuEvBrYWUtnZvZTDVe_R_z8LpgAA/view?usp=sharing).

**Table of Content:**
1. Introduction
2. Virtual Environment
3. Datasets.
4. Dashboard

## Introduction

Tbd.

> Freshwater is the most important resource for mankind, cross-cutting all social, economic and environmental activities. It is a condition for all life on our planet, an enabling limiting factor for any social and technological development, a possible source of welfare or misery, cooperation or conflict. (Unesco)

The exponenetial growth of satellite-based information over the past four decades has provided unprecedented opportunities to improve water resource manegement.

## Virtual Environement

To setup your local environemnt it is recommended to create a virtual environment using condas. Make sure you have it installed on your computer and then execute the command below:

```conda env create -f environment.yml```

The `environment.yml` file ensures that all dependiences will be downloaded.

After the enviroment is created, it is necessary to activate the virtual environemnt as follows:

```conda activate deep-water```

The virtual environment can be deactivate in a single line of code.

```conda deactivate```

## Datasets

[NWPU-Resic45](https://www.tensorflow.org/datasets/catalog/resisc45) dataset is a pubicly available benchmark for Remote Sensing Image Scene Classification (RESIC), created by [Nortwestern Polytechnical University](https://en.nwpu.edu.cn/) (NWPU). This dataset contains 31,500 images, covering 45 scene classes (including water classes) with 700 images in each class.

## Dashboard

The dashboard can be executed with the following command:

```python app.py```