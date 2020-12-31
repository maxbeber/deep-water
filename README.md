# Deep Water

This projects track changes in water level using satellite imagery and deep learning.

A demo video is available [here](https://drive.google.com/file/d/1iATFNuEvBrYWUtnZvZTDVe_R_z8LpgAA/view?usp=sharing).

**Table of Content:**
1. Introduction
2. Virtual Environment
3. Datasets
4. Data Augmentation
5. Baseline
6. Model Optimization
7. Dashboard
8. Next Steps

## Introduction

The motivation for this study is this [article](https://www.nationalgeographic.com/magazine/2018/03/drying-lakes-climate-change-global-warming-drought/) written in March 2018 from the [National Geographic](https://www.nationalgeographic.com/) magazine.

> Freshwater is the most important resource for mankind, cross-cutting all social, economic and environmental activities. It is a condition for all life on our planet, an enabling limiting factor for any social and technological development, a possible source of welfare or misery, cooperation or conflict. (UNESCO)

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

## Next Steps

The topics below can be studied and analysed in the context of the project:

- Apply post-processing techniques such as defrosting;
- Collect satellite imagery with clouds;
- Collect more data using the [sentinelsat](https://pypi.org/project/sentinelsat/) package;
- Estimate the volume of a given lake;