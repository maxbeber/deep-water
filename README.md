# Deep Water

This project aims to track changes in water level using satellite imagery and deep learning. As part of my studies, I worked on this project with my colleague Karl as part of our portfolio project of the Data Science Retreat. The retreat consists of a three months intensive in-person bootcamp in Berlin.

**Table of Content:**
1. Introduction
2. Datasets
3. Data Augmentation
4. Baseline
5. Model Optimization
6. Dashboard
7. Technical Stack
8. Virtual Environment
9. Next Steps

## Introduction

The motivation for this project is the article [Some of the World's Biggest Lakes Are Drying Up](https://www.nationalgeographic.com/magazine/2018/03/drying-lakes-climate-change-global-warming-drought/) found in the March 2018 edition of the [National Geographic](https://www.nationalgeographic.com/) magazine.

> Freshwater is the most important resource for mankind, cross-cutting all social, economic and environmental activities. It is a condition for all life on our planet, an enabling limiting factor for any social and technological development, a possible source of welfare or misery, cooperation or conflict. (UNESCO)

The exponenetial growth of satellite-based information over the past four decades has provided unprecedented opportunities to improve water resource manegement.

## Datasets

[NWPU-Resic45](https://www.tensorflow.org/datasets/catalog/resisc45) dataset is a pubicly available benchmark for Remote Sensing Image Scene Classification (RESIC), created by [Nortwestern Polytechnical University](https://en.nwpu.edu.cn/) (NWPU). This dataset contains 31,500 images, covering 45 scene classes (including water classes) with 700 images in each class.

The second dataset is a time-series of cloudless Sentinel-2 imagery including 17 criticaly endangered lakes as following:
- [Lake Poopo](https://en.wikipedia.org/wiki/Lake_Poop%C3%B3), Bolivia
- [Lake Urmia](https://en.wikipedia.org/wiki/Lake_Urmia), Iran
- [Lake Mojave](https://en.wikipedia.org/wiki/Lake_Mohave), USA
- [Aral sea](https://en.wikipedia.org/wiki/Aral_Sea), Kazahkstan
- [Lake Copais](https://en.wikipedia.org/wiki/Lake_Copais), Greece
- [Lake Ramganga](https://en.wikipedia.org/wiki/Ramganga_Dam), India
- [Qinghai Lake](https://en.wikipedia.org/wiki/Qinghai_Lake), China
- [Salton Sea](https://en.wikipedia.org/wiki/Salton_Sea), USA
- [Lake Faguibine](https://earthobservatory.nasa.gov/images/8991/drying-of-lake-faguibine-mali), Mali
- [Mono Lake](https://en.wikipedia.org/wiki/Mono_Lake), USA
- [Walker Lake](https://en.wikipedia.org/wiki/Walker_Lake_(Nevada)), USA
- [Lake Balaton](https://en.wikipedia.org/wiki/Lake_Balaton), Hungary
- [Lake Koroneia](https://en.wikipedia.org/wiki/Lake_Koroneia), Greece
- [Lake Salda](https://en.wikipedia.org/wiki/Lake_Salda), Turkey
- [Lake Burdur](https://en.wikipedia.org/wiki/Lake_Burdur), Turkey
- [Lake Mendocino](https://en.wikipedia.org/wiki/Lake_Mendocino), USA
- [Elephant Butte Reservoir](https://en.wikipedia.org/wiki/Elephant_Butte_Reservoir), USA

## Data Augmentation

The following T

## Dashboard

The dashboard can be executed with the following command:

```python app.py```

A demo is available [here](https://drive.google.com/file/d/1iATFNuEvBrYWUtnZvZTDVe_R_z8LpgAA/view?usp=sharing).

## Technical Stack

The following libraries are required to create the virtual environment. The creation of the virtual environment is detailed in the next section.

- Cython
- Dash
- Matplotlib
- NumPy
- Pillow
- Pydensecrf
- Rasterio
- Requests
- Tensorflow 2.4

## Virtual Environement

To setup your local environemnt it is recommended to create a virtual environment using condas. Make sure you have it installed on your computer and then execute the command below:

```conda env create -f environment.yml```

The `environment.yml` file ensures that all dependiences will be downloaded.

After the enviroment is created, it is necessary to activate the virtual environemnt as follows:

```conda activate deep-water```

The virtual environment can be deactivate in a single line of code.

```conda deactivate```

## Next Steps

The topics below can be studied and analysed in the context of the project:

- Apply post-processing techniques such as defrosting;
- Collect satellite imagery with clouds;
- Collect more data using the [sentinelsat](https://pypi.org/project/sentinelsat/) package;
- Estimate the volume of a given lake;