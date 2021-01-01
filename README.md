# Deep Water

This project aims to track changes in water level using satellite imagery and deep learning. Throughout my studies, I've worked on this project with my friend Karl as part of our [portfolio project](https://www.meetup.com/Data-Science-Retreat/events/273185026/) of the Data Science Retreat. The retreat consists of a three months intensive in-person Data Science bootcamp in Berlin, Germany.

**Table of Content:**
1. Introduction
2. Datasets
3. Labeling
4. Data Augmentation
5. Metrics
6. Baseline
7. Model Optimization
8. Model Results
9. Dashboard
10. Technical Stack
11. Virtual Environment
12. Next Steps

## Introduction

The motivation for this project is the article [Some of the World's Biggest Lakes Are Drying Up](https://www.nationalgeographic.com/magazine/2018/03/drying-lakes-climate-change-global-warming-drought/) found in the March 2018 edition of the [National Geographic](https://www.nationalgeographic.com/) magazine.

> Freshwater is the most important resource for mankind, cross-cutting all social, economic and environmental activities. It is a condition for all life on our planet, an enabling limiting factor for any social and technological development, a possible source of welfare or misery, cooperation or conflict. ([UNESCO](https://en.unesco.org/))

The exponenetial growth of satellite-based information over the past four decades has provided unprecedented opportunities to improve water resource manegement.

## Datasets

[NWPU-Resic-45](https://www.tensorflow.org/datasets/catalog/resisc45) dataset is a pubicly available benchmark for Remote Sensing Image Scene Classification (RESIC), created by [Nortwestern Polytechnical University](https://en.nwpu.edu.cn/) (NWPU). This dataset contains 31,500 images, covering 45 scene classes (including water classes) with 700 images in each class.

The second dataset is a time-series of cloudless Sentinel-2 imagery including 17 criticaly endangered lakes as following:
- [Lake Poopo](https://en.wikipedia.org/wiki/Lake_Poop%C3%B3), Bolivia;
- [Lake Urmia](https://en.wikipedia.org/wiki/Lake_Urmia), Iran;
- [Lake Mojave](https://en.wikipedia.org/wiki/Lake_Mohave), USA;
- [Aral sea](https://en.wikipedia.org/wiki/Aral_Sea), Kazahkstan;
- [Lake Copais](https://en.wikipedia.org/wiki/Lake_Copais), Greece;
- [Lake Ramganga](https://en.wikipedia.org/wiki/Ramganga_Dam), India;
- [Qinghai Lake](https://en.wikipedia.org/wiki/Qinghai_Lake), China;
- [Salton Sea](https://en.wikipedia.org/wiki/Salton_Sea), USA;
- [Lake Faguibine](https://earthobservatory.nasa.gov/images/8991/drying-of-lake-faguibine-mali), Mali;
- [Mono Lake](https://en.wikipedia.org/wiki/Mono_Lake), USA;
- [Walker Lake](https://en.wikipedia.org/wiki/Walker_Lake_(Nevada)), USA;
- [Lake Balaton](https://en.wikipedia.org/wiki/Lake_Balaton), Hungary;
- [Lake Koroneia](https://en.wikipedia.org/wiki/Lake_Koroneia), Greece;
- [Lake Salda](https://en.wikipedia.org/wiki/Lake_Salda), Turkey;
- [Lake Burdur](https://en.wikipedia.org/wiki/Lake_Burdur), Turkey;
- [Lake Mendocino](https://en.wikipedia.org/wiki/Lake_Mendocino), USA;
- [Elephant Butte Reservoir](https://en.wikipedia.org/wiki/Elephant_Butte_Reservoir), USA.

## Labeling

The [MakeSense](https://www.makesense.ai/) online tool has been used for labeling both datasets images. It only requires a web browser and you are ready to go. It's an excellent choice for small computer vision deep learning projects, making the process of preparing the dataset easier and faster.

## Data Augmentation

The following techniques have been applied during training:

- Height shift up to 30%;
- Horizontal flip;
- Rotation up to 45 degrees;
- No shear;
- Vertical flip;
- Width shift up to 30%;
- Zoom between 75% and 125%.

## Metrics

The following metrics have been used to evaluate the semantic segmenation model:

- Jaccard Index
- Dice Coefficient

More information about both of these metrics can be found [here](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2).

## Baseline

The baseline consists of a simple U-Net model architecture. This strategy allow us to modify the model for our own purposes and fine-tunning it as necessary for our development purposes. By using this network architecture, we could spend more time understanding the optimization strategies.

### Without Data Augmentation

Train/Validation/Test splits based on Resic-45 dataset only:
- training set: 489 images;
- validation set: 140 images;
- test set: 71 images.

Model performance:

![Baseline results without image augmentation](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/baseline-no-augmentation.png)

### With Data Augmentation

Train/Validation/Test splits based on Resic-45 dataset only:
- training set: 979 images;
- validation set: 280 images;
- test set: 122 images.

Model performance:

![Baseline results using image augmentation](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/baseline-with-augmentation.png)

It can be seen clearly that the baseline model overfits using image augmentation.

## Model Optimization

The following strategies have been explored:

1. Using Early Stopping and adaptive learning rates;
2. Using a bigger model (and dropout);
3. Using regularization (Batch Normalization);
4. Using residual connections;
4. Dealing with class imbalance using dice loss;
6. Refining label images using CRFs;
7. Ensemble predictions.

Train/Validation/Test splits:
- training set: 489 images from Resic-45 dataset randomly transformed at each epoch using one of the techniques described in the fourth section _Data Augmentation_;
- validation set: 211 images from Resic-45 dataset;
- test set: 359 images from Sentinel-2 dataset.

Model performance using binary cross entropy as the loss function:

![Model optimization using binary cross entropy](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/model-optimization-binary-cross-entropy.png)

Model performance using dice loss as the loss function:

![Model optimization using dice loss](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/model-optimization-dice-loss.png)

## Model Results

The test set to measure the results presented below is based on 182 images from Sentinel-2 dataset.

Model 1: U-Net residual model trained without label correction:

![Model results: unet residual large dice](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/model-results-unet-residual-large-dice.png)

Model 2: U-Net residual model trained with label correction using Conditional Random Fields:

![Model results: unet residual large dice](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/model-results-unet-residual-large-dice-with-crf.png)

Model 3: Ensemble model based on the two previous models:

![Model results: unet residual large dice](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/model-results-combined-model.png)

The ensemble model is the one with highest accuracy (97.15%) and is the one used in the Dashboard application that will be covered in the next section.

## Dashboard

The dashboard can be executed with the following command:

```python app.py```

A demo is available [here](https://drive.google.com/file/d/1iATFNuEvBrYWUtnZvZTDVe_R_z8LpgAA/view?usp=sharing).

**Use Case 1: Lake Copais, Greece (2019)**

![Use Case 1: Lake Copais](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/use-case-lake-copais.png)

**Use Case 2: Lake Di Cancano, Italy (2019)**

![Use Case 2: Lake Di Cancano](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/use-case-lake-di-cancano.png)

**Use Case 3: Lake Salda, Turkey (2016)**

![Use Case 3: Lake Salda](https://github.com/maxbeber/deep-water/blob/develop/assets/documentation/use-case-lake-salda.png)

## Technical Stack

The following libraries are required to create the virtual environment. The creation of the virtual environment is detailed in the next section.

- Cython
- Dash
- Matplotlib
- NumPy
- Pillow
- Plotly
- [Pydensecrf](https://github.com/lucasb-eyer/pydensecrf)
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
- Estimate the volume of a given water body.