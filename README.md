# CS230 Semantic Segmentation of Extreme Climate Events


Tropical cyclones (TCs), also known as hurricanes, typhoons or tropical storms, are the most destructive type of extreme weather events and have caused $1.1 trillion in damage and 6,697 deaths since 1980 in the US alone. 

In this project, we apply the light-weight CGNet context guided computer vision architecture to semantic segmentation for the identification of tropical cyclones in climate data.


## Directory structure:

baseline
    baseline.ipynb –- Notebook to evaluation the baseline implementation on train and test sets
    config.json -- Configuation for baseline implementation
    weights.pth -- Pre-trained weights for baseline implementation

climatenet -- ClimateNet library for baseline implementation
    climatenet
    example.py
    README.MD

data
    data_exploration_climatenet.ipynb -- Notebook to analyze and visualize the ClimateNet dataset
    download_climatenet.ipynb -- Script to download the ClimateNet dataset

README.MD


## Data

ClimateNet is an open, community-sourced, human expert-labeled data set, mapping the outputs of Community Atmospheric Model (CAM5.1) climate simulation runs, for 459 time steps from 1996 to 2013. 

Each example is a netCDF file containing an array (1152, 768) for one time step, with each pixel mapping to a (latitude, longitude) point, with 16 channels for key atmospheric variables and one class label.

![](data/climatenet_channels.png.png)
 

| Channel | Description                                               | Units  | 
|---------|-----------------------------------------------------------|--------|
| TMQ     | Total (vertically integrated) precipitable water          | kg/m^2 | 
| U850    | Zonal wind at 850 mbar pressure surface                   | m/s    | 
| V850    | Meridional wind at 850 mbar pressure surface              | m/s    | 
| UBOT    | Lowest level zonal wind                                   | m/s    | 
| VBOT    | Lowest model level meridional wind                        | m/s    | 
| QREFHT  | Reference height humidity                                 | kg/kg  | 
| PS      | Surface pressure                                          | Pa     | 
| PSL     | Sea level pressure                                        | Pa     |  
| T200    | Temperature at 200 mbar pressure surface                  | K      | 
| T500    | Temperature at 500 mbar pressure surface                  | K      | 
| PRECT   | Total (convective and large-scale) precipitation rate     | m/s    |  
| TS      | Surface temperature (radiative)                           | K      | 
| TREFHT  | Reference height temperature                              | K      | 
| Z1000   | Geopotential Z at 1000 mbar pressure surface              | m      | 
| Z200    | Geopotential Z at 200 mbar pressure surface               | m      | 
| ZBOT    | Lowest modal level height                                 | m      | 
| LABELS  | 0: Background, 1: Tropical Cyclone, 2: Atmospheric river  | -      |  


The data set is split in a \textbf{training set} of 398 (map, labels) pairs spanning years 1996 to 2010 in the CAM5.1 climate simulation, and a \textbf{test set} of 61 (map, labels) pairs spanning 2011 to 2013.

You can find the data at [https://portal.nersc.gov/project/ClimateNet/](https://portal.nersc.gov/project/ClimateNet/) and we provide a notebook to download the data automatically.





## ClimateNet library

ClimateNet is a Python library for deep learning-based Climate Science. It provides tools for quick detection and tracking of extreme weather events, and is used as the implementation of our baseline model.

References: 

Lukas Kapp-Schwoerer, Andre Graubner, Sol Kim, and Karthik Kashinath. Spatio-temporal segmentation and tracking of weather patterns with light-weight neural networks. AI for Earth Sciences Workshop at NeurIPS 2020. [https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_55.pdf](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_55.pdf).

ClimateNet library repository: [https://github.com/andregraubner/ClimateNet](https://github.com/andregraubner/ClimateNet).


## Baseline

We use the library and the published implementation of the CGNet network as our baseline, and assess baseline performance on the latest published weights trained over the ClimateNet training set for 15 epochs with the Jaccard loss (weights available at [https://portal.nersc.gov/project/ClimateNet/climatenet_new/model/](https://portal.nersc.gov/project/ClimateNet/climatenet_new/model/).)


## References

Dataset: https://gmd.copernicus.org/articles/14/107/2021/

Methods: https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_55.pdf
