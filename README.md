# SEN12MS Toolbox
This repository is supposed to collect tools and utilities for working with the SEN12MS dataset. 

The dataset itself can be downloaded here: https://mediatum.ub.tum.de/1474000

Information about the dataset can be found in the related publication:
> Schmitt M, Hughes LH, Qiu C, Zhu XX (2019) SEN12MS - a curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion. In: ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences IV-2/W7: 153-160

Link: https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/IV-2-W7/153/2019/

```
@inproceedings{Schmitt2019,
    author = {Michael Schmitt and Lloyd Haydn Hughes and Chunping Qiu and Xiao Xiang Zhu},
    title = {SEN12MS -- a curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion},
    booktitle={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences}, 
    volume={IV-2/W7},
    year = {2019},
    pages = {153-160},
    doi={10.5194/isprs-annals-IV-2-W7-153-2019}
}
```

## Contents
The repository contains the following folders:
### splits
In this folder, text files containing suggestions for splits are stored, pointing either at complete folders or individual files. Due to the folder structure and naming convention of SEN12MS, such file/folder list files should only point to Sentinel-1 data (i.e. with the identifier `_s1_` in folder and/or file name. After reading in such a file, the identifier can easily be replaced to `_s2_` or `_lc_` to address the corresponding Sentinel-2 or land cover data.  
Current split suggestions:
- `SEN12MS_holdOutScenes.txt`: this file contains scenes to form a hold-out dataset. The scenes were selected with great care to ensure that both the spatial and seasonal distributions are equal to the ones of the complete dataset. These hold-out scenes contain about 10% of all patches of the dataset.

### utils
In this folder, other utilities that can help to load, process, or analyze the data can be stored.
- `Sen12MSOverview.ipynb`: this notebook analyzes the class distribution of the whole SEN12MS dataset and plots the individual ROIs onto a world map

### dataLoaders
TODO: Data loaders for efficient loading of the data into common deep learning frameworks.

## Other Resources
SEN12MS is used as backbone dataset of the 2020 IEEE-GRSS Data Fusion Contest (DFC2020). In the frame of the contest, high-resolution (GSD: 10m) validation and test data is released. The data and more information can be retrieved via the following links:
- http://www.grss-ieee.org/community/technical-committees/data-fusion/ Homepage of the Image Analysis and Data Fusion Committee (IADFC) of the IEEE-GRSS with detailed information about the contest.
- https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest IEEEDataPort page providing the high-resolution validation and test data for download
- https://competitions.codalab.org/competitions/22289 CodaLab page hosting the actual competition, including forum and leaderboard

