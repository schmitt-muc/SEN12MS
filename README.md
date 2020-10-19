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

## models   
In this folder, you can find code for ResNet and DenseNet models aiming at single-label and multi-label scene classification. The files are described as follows:
- 

## Additional resources for SEN12MS-based DL models
Pre-trained weights and optimization parameters for these models can be downloaded from here: 
https://syncandshare.lrz.de/public?folderID=Mlh2b1p5WVRTendxSkFqN3NwNHJI. 
- In addition, the following repository created by Lukas Liebel contains DeepLabv3 and Unet models adapted to the peculiarities of SEN12MS, so that they can be directly trained and evaluated on SEN12MS (and DFC2020 data, see below) without much further ado:
https://github.com/lukasliebel/dfc2020_baseline.

## DFC2020
SEN12MS is used as backbone dataset of the 2020 IEEE-GRSS Data Fusion Contest (DFC2020). In the frame of the contest, high-resolution (GSD: 10m) validation and test data is released. The data and more information can be retrieved via the following links:
- http://www.grss-ieee.org/community/technical-committees/data-fusion/ Homepage of the Image Analysis and Data Fusion Committee (IADFC) of the IEEE-GRSS with detailed information about the contest.
- https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest IEEEDataPort page providing the high-resolution validation and test data for download
- https://competitions.codalab.org/competitions/22289 CodaLab page hosting the actual competition, including forum and leaderboard

## Papers working with SEN12MS Data
- Abady L, Barni M, Garzelli A, Tondi BL (2020) GAN generation of synthetic multispectral satellite images. In: Proc. SPIE 11533, Image and Signal Processing for Remote Sensing XXVI: 115330L. doi:10.1117/12.2575765
- Hu L, Robinson C, Dilkina B (2020) Model generalization in deep learning applications for land cover mapping. Preprint available at https://arxiv.org/abs/2008.10351
- Yu Q, Liu W, Li J (2020) Spatial resolution enhancement of land cover mapping using deep convolutional nets. Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci. XLIII-B1-2020: 85–89. https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B1-2020/85/2020/
- Yuan Y, Tian J, Reinartz P (2020) Generating artificial near infrared spectral band from RGB image using conditional generative adversarial network. ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci. V-3-2020: 279–285. https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2020/279/2020/
- Rußwurm M, Wang S, Körner M, Lobell D (2020) Meta-learning for few-shot land cover classification. In: Proc. CVPRW. https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Russwurm_Meta-Learning_for_Few-Shot_Land_Cover_Classification_CVPRW_2020_paper.html
- Schmitt M, Prexl J, Ebel P, Liebel L, Zhu XX (2020) Weakly supervised semantic segmentation of satellite images for land cover mapping - challenges and opportunities. ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci. V-3-2020: 795-802. https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2020/795/2020/
- Yokoya N, Ghamisi P, Hänsch R, Schmitt M (2020) 2020 IEEE GRSS Data Fusion Contest: Global land cover mapping with weak supervision. IEEE Geosci. Remote Sens. Mag. 8 (1): 154-157. https://ieeexplore.ieee.org/document/9028003
