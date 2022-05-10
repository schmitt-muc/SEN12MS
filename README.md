# SEN12MS Toolbox
This repository is supposed to collect tools and utilities for working with the SEN12MS dataset. 

## Dataset Link
The dataset itself can be downloaded here: https://mediatum.ub.tum.de/1474000

## Publications
Information about the dataset can be found in two related publications:

The original description of the dataset and its generation is found in
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

There is also a second paper describing a small re-design of the SEN12MS dataset for single-label and multi-label scene classification tasks, including benchmark results on state-of-the-art CNN architectures:
> Schmitt M, Wu Y-L (2021) Remote sensing image classification with the SEN12MS dataset. 
ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences V-2-2021: 101–106

Link: https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2021/101/2021/
```
@inproceedings{Schmitt2021,
      title={Remote sensing image classification with the SEN12MS dataset}, 
      author={Michael Schmitt and Yu-Lun Wu},
      year={2021},
      booktitle={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
      volume={V-2-2021},
      pages={101-106},
      doi={10.5194/isprs-annals-V-2-2021-101-2021}
}
```
If you use the dataset in the context of scientific publications, please cite the appropriate reference in your paper!

## Contents
The repository contains the following folders:

### labels  
In this folder, text and pickle files containing single labels for every scene (patch) of SEN12MS are provided, following the IGBP land cover scheme. They can be used to train scene classification instead of semantic segmentation models. The class numbers of both the original IGBP scheme and the simplified IGBP scheme can be found in (Schmitt & Wu, 2021). 
- `single-label_IGBPfull_ClsNum`: This file contains scene labels based on the full IGBP land cover scheme, represented by actual class numbers.
- `single-label_IGBP_full_OneHot`: This file contains scene labels based on the full IGBP land cover scheme, represented by a one-hot vector encoding.
- `single-label_IGBPsimple_ClsNum`: This file contains scene labels based on the simplified IGBP land cover scheme, represented by actual class numbers.
- `single-label_IGBPsimple_OneHot`: This file contains scene labels based on the simplified IGBP land cover scheme, represented by a one-hot vector encoding.
All these files are available both in plain ASCII (.txt) format, as well as .pkl format. 

In addition, there is a list of multi-class labels for every scene:
- `IGBP_probability_labels.pkl`: This file contains scene labels based on the full IGBP land cover scheme, represented by the probability vectors. The probability vector shows the percentages of coverage of different classes in a scene/patch.

Please note: The python scripts provided in the `classification` folder of this repository only read probability labels in the original IGBP scheme (i.e. `IGBP_probability_labels.pkl`), and convert them into single-label/ multi-label in the simplified IGBP land cover scheme on the fly. The other files are intended for sake of convenience to be used with other frameworks.

### splits
In this folder, text files containing suggestions for splits are stored, pointing either at complete folders or individual files. Due to the folder structure and naming convention of SEN12MS, such file/folder list files should only point to Sentinel-1  or Sentinel-2 data (i.e. with the identifier `_s1_` or `_s2_`in folder and/or file name. After reading in such a file, the identifier can easily be replaced to `_s2_`, `_s1_` or `_lc_`, respectively, to address the corresponding Sentinel-2, Sentinel-1 or land cover data.  
Current split suggestions:
- `SEN12MS_holdOutScenes.txt`: this file contains scenes to form a hold-out dataset. The scenes were selected with great care to ensure that both the spatial and seasonal distributions are equal to the ones of the complete dataset. These hold-out scenes contain about 10% of all patches of the dataset.
- `train_list`: list of image patches used for training the scene classification models provided in this repository. All patches of this list are NOT contained in the above-mentioned set of hold-out scenes.  
Please note: One triplet was removed from this list, because the corresponding SAR patch was damaged. The removed file ID is `ROIs1868_summer_s1_146_p202`. The label still exists in the label files, as the corresponding Sentinel-2 data is not broken.
- `test_list`: list of image patches used for testing. These patches are all from the scenes defined in the hold-out set above.
- `val_list`: Not provided. Based on your individual considerations, you should extract a subset from the files in `train_list` to be used as validation set. This can be done, e.g., by random sampling, or by selecting specific patches from specific scenes using the SEN12MS file tree structure.

The file lists are available both in plain ASCII (.txt) format, as well as .pkl format, with the .pkl files being the ones that area read by the scripts provided in the `classification` folder of this repository.

### utils
In this folder, other utilities that can help to load, process, or analyze the data can be stored.
- `Sen12MSOverview.ipynb`: this notebook analyzes the class distribution of the whole SEN12MS dataset and plots the individual ROIs onto a world map

### classification  
In this folder, you can find codes for image classification CNNs (e.g. ResNet and DenseNet models) aiming at single-label and multi-label scene classification. They were developed using Python 3.7.7 and using several packages (NumPy, Rasterio, Scikit-Learn, TensorboardX, Torch, TorchVision, TQDM). To install the packages run `pip install requirements.txt` with your development environment activated from the `classification` folder.

The files needed for training and evaluating SEN12MS-based classification models are described as follows:
- `dataset.py`: This python script reads the data from SEN12MS and the probability label file. It converts the probability labels into single-label or multi-label annotations.
- `main_train.py`: This python script is used to train the model. It requires several input arguments to specify the scenario for training (e.g. label type, simplified/original IGBP scheme, models, learning rate etc.). Here is an example of the input arguments:  
`CUDA_VISIBLE_DEVICES=0 \  
python main_train.py \  
  --exp_name experiment_name \  
  --data_dir /work/share/sen12ms \  
  --label_split_dir /home/labels_splits \  
--use_RGB \  
  --IGBP_simple \  
  --label_type multi_label \  
  --threshold 0.1 \  
  --model DenseNet121 \  
  --lr 0.001\  
  --decay 1e-5 \  
  --batch_size 64 \  
  --num_workers 4 \  
  --epochs 100 \`  
These arguments will be saved into a .txt file automatically. This .txt file can be used in the testing for reading the arguments. The `threshold` parameter is used to filter out the labels with lower probabilities. Note that this threshold has no influence on single-label classification. More explanation of the arguments is in the `main_train.py` file. Note that the probability label file and the split lists should be put under the same folder during training and testing. The script reads .pkl format instead of .txt files.
- `test.py`: This python script is used to test the model. It is a semi-automatic script and reads the argument file generated in the training process to decide the label type, model type etc. However, it still requires user to input some basic arguments, such as the path of data directory. Here is an example of the input arguments:  
`CUDA_VISIBLE_DEVICES=0 \  
python test.py \  
  --config_file /home/single_DenseNet_RGB/logs/20201019_000519_arguments.txt \  
  --data_dir /work/share/sen12ms \  
  --label_split_dir /home/labels_splits \  
  --checkpoint_pth /home/major_DenseNet_RGB/checkpoints/20201019_000519_model_best.pth \  
  --batch_size 64 \  
  --num_workers 4 \`  
All other arguments will be read from the argument .txt file created when calling the training function.
- `metrics.py`: This script contains several metrics used to evaluate single-label/multi-label classification test results.
- `models/DenseNet.py`: This script contains several DenseNet models with different depth.
- `models/ResNet.py`: This script contains several ResNet models with different depth.
- `VGG.py`: This script contains VGG16 and VGG19 models. However, it is not used in the experiments.

Pre-trained weights and optimization parameters for these models can be downloaded from here: 
https://syncandshare.lrz.de/getlink/fiCDbqiiSFSNwot5exvUcW1y/trained_models. 

The models' respective input modalities are specified by their suffixes:
- `_RGB` means that only Sentinel-2 RGB imagery is used 
- `_s2` indicates that full multi-spectral Sentinel-2 data were used
- `_s1s2` represents data fusion-based models analyzing both Sentinel-1 and Sentinel-2 data 

## Additional Resources

### Semantic Segmentation 
The following repository created by Lukas Liebel contains DeepLabv3 and Unet models adapted to the peculiarities of SEN12MS, so that they can be directly trained and evaluated on SEN12MS (and DFC2020 data, see below) without much further ado:
https://github.com/lukasliebel/dfc2020_baseline.

### DFC2020
SEN12MS is used as backbone dataset of the 2020 IEEE-GRSS Data Fusion Contest (DFC2020). In the frame of the contest, high-resolution (GSD: 10m) validation and test data is released. The data and more information can be retrieved via the following links:
- https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/#past-data-fusion-contests Homepage of the Image Analysis and Data Fusion Committee (IADFC) of the IEEE-GRSS with detailed information about the contest.
- https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest IEEEDataPort page providing the high-resolution validation and test data for download
- https://competitions.codalab.org/competitions/22289 CodaLab page hosting the actual competition, including forum and leaderboard

### Papers working with SEN12MS Data
- Abady L, Barni M, Garzelli A, Tondi BL (2020) GAN generation of synthetic multispectral satellite images. In: Proc. SPIE 11533, Image and Signal Processing for Remote Sensing XXVI: 115330L. https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11533/2575765/GAN-generation-of-synthetic-multispectral-satellite-images/10.1117/12.2575765.full?SSO=1
- Ciocarlan A, Stoian A (2021) Ship detection in Sentinel 2 multi-spectral images with self-supervised learning. Remote Sensing 13 (21): 4255. https://doi.org/10.3390/rs13214255 
- Doi K, Sakurada K, Onishi M, Iwasaki A (2020) GAN-based SAR-to-optical image translation with region information. In: Proc. IGARSS: 2069-2072. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9323085
- Horvath J, Mas Montserrat D, Delp EJ (2021) Nested attention U-Net: a splicing detection method for satellite images. In: Proc. ICPR 2020 Workshops, LNCS 12666: 516–529. https://link.springer.com/chapter/10.1007%2F978-3-030-68780-9_41 
- Hu L, Robinson C, Dilkina B (2020) Model generalization in deep learning applications for land cover mapping. Preprint available at https://arxiv.org/abs/2008.10351
- Jain P, Schoen-Phelan B, Ross R (2021) Multi-modal self-supervised representation learning for Earth
observation. In: Proc. IGARSS: 3241-3244. https://ieeexplore.ieee.org/document/9553741
- Kakogeorgiou I, Karantzalos K (2021) Evaluating explainable artificial intelligence methods for multi-label deep learning classification tasks in remote sensing. Preprint available at https://www.researchgate.net/publication/350647715_Evaluating_Explainable_Artificial_Intelligence_Methods_for_Multi-label_Deep_Learning_Classification_Tasks_in_Remote_Sensing/citations
- Li Y, Li B (2021) Super-resolution of Sentinel-2 images at 10m resolution without reference images. Preprint available at https://doi.org/10.20944/preprints202104.0556.v1
- Rußwurm M, Wang S, Körner M, Lobell D (2020) Meta-learning for few-shot land cover classification. In: Proc. CVPRW. https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Russwurm_Meta-Learning_for_Few-Shot_Land_Cover_Classification_CVPRW_2020_paper.html
- Schmitt M, Prexl J, Ebel P, Liebel L, Zhu XX (2020) Weakly supervised semantic segmentation of satellite images for land cover mapping - challenges and opportunities. ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci. V-3-2020: 795-802. https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2020/795/2020/
- Xu H, He M, Rao Z, Li W (2021) HIM-Net: a new neural network approach for SAR and optical image template matching. In: Proc. ICIP: 3827-3831. https://ieeexplore.ieee.org/document/9506616  
- Yokoya N, Ghamisi P, Hänsch R, Schmitt M (2020) 2020 IEEE GRSS Data Fusion Contest: Global land cover mapping with weak supervision. IEEE Geosci. Remote Sens. Mag. 8 (1): 154-157. https://ieeexplore.ieee.org/document/9028003
- Yu Q, Liu W, Li J (2020) Spatial resolution enhancement of land cover mapping using deep convolutional nets. Int. Arch. Photogramm. Remote Sens. Spatial Inf. Sci. XLIII-B1-2020: 85–89. https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLIII-B1-2020/85/2020/
- Yu, Q, Liu W, Gonçalves WN, Junior JM, Li J (2021) Spatial resolution enhancement for large-scale land cover mapping via weakly supervised deep learning. Photogramm. Eng. Remote Sensing 87 (6): 405-412. https://www.ingentaconnect.com/contentone/asprs/pers/2021/00000087/00000006/art00009
- Yuan Y, Tian J, Reinartz P (2020) Generating artificial near infrared spectral band from RGB image using conditional generative adversarial network. ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci. V-3-2020: 279–285. https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-3-2020/279/2020/
- Zhang D, Gade M, Zhang J (2021) SOFNet: SAR-optical fusion network for land cover classification. In: Proc. IGARSS: 2409-2412. https://ieeexplore.ieee.org/document/9554070 
