import os
import glob
import random
import rasterio
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

# indices of sentinel-2 bands related to land
S2_BANDS_LD = [2, 3, 4, 5, 6, 7, 8, 9, 12, 13]
S2_BANDS_RGB = [2, 3, 4] # B(2),G(3),R(4)


# util function for reading s2 data
def load_s2(path, imgTransform, s2_band): 
    bands_selected = s2_band
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    if not imgTransform:
        s2 = np.clip(s2, 0, 10000)
        s2 /= 10000
    s2 = s2.astype(np.float32)
    return s2


# util function for reading s1 data
def load_s1(path, imgTransform):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    if not imgTransform:
        s1 /= 25
        s1 += 1
    s1 = s1.astype(np.float32)
    return s1
    

# util function for reading data from single sample
def load_sample(sample, labels, label_type, threshold, imgTransform, use_s1, use_s2, use_RGB, IGBP_s):

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_LD)
    # load only RGB   
    if use_RGB and use_s2==False:
        img = load_s2(sample["s2"], imgTransform, s2_band=S2_BANDS_RGB)
        
    # load s1 data
    if use_s1:
        if use_s2 or use_RGB:
            img = np.concatenate((img, load_s1(sample["s1"], imgTransform)), axis=0)
        else:
            img = load_s1(sample["s1"], imgTransform)
            
    # load label
    lc = labels[sample["id"]]
    
    # covert label to IGBP simplified scheme
    if IGBP_s:
        cls1 = sum(lc[0:5]);
        cls2 = sum(lc[5:7]); 
        cls3 = sum(lc[7:9]);
        cls6 = lc[11] + lc[13];
        lc = np.asarray([cls1, cls2, cls3, lc[9], lc[10], cls6, lc[12], lc[14], lc[15], lc[16]])
        
    if label_type == "multi_label":
        lc_hot = (lc >= threshold).astype(np.float32)     
    else:
        loc = np.argmax(lc, axis=-1)
        lc_hot = np.zeros_like(lc).astype(np.float32)
        lc_hot[loc] = 1
             
    rt_sample = {'image': img, 'label': lc_hot, 'id': sample["id"]}
    
    if imgTransform is not None:
        rt_sample = imgTransform(rt_sample)
    
    return rt_sample


#  calculate number of input channels  
def get_ninputs(use_s1, use_s2, use_RGB):
    n_inputs = 0
    if use_s2:
        n_inputs += len(S2_BANDS_LD)
    if use_s1:
        n_inputs += 2
    if use_RGB and use_s2==False:
        n_inputs += 3
        
    return n_inputs


# class SEN12MS..............................
class SEN12MS(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""
    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val/test split and can be obtained from:
    # https://github.com/MSchmitt1984/SEN12MS/

    def __init__(self, path, ls_dir=None, imgTransform=None, 
                 label_type="multi_label", threshold=0.1, subset="train",
                 use_s2=False, use_s1=False, use_RGB=False, IGBP_s=True):
        """Initialize the dataset"""

        # inizialize
        super(SEN12MS, self).__init__()
        self.imgTransform = imgTransform
        self.threshold = threshold
        self.label_type = label_type

        # make sure input parameters are okay
        if not (use_s2 or use_s1 or use_RGB):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2, s1, RGB] to True!")
        self.use_s2 = use_s2
        self.use_s1 = use_s1
        self.use_RGB = use_RGB
        self.IGBP_s = IGBP_s
        
        assert subset in ["train", "val", "test"]
        assert label_type in ["multi_label", "single_label"] # new !!
        
        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2, use_RGB)

        # provide number of IGBP classes 
        if IGBP_s == True:
            self.n_classes = 10
        else:
            self.n_classes = 17 

        # make sure parent dir exists
        assert os.path.exists(path)
        assert os.path.exists(ls_dir)


# While runing below section <create sample list> on unix server, the codes were
# correct, however, the machine read the holdout-list wrongly. Therefore I recommend 
# to use the next section <import split list> to directly read the split lists.
# It would be more robust. Although I set the random seed in selecting validation scenes
# in <create sample list>, the seed may result in different selection on different
# machine, but we additionally provide the validation_scenes in online foler
#        ------------------------ create sample list -------------------------
#        if label_type == "multi_label" or label_type == "single_label":
#            # find and index samples
#            self.samples = []
#            if subset == "train":
#                pbar = tqdm(total=162555)   # 165902 samples in train set
#            if subset == "val":
#                pbar = tqdm(total=18550)   # 18055 samples in val set
#            if subset == "test":
#                pbar = tqdm(total=18106)   # 14760 samples in test set
#            pbar.set_description("[Load]")
#            try:
#                holdout_list = list(pd.read_csv(os.path.join(
#                        path,"SEN12MS_holdOutScenes.txt"),header=None)[0])
#            except:
#                holdout_list = list(pd.read_csv(os.path.join(
#                        ls_dir,"SEN12MS_holdOutScenes.txt"),header=None)[0])
#                
#            holdout_list = [x.replace("s1_", "s2_") for x in holdout_list]
#    
#            # compile a list of paths to all samples
#            if subset=="train" or subset=="val":
#                train_list = []
#                for seasonfolder in ['ROIs1970_fall', 'ROIs1158_spring',
#                                     'ROIs2017_winter', 'ROIs1868_summer']:
#                    train_list += [os.path.join(seasonfolder, x) for x in
#                                   os.listdir(os.path.join(path, seasonfolder))]    
#                train_list = [x for x in train_list if "s2_" in x]
#                train_list = [x for x in train_list if x not in holdout_list] # added replace 3 lines before
#                
#                if subset == "train":
#                    sample_dirs = train_list
#                
#                elif subset == "val":
#                    random.seed(2071)
#                    val_list = random.sample(train_list, 25) # pick 25 scenes randomly
#                    sample_dirs = val_list
#
#            if subset == "test":
#                sample_dirs = holdout_list
#    
#            for folder in sample_dirs:
#                s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
#                                         recursive=True)
#    
#                # INFO there is one "broken" file in the sen12ms dataset with nan
#                #      values in the s1 data. we simply ignore this specific sample
#                #      at this point. id: ROIs1868_summer_xx_146_p202
#                if folder == "ROIs1868_summer/s2_146":
#                    broken_file = os.path.join(path, "ROIs1868_summer",
#                                               "s2_146",
#                                               "ROIs1868_summer_s2_146_p202.tif")
#                    s2_locations.remove(broken_file)
#                    pbar.write("ignored one sample because of nan values in "
#                               + "the s1 data")
#    
#                for s2_loc in s2_locations:
#                    s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
#    
#                    pbar.update()
#                    self.samples.append({"s1": s1_loc, "s2": s2_loc,
#                                         "id": os.path.basename(s2_loc)})
#    
#            pbar.close()
#      
#-------------------------------- import split lists--------------------------------
        if label_type == "multi_label" or label_type == "single_label":
            # find and index samples
            self.samples = []
            if subset == "train":
                pbar = tqdm(total=162555)   # 162556-1 samples in train set 
                # 1 broken file "ROIs1868_summer_s2_146_p202" had been removed 
                # from the list already
            if subset == "val":
                pbar = tqdm(total=18550)   # 18550 samples in val set
            if subset == "test":
                pbar = tqdm(total=18106)   # 18106 samples in test set
            pbar.set_description("[Load]")
            
            if subset == "train":
                file =os.path.join(ls_dir, 'train_list.pkl')
                sample_list = pkl.load(open(file, "rb"))
                
            elif subset == "val":
                file =os.path.join(ls_dir, 'val_list.pkl')
                sample_list = pkl.load(open(file, "rb"))
                
            else:
                file =os.path.join(ls_dir, 'test_list.pkl')
                sample_list = pkl.load(open(file, "rb"))
                
            # remove broken file
            broken_file = 'ROIs1868_summer_s2_146_p202.tif'
            if broken_file in sample_list:
                sample_list.remove(broken_file)
            
            #
            pbar.set_description("[Load]")
            
            for s2_id in sample_list:
                mini_name = s2_id.split("_")
                s2_loc = os.path.join(path, (mini_name[0]+'_'+mini_name[1]),
                                      (mini_name[2]+'_'+mini_name[3]), s2_id)
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                
                pbar.update()
                self.samples.append({"s1": s1_loc, "s2": s2_loc, 
                                     "id": s2_id})
       
            pbar.close()
#----------------------------------------------------------------------               
        
        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the sen12ms subset", subset)
        
        # import lables as a dictionary
        label_file = os.path.join(ls_dir,'IGBP_probability_labels.pkl')

        a_file = open(label_file, "rb")
        self.labels = pkl.load(a_file)
        a_file.close()
        

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        labels = self.labels
        return load_sample(sample, labels, self.label_type, self.threshold, self.imgTransform, 
                           self.use_s1, self.use_s2, self.use_RGB, self.IGBP_s)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)




#%% data normalization

class Normalize(object):
    def __init__(self, bands_mean, bands_std):
        
        self.bands_s1_mean = bands_mean['s1_mean']
        self.bands_s1_std = bands_std['s1_std']

        self.bands_s2_mean = bands_mean['s2_mean']
        self.bands_s2_std = bands_std['s2_std']
        
        self.bands_RGB_mean = bands_mean['s2_mean'][0:3]
        self.bands_RGB_std = bands_std['s2_std'][0:3]
        
        self.bands_all_mean = self.bands_s2_mean + self.bands_s1_mean
        self.bands_all_std = self.bands_s2_std + self.bands_s1_std

    def __call__(self, rt_sample):

        img, label, sample_id = rt_sample['image'], rt_sample['label'], rt_sample['id']

        # different input channels
        if img.size()[0] == 12:
            for t, m, s in zip(img, self.bands_all_mean, self.bands_all_std):
                t.sub_(m).div_(s) 
        elif img.size()[0] == 10:
            for t, m, s in zip(img, self.bands_s2_mean, self.bands_s2_std):
                t.sub_(m).div_(s)          
        elif img.size()[0] == 5:
            for t, m, s in zip(img, 
                               self.bands_RGB_mean + self.bands_s1_mean,
                               self.bands_RGB_std + self.bands_s1_std):
                t.sub_(m).div_(s)                                
        elif img.size()[0] == 3:
            for t, m, s in zip(img, self.bands_RGB_mean, self.bands_RGB_std):
                t.sub_(m).div_(s)
        else:
            for t, m, s in zip(img, self.bands_s1_mean, self.bands_s1_std):
                t.sub_(m).div_(s)            
        
        return {'image':img, 'label':label, 'id':sample_id}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, rt_sample):
        
        img, label, sample_id = rt_sample['image'], rt_sample['label'], rt_sample['id']
        
        rt_sample = {'image': torch.tensor(img), 'label':label, 'id':sample_id}
        return rt_sample





#%%...........................................................................
# DEBUG usage examples
if __name__ == "__main__":
    
    # statistics for "multi_label"
    bands_mean = {'s1_mean': [-11.76858, -18.294598],
                  's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                              2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}
    
    bands_std = {'s1_std': [4.525339, 4.3586307],
                 's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                            1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]} 
    
    # data path
    data_dir = "/work/share/sen12ms"    # SEN12MS dir
    list_dir = "/home/labels_splits/"    # split lists/ label dirs
  
    # define image transform
    imgTransform=transforms.Compose([ToTensor(),Normalize(bands_mean, bands_std)])
    
    # test multi_label part with normalization
    print("\n\nSEN12MS train")
    ds = SEN12MS(data_dir, list_dir, imgTransform, 
                 label_type="multi_label", threshold=0.1, subset="train", 
                 use_s1=False, use_s2=False, use_RGB=True, IGBP_s=True)
    s_nor = ds.__getitem__(100)
    print("id:", s_nor["id"], "\n",
          "input shape:", s_nor["image"].shape, "\n",
          "number of classes", ds.n_classes)
    







