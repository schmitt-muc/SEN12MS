import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import sys
sys.path.append('../')

from dataset import SEN12MS, ToTensor, Normalize
from models.VGG import VGG16, VGG19
from models.ResNet import ResNet50, ResNet101, ResNet152
from models.DenseNet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from metrics import MetricTracker, Precision_score, Recall_score, F1_score, \
    F2_score, Hamming_loss, Subset_accuracy, Accuracy_score, One_error, \
    Coverage_error, Ranking_loss, LabelAvgPrec_score, calssification_report, \
    conf_mat_nor, get_AA, multi_conf_mat, OA_multi



model_choices = ['VGG16', 'VGG19',
                 'ResNet50','ResNet101','ResNet152',
                 'DenseNet121','DenseNet161','DenseNet169','DenseNet201']
label_choices = ['multi_label', 'single_label']

# ------------------------ define and parse arguments -------------------------
parser = argparse.ArgumentParser()

# configure
parser.add_argument('--config_file', type=str, default=None,
                    help='path to config file')

# data directory
parser.add_argument('--data_dir', type=str, default=None,
                    help='path to SEN12MS dataset')
parser.add_argument('--label_split_dir', type=str, default=None,
                    help="path to label data and split list")
parser.add_argument('--checkpoint_pth', type=str, default=None,
                    help='path to the pretrained weights file')

# hyperparameters
parser.add_argument('--batch_size', type=int, default=64,
                    help='mini-batch size (default: 64)')
parser.add_argument('--num_workers',type=int, default=4,
                    help='num_workers for data loading in pytorch')

args = parser.parse_args()


# -------------------------------- Main Program ------------------------------
def main():
    global args

# -------------------------- load config from file
    # load config
    config_file = args.config_file
        
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            (key, val) = line.split()
            config[(key[0:-1])] = val
            
    # Convert string to boolean
    boo_use_s1 = config['use_s1'] == 'True'
    boo_use_s2 = config['use_s2'] == 'True'
    boo_use_RGB = config['use_RGB'] == 'True'
    boo_IGBP_simple = config['IGBP_simple'] == 'True'
    
    # define label_type
    cf_label_type = config['label_type']
    assert cf_label_type in label_choices
    
    # define threshold 
    cf_threshold = float(config['threshold'])
    
    
    # define labels used in cls_report
    if boo_IGBP_simple:
        ORG_LABELS = ['1','2','3','4','5','6','7','8','9','10']
    else:
        ORG_LABELS = ['1','2','3','4','5','6','7','8','9','10',
                      '11','12','13','14','15','16','17']
    
    
# ----------------------------------- data
    # define mean/std of the training set (for data normalization)    
    bands_mean = {'s1_mean': [-11.76858, -18.294598],
                  's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                              2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}
                  
    bands_std = {'s1_std': [4.525339, 4.3586307],
                 's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                            1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]} 
                
                
    # load test dataset
    imgTransform = transforms.Compose([ToTensor(),Normalize(bands_mean, bands_std)])
    
    test_dataGen = SEN12MS(args.data_dir, args.label_split_dir,
                           imgTransform = imgTransform,
                           label_type=cf_label_type, threshold=cf_threshold, subset="test", 
                           use_s1=boo_use_s1, use_s2=boo_use_s2, use_RGB=boo_use_RGB,
                           IGBP_s=boo_IGBP_simple)
    
    # number of input channels
    n_inputs = test_dataGen.n_inputs
    print('input channels =', n_inputs)
    
    # set up dataloaders
    test_data_loader = DataLoader(test_dataGen,
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers, 
                                  shuffle=False, 
                                  pin_memory=True)
    
# -------------------------------- ML setup    
    # cuda
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    # define number of classes
    if boo_IGBP_simple:
        numCls = 10
    else:
        numCls = 17
        
    print('num_class: ', numCls)
    
    # define model
    if config['model'] == 'VGG16':
        model = VGG16(n_inputs, numCls)
    elif config['model'] == 'VGG19':
        model = VGG19(n_inputs, numCls)
        
    elif config['model'] == 'ResNet50':
        model = ResNet50(n_inputs, numCls)
    elif config['model'] == 'ResNet101':
        model = ResNet101(n_inputs, numCls)
    elif config['model'] == 'ResNet152':
        model = ResNet152(n_inputs, numCls)
        
    elif config['model'] == 'DenseNet121':
        model = DenseNet121(n_inputs, numCls)
    elif config['model'] == 'DenseNet161':
        model = DenseNet161(n_inputs, numCls)
    elif config['model'] == 'DenseNet169':
        model = DenseNet169(n_inputs, numCls)
    elif config['model'] == 'DenseNet201':
        model = DenseNet201(n_inputs, numCls)                            
    else:
        raise NameError("no model")
    
    
    # move model to GPU if is available
    if use_cuda:
        model = model.cuda()
        
    # import model weights
    checkpoint = torch.load(args.checkpoint_pth)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint_pth, checkpoint['epoch']))

    # set model to evaluation mode
    model.eval()

    # define metrics
    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    subset_acc_ = Subset_accuracy()
    acc_score_ = Accuracy_score() # from original script, not recommeded, seems not correct
    one_err_ = One_error()
    coverage_err_ = Coverage_error()
    rank_loss_ = Ranking_loss()
    labelAvgPrec_score_ = LabelAvgPrec_score()
    
    calssification_report_ = calssification_report(ORG_LABELS) 
    
# -------------------------------- prediction
    y_true = []
    predicted_probs = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_data_loader, desc="test")):
          
            # unpack sample
            bands = data["image"]
            labels = data["label"]
    
            # move data to gpu if model is on gpu
            if use_cuda:
                bands = bands.to(torch.device("cuda"))
                #labels = labels.to(torch.device("cuda"))
            
            # forward pass 
            logits = model(bands)

            # convert logits to probabilies
            if cf_label_type == 'multi_label':
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                sm = torch.nn.Softmax(dim=1)
                probs = sm(logits).cpu().numpy()
                  
            labels = labels.cpu().numpy() # keep true & pred label at same loc.
            predicted_probs += list(probs)
            y_true += list(labels) 
                  
            
    predicted_probs = np.asarray(predicted_probs)
    # convert predicted probabilities into one/multi-hot labels 
    if cf_label_type == 'multi_label':
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    else:
        loc = np.argmax(predicted_probs, axis=-1)
        y_predicted = np.zeros_like(predicted_probs).astype(np.float32)
        for i in range(len(loc)):
            y_predicted[i,loc[i]] = 1
            
    y_true = np.asarray(y_true)  
    
# --------------------------- evaluation with metrics  
    # general
    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
    macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
    hamming_loss = hamming_loss_(y_predicted, y_true)
    subset_acc = subset_acc_(y_predicted, y_true)
    macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)
    # ranking-based
    one_error = one_err_(predicted_probs, y_true)
    coverage_error = coverage_err_(predicted_probs, y_true)
    rank_loss = rank_loss_(predicted_probs, y_true)
    labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)

    cls_report = calssification_report_(y_predicted, y_true)
    
    
    if cf_label_type == 'multi_label':
        [conf_mat, cls_acc, aa] = multi_conf_mat(y_predicted, y_true, n_classes=numCls)
        # the results derived from multilabel confusion matrix are not recommended to use
        oa = OA_multi(y_predicted, y_true)
        # this oa can be Jaccard index 
        
        info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "samplePrec" : sample_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "sampleRec" : sample_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "sampleF1" : sample_f1,
            "macroF2" : macro_f2,
            "microF2" : micro_f2,
            "sampleF2" : sample_f2,
            "HammingLoss" : hamming_loss,
            "subsetAcc" : subset_acc,
            "macroAcc" : macro_acc,
            "microAcc" : micro_acc,
            "sampleAcc" : sample_acc,
            "oneError" : one_error,
            "coverageError" : coverage_error,
            "rankLoss" : rank_loss,
            "labelAvgPrec" : labelAvgPrec,
            "clsReport": cls_report,
            "multilabel_conf_mat": conf_mat,
            "class-wise Acc": cls_acc,
            "AverageAcc": aa,
            "OverallAcc": oa}    
                
    else:
        conf_mat = conf_mat_nor(y_predicted, y_true, n_classes=numCls)
        aa = get_AA(y_predicted, y_true, n_classes=numCls) # average accuracy, \
        # zero-sample classes are not excluded

        info = {
            "macroPrec" : macro_prec,
            "microPrec" : micro_prec,
            "samplePrec" : sample_prec,
            "macroRec" : macro_rec,
            "microRec" : micro_rec,
            "sampleRec" : sample_rec,
            "macroF1" : macro_f1,
            "microF1" : micro_f1,
            "sampleF1" : sample_f1,
            "macroF2" : macro_f2,
            "microF2" : micro_f2,
            "sampleF2" : sample_f2,
            "HammingLoss" : hamming_loss,
            "subsetAcc" : subset_acc,
            "macroAcc" : macro_acc,
            "microAcc" : micro_acc,
            "sampleAcc" : sample_acc,
            "oneError" : one_error,
            "coverageError" : coverage_error,
            "rankLoss" : rank_loss,
            "labelAvgPrec" : labelAvgPrec,
            "clsReport": cls_report,
            "conf_mat": conf_mat,
            "AverageAcc": aa }

    print("saving metrics...")
    pkl.dump(info, open("test_scores.pkl", "wb"))



if __name__ == "__main__":
    main()
