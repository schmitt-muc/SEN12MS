# Modified from Jian Kang, https://www.rsim.tu-berlin.de/menue/team/dring_jian_kang/
# Modified and extended by Yu-Lun Wu, TUM
""" 
metrics utilized for evaluating multi-label/ multi-class(single-label) classification system
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, fbeta_score, \
    classification_report, hamming_loss, accuracy_score, coverage_error, label_ranking_loss,\
    label_ranking_average_precision_score, confusion_matrix, multilabel_confusion_matrix



def conf_mat_nor(predict_labels, true_labels, n_classes):
    """ return the normalized confusion matrix (respect to y_true)
        input labels are in one-hot encoding, n_class = number of label classes
        This function only applied to single-label
    """
    
    assert (np.sum(true_labels, axis=1) == 1).all()
    assert (np.sum(predict_labels, axis=1) == 1).all()
    
    true_idx = np.where(true_labels == 1)[1]
    pred_idx = np.where(predict_labels == 1)[1]
    
    
    con = confusion_matrix(true_idx, pred_idx, labels=np.arange(n_classes))
    b = con.sum(axis=1)[:, None]
    con_nor = np.divide(con, b, where=(b != 0))
    
    return con_nor


def get_AA(predict_labels, true_labels, n_classes):
    """ only applied to single-label
        zero sample classes are not excluded in the calculation
        would be 0 in the calculation
    """
    con_nor = conf_mat_nor(predict_labels, true_labels, n_classes)
    AA = np.diagonal(con_nor).sum()/n_classes
    
    return AA


def multi_conf_mat(predict_labels, true_labels, n_classes):
    """ according to sklearn website, from the multi-label confusion matrix 
        we can calculate the classwise accuracy and average accuracy for 
        multi-label, but this seemed to be questionable and is not recommeded
    """
    mcon_mat = multilabel_confusion_matrix(true_labels, predict_labels, 
                                           labels=np.arange(n_classes))
    
    # calc. classwise accuracy
    n_samples = np.sum(mcon_mat[0,:,:])
    cls_acc = (mcon_mat[:,0,0] + mcon_mat[:,1,1])[:, None]/n_samples
    
    # average accuracy
    multilabel_aa = np.mean(cls_acc)

    return mcon_mat, cls_acc, multilabel_aa


def OA_multi(predict_labels, true_labels):
    """
    Overall accuracy for multi-label case 
    """
    num_true = np.sum((true_labels==1), axis=-1)
    num_pred = np.sum((predict_labels==1), axis=-1)
    num_cross = np.sum(np.logical_and((predict_labels==1), (true_labels==1)), axis=-1)
    
    acc = num_cross/(num_true + num_pred - num_cross)
    overall_acc = np.mean(acc)
    return overall_acc.astype(np.float32)




class MetricTracker(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        

class Precision_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        sample_prec = precision_score(true_labels, predict_labels, average='samples')
        micro_prec = precision_score(true_labels, predict_labels, average='micro')
        macro_prec = precision_score(true_labels, predict_labels, average='macro')

        return macro_prec, micro_prec, sample_prec    


class Recall_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        sample_rec = recall_score(true_labels, predict_labels, average='samples')
        micro_rec = recall_score(true_labels, predict_labels, average='micro')
        macro_rec = recall_score(true_labels, predict_labels, average='macro')

        return macro_rec, micro_rec, sample_rec


class F1_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        macro_f1 = f1_score(true_labels, predict_labels, average="macro")
        micro_f1 = f1_score(true_labels, predict_labels, average="micro")
        sample_f1 = f1_score(true_labels, predict_labels, average="samples")

        return macro_f1, micro_f1, sample_f1


class F2_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        macro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="macro")
        micro_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="micro")
        sample_f2 = fbeta_score(true_labels, predict_labels, beta=2, average="samples")

        return macro_f2, micro_f2, sample_f2

class Hamming_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        return hamming_loss(true_labels, predict_labels)

class Subset_accuracy(nn.Module):
    """ In multilabel classification, this function computes subset accuracy:
        exact match.
        In binary and multiclass classification, this function is equal to the
        jaccard_score function.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predict_labels, true_labels):

        return accuracy_score(true_labels, predict_labels)

class Accuracy_score(nn.Module):
    """ This Accuracy_score code is from the script before modification, it doesn't
        seem to be correct, thus was not used in the report
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, predict_labels, true_labels):

        # sample accuracy
        TP = (np.logical_and((predict_labels == 1), (true_labels == 1))).astype(int)
        union = (np.logical_or((predict_labels == 1), (true_labels == 1))).astype(int)
        TP_sample = TP.sum(axis=1)
        union_sample = union.sum(axis=1)

        sample_Acc = TP_sample/union_sample

        assert np.isfinite(sample_Acc).all(), 'Nan found in sample accuracy'

        FP = (np.logical_and((predict_labels == 1), (true_labels == 0))).astype(int)
        TN = (np.logical_and((predict_labels == 0), (true_labels == 0))).astype(int)
        FN = (np.logical_and((predict_labels == 0), (true_labels == 1))).astype(int)

        TP_cls = TP.sum(axis=0)
        FP_cls = FP.sum(axis=0)
        TN_cls = TN.sum(axis=0)
        FN_cls = FN.sum(axis=0)

        assert (TP_cls+FP_cls+TN_cls+FN_cls == predict_labels.shape[0]).all(), 'wrong'

        macro_Acc = np.mean((TP_cls + TN_cls) / (TP_cls + FP_cls + TN_cls + FN_cls))

        micro_Acc = (TP_cls.mean() + TN_cls.mean()) / (TP_cls.mean() + FP_cls.mean() + TN_cls.mean() + FN_cls.mean())

        return macro_Acc, micro_Acc, sample_Acc.mean()


class One_error(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        row_inds = np.arange(predict_probs.shape[0])
        col_inds = np.argmax(predict_probs, axis=1)
        return np.mean((true_labels[tuple(row_inds), tuple(col_inds)] == 0).astype(int))

class Coverage_error(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return coverage_error(true_labels, predict_probs)

class Ranking_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_loss(true_labels, predict_probs)

class LabelAvgPrec_score(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predict_probs, true_labels):
        return label_ranking_average_precision_score(true_labels, predict_probs)

class calssification_report(nn.Module):

    def __init__(self, target_names):
        super().__init__()
        self.target_names = target_names
    def forward(self, predict_labels, true_labels):

        report = classification_report(true_labels, predict_labels, target_names=self.target_names, output_dict=True)

        return report

if __name__ == "__main__":
    acc = Accuracy_score()

    aa = (np.random.randn(100,20)>=0).astype(int)

    bb = (np.random.randn(100,20)>=0).astype(int)

    samp_acc, macro_acc, micro_acc = acc(aa, bb)

    print(samp_acc)
    print(macro_acc)
    print(micro_acc)











