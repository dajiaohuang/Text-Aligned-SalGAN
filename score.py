from sklearn.metrics import roc_auc_score
import numpy as np
import torch

def AUC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1).detach().cpu()
    gt = gt.view(-1).detach().cpu()
    thres_gt = gt.sort()[0][int(0.8*len(gt))]
    thres_sal = saliency_mask.sort()[0][int(0.8*len(saliency_mask))]
    gt = (gt > thres_gt)
    saliency_mask = (saliency_mask > thres_sal).float()
    return roc_auc_score(gt.numpy(), saliency_mask.numpy())


def sAUC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1).detach().cpu()
    gt = gt.view(-1).detach().cpu()
    thres_gt = gt.sort()[0][int(0.8 * len(gt))]
    thres_sal = saliency_mask.sort()[0][int(0.8 * len(saliency_mask))]
    gt = (gt > thres_gt).numpy()
    saliency_mask = (saliency_mask > thres_sal).float().numpy()
    positive_indices = np.where(gt > 0)[0]
    negative_indices = np.where(gt == 0)[0]
    np.random.shuffle(negative_indices)
    shuffled_indices = np.concatenate([positive_indices, negative_indices[:len(positive_indices)]])
    shuffled_gt = np.zeros_like(gt)
    shuffled_gt[shuffled_indices] = 1

    return roc_auc_score(shuffled_gt, saliency_mask)

def CC(saliency_mask, gt):
    saliency_mask = saliency_mask.view(-1)
    gt = gt.view(-1)
    combined = torch.stack((saliency_mask, gt))
    return float(torch.corrcoef(combined)[0, 1])




def NSS(saliency_mask, gt):
    """
    计算 Normalized Scanpath Saliency (NSS)
    """
    saliency_mask = saliency_mask.view(-1)
    gt = gt.view(-1)
    saliency_mask = (saliency_mask - saliency_mask.mean()) / saliency_mask.std()
    gt = (gt > 0)
    nss_score = torch.sum(saliency_mask * gt) / torch.sum(gt)

    return float(nss_score)