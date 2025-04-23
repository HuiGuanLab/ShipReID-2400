# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch.nn as nn
import torch

from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .token_triplet_loss import TokenTripletLoss
from .center_loss import CenterLoss
from .token_info_nce_loss import TokenInfoNCE
from .token_info_nce_loss_max import TokenInfoNCEMax

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if 'sum' == cfg.MODEL.TOKEN_CONTRAST_TYPE:
        token_contrast = TokenInfoNCE(cfg.SOLVER.INFO_NCE_T)
    elif 'max' == cfg.MODEL.TOKEN_CONTRAST_TYPE:
        token_contrast = TokenInfoNCEMax(cfg.SOLVER.INFO_NCE_T)
    elif 'triplet' == cfg.MODEL.TOKEN_CONTRAST_TYPE:
        if cfg.MODEL.NO_TOKEN_MARGIN:
            token_contrast = TokenTripletLoss()
        else:
            token_contrast = TokenTripletLoss(cfg.SOLVER.TOKEN_MARGIN) 
    
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, part_score, feat, part_feat, part_features, target, target_cam, M=None, i2tscore = None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(part_score, list):
                        PART_ID_LOSS = [xent(scor, target) for scor in part_score[0:]]
                        PART_ID_LOSS = sum(PART_ID_LOSS)
                    else:
                        PART_ID_LOSS = xent(part_score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS) 
                    else:   
                        TRI_LOSS = triplet(feat, feats, target)[0]
                    
                    if isinstance(part_feat, list):
                        PART_TRI_LOSS = [triplet(feats, feats, target)[0] for feats in part_feat[0:]]
                        PART_TRI_LOSS = sum(PART_TRI_LOSS) 
                    else:   
                        PART_TRI_LOSS = triplet(part_feat, part_feat, target)[0]
                    
                    TOKEN_CONTRAST_LOSS = [token_contrast(f, pf, target) for f, pf in list(zip(feat, part_features))]
                    TOKEN_CONTRAST_LOSS = sum(TOKEN_CONTRAST_LOSS)

                    ID_LOSS = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS
                    TRI_LOSS = cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    PART_ID_LOSS = cfg.MODEL.PART_ID_LOSS_WEIGHT * PART_ID_LOSS
                    PART_TRI_LOSS = cfg.MODEL.PART_TRIPLET_LOSS_WEIGHT * PART_TRI_LOSS
                    TOKEN_CONTRAST_LOSS = cfg.MODEL.TOKEN_CONTRAST_LOSS_WEIGHT * TOKEN_CONTRAST_LOSS 

                    loss = ID_LOSS + PART_ID_LOSS + TRI_LOSS + PART_TRI_LOSS + \
                           TOKEN_CONTRAST_LOSS

                    return loss, (ID_LOSS, PART_ID_LOSS, TRI_LOSS, PART_TRI_LOSS, TOKEN_CONTRAST_LOSS)
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                            TRI_LOSS = sum(TRI_LOSS)
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


